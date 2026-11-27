# Ported from the official SOYO-main/methods/soyo.py (`_train`, `train_function`,
# `compress_train_resample`, `_eval_cnn`) and utils/soyo_utils.py (`feature_compression`,
# `random_sampling`, `soyo_network`), adapted for MDIL. Pair with backbone/soyo.py, run
# with --model soyo --backbone soyo.
#
# NEW DEPENDENCY: `pip install scikit-learn` (the official GMC uses
# `sklearn.mixture.GaussianMixture` with full covariance — kept exactly, rather than a
# hand-rolled diagonal-covariance EM).
#
# BUGFIX (previous version underfit badly, ~10% on CIFAR-100 domain 0): the official
# code trains its OWN SGD(momentum=0.9) + CosineAnnealingLR schedule *per domain*
# (30 epochs, lr=0.01, weight_decay=2e-4 — see configs/domainnet_soyo_vit.yaml), managed
# entirely inside `_train`/`train_function`. The previous port just used whatever generic
# optimizer/LR the outer harness happened to be configured with, and applied NO learning
# rate decay at all — with SOYO injecting a bottleneck adapter at every one of the 12
# ViT blocks (a much bigger perturbation than the other, prompt-only, integrated
# methods), that is enough to plateau well short of convergence, which matches exactly
# the "loss drops a lot but not deep enough" symptom. This version builds its own
# SGD+momentum+CosineAnnealingLR optimizer in `begin_task` (ignoring the harness's
# generic `--optimizer`/`--lr`) and steps the scheduler once per epoch via the
# `end_epoch` hook (`utils/training_domain.py` already calls `model.end_epoch(epoch,
# dataset)` once per epoch — the previous version simply never used it). You still
# control training LENGTH via the harness's own `--n_epochs` (there is no per-model way
# to override how many epochs the outer loop runs); the official config uses 30 epochs
# per domain — pass `--n_epochs 30 --batch_size 128` to match it.
#
# Kept FAITHFUL to the official code:
#  - GMC: `sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')` fit
#    on the domain's real (frozen, prompt-free) features at the end of every domain.
#  - DFR: synthetic features are RESAMPLED from every stored past-domain GMM and mixed
#    with the CURRENT domain's real features in every batch (`num_new` real + `num_rs`
#    resampled, proportioned by `1/(k+1)` as in the official code) to train the SOYO
#    domain classifier over several epochs — not a one-shot closed-form fit. Its own
#    SGD(momentum=0.9)+CosineAnnealingLR schedule (`soyo_epoch=30, soyo_lr=0.1,
#    soyo_weight_decay=2e-4`, from the same official config) is unaffected by this bugfix
#    — it was already implemented as its own self-contained loop.
#  - domain classifier: official is a single `nn.Linear(dim, total_sessions)`; here it is
#    a growing `nn.ModuleList` of per-domain heads (this repo's standard growth pattern),
#    which is functionally identical once concatenated.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class Soyo(ContinualModel):
    NAME = 'soyo'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--soyo_gmm_components', type=int, default=2, help='Official default: 2.')
        parser.add_argument('--soyo_soyo_epoch', type=int, default=30,
                             help='#epochs to train the SOYO domain classifier (official: `soyo_epoch`=30).')
        parser.add_argument('--soyo_soyo_lr', type=float, default=0.1, help='Official: `soyo_lr`=0.1.')
        parser.add_argument('--soyo_soyo_weight_decay', type=float, default=2e-4, help='Official: `soyo_weight_decay`=2e-4.')
        parser.add_argument('--soyo_lr', type=float, default=0.01,
                             help='Main-network LR (official `init_lr`/`lr`, both 0.01 in configs/domainnet_soyo_vit.yaml). '
                                  'Overrides the harness\'s generic --lr for this model.')
        parser.add_argument('--soyo_weight_decay', type=float, default=2e-4, help='Official `init_weight_decay`/`weight_decay`.')
        parser.add_argument('--soyo_momentum', type=float, default=0.9, help='Official SGD momentum.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Soyo, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.n_components = args.soyo_gmm_components
        self.scheduler = None

    # ------------------------------------------------------------------ train ----
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss_ce = self.loss(outputs, labels)
        loss_ce.backward()
        self.opt.step()
        return loss_ce.item()

    def myPrediction(self, x, k):
        with torch.no_grad():
            return self.net.myprediction(x, k)

    @torch.no_grad()
    def cal_expert_dist(self, x):
        """Uses the TRAINED SOYO domain classifier (`_retrain_domain_classifier`, the
        DFR step), not a raw distance — matches the official `_eval_cnn`'s
        `domain_logits = self.soyo_network(feature)` routing."""
        feat = self.net.extract_vector(x)
        logits = torch.cat([h(feat) for h in self.domain_heads], dim=1)
        neg_log_probs = -F.log_softmax(logits, dim=1).mean(dim=0)
        return neg_log_probs.tolist()

    # ------------------------------------------------------------ GMC (feature_compression) ----
    @torch.no_grad()
    def _collect_domain_features(self, dataset):
        train_iter = iter(dataset.train_loader)
        feats = []
        pbar = tqdm(desc=f"[soyo] collecting features for domain {self.current_task + 1} (GMC)",
                    disable=False, mininterval=0.5)
        for data in train_iter:
            x = data[0].to(self.device)
            feats.append(self.net.extract_vector(x).cpu())
            pbar.update()
        pbar.close()
        return torch.cat(feats, dim=0).numpy()

    def _fit_gmm(self, features_np):
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full',
                               init_params='kmeans')
        gmm.fit(features_np)
        return gmm.means_, gmm.covariances_, gmm.weights_

    @staticmethod
    def _resample_from_gmm(n_samples, compression, n_components):
        means, covariances, weights = compression
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.means_, gmm.covariances_, gmm.weights_ = means, covariances, weights
        # sklearn's GaussianMixture requires `precisions_cholesky_` to sample; fit() sets
        # it, but we only have stored params, so recompute it (official code does the
        # same implicitly by re-instantiating a GMM with these params before .sample()).
        gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(covariances)).transpose(0, 2, 1)
        return gmm.sample(n_samples)[0]

    # ------------------------------------------------------------------ DFR ----
    def _retrain_domain_classifier(self, dataset):
        """Official `compress_train_resample`: mix real current-domain features with
        resampled features from every stored GMM, train the domain classifier for
        `soyo_epoch` epochs."""
        k_old = len(self.net.gmms)
        embed_dim = self.net.embed_dim
        self.domain_heads = nn.ModuleList(
            [nn.Linear(embed_dim, 1, device=self.device) for _ in range(k_old + 1)])

        rs_features, rs_targets = None, None
        if k_old > 0:
            n_samples = len(dataset.train_loader.dataset)
            rs_feats, rs_tgts = [], []
            for domain_idx, compression in enumerate(self.net.gmms):
                rs_feats.append(self._resample_from_gmm(n_samples, compression, self.n_components))
                rs_tgts.append(np.full((n_samples,), domain_idx))
            rs_features = torch.tensor(np.vstack(rs_feats), dtype=torch.float32, device=self.device)
            rs_targets = torch.tensor(np.hstack(rs_tgts), dtype=torch.long, device=self.device)

        batch_size = dataset.train_loader.batch_size or 32
        num_new = max(1, batch_size // (1 + k_old))
        num_rs = batch_size - num_new

        opt = torch.optim.SGD(self.domain_heads.parameters(), momentum=0.9,
                               lr=self.args.soyo_soyo_lr, weight_decay=self.args.soyo_soyo_weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.soyo_soyo_epoch)

        pbar = tqdm(range(self.args.soyo_soyo_epoch), desc="[soyo] training domain classifier (DFR)",
                    disable=False, mininterval=0.5)
        for _ in pbar:
            for data in dataset.train_loader:
                x = data[0].to(self.device)
                with torch.no_grad():
                    feat = self.net.extract_vector(x)
                domain_target = torch.full((feat.shape[0],), self.current_task,
                                            dtype=torch.long, device=self.device)
                if k_old > 0:
                    sel_new = torch.randperm(feat.shape[0])[:num_new]
                    sel_rs = torch.randperm(rs_features.shape[0])[:num_rs]
                    final_feat = torch.cat([feat[sel_new], rs_features[sel_rs]], dim=0)
                    final_target = torch.cat([domain_target[sel_new], rs_targets[sel_rs]], dim=0)
                else:
                    final_feat, final_target = feat, domain_target
                logits = torch.cat([h(final_feat) for h in self.domain_heads], dim=1)
                loss = F.cross_entropy(logits, final_target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            pbar.update()
        pbar.close()

    # ------------------------------------------------------------- task hooks ----
    def begin_task(self, dataset) -> None:
        if self.current_task > 0:
            # official SOYO always initializes fresh parameters for a new domain (no
            # cloning from a "nearest" one); `idx=0` is passed only for logging parity
            self.net.CreateNewExper(0, dataset.N_CLASSES)

        # official: SGD(momentum=0.9, lr=init_lr/lr, weight_decay=init_weight_decay/weight_decay)
        # + CosineAnnealingLR(T_max=init_epoch/epochs) — built fresh per domain, exactly
        # like the official `_train`. NOTE: this overrides the harness's generic
        # --optimizer/--lr for this model; training LENGTH is still whatever the harness
        # runs (--n_epochs) since only the outer loop controls how many times `observe`
        # is called — pass --n_epochs 30 to match the official config.
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        self.opt = torch.optim.SGD(trainable, lr=self.args.soyo_lr, momentum=self.args.soyo_momentum,
                                    weight_decay=self.args.soyo_weight_decay)
        t_max = max(getattr(self.args, 'n_epochs', 30) or 30, 1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=t_max)

    def end_epoch(self, epoch, dataset) -> None:
        """Official `train_function`: `scheduler.step()` once at the end of every epoch —
        the previous version never overrode this hook, so the LR never decayed."""
        if self.scheduler is not None:
            self.scheduler.step()

    def end_task(self, dataset) -> None:
        for p in self.net.classifierArr[self.net.c_expert].parameters():
            p.requires_grad = False
        for p in self.net.down_pool[self.net.c_expert].parameters():
            p.requires_grad = False
        for p in self.net.up_pool[self.net.c_expert].parameters():
            p.requires_grad = False
        for p in self.net.prompt_pool[self.net.c_expert].parameters():
            p.requires_grad = False

        features_np = self._collect_domain_features(dataset)          # GMC step 1: collect
        compression = self._fit_gmm(features_np)                      # GMC step 2: compress
        self.net.gmms.append(compression)
        self._retrain_domain_classifier(dataset)                      # DFR: resample + retrain