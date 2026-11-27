# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
#
# This file must be paired with backbone/l2p.py and dataset --backbone l2p.
#
# Original method: Learning to Prompt for Continual Learning (L2P), Wang et al., CVPR 2022.
#
# How this plugs into utils/training_domain.py:
#   - COMPATIBILITY includes 'class-il' -> evaluate() will route through
#     `model.cal_expert_dist(inputs)` (to guess which domain/head a test batch belongs to,
#     WITHOUT any task-id oracle) and `model.myPrediction(inputs, expert_idx)` for the
#     actual prediction — the exact same contract used by models/LEAR.py.
#   - `begin_task`/`end_task` follow the same pattern as models/LEAR.py: at the end of a
#     domain we fit a per-domain Gaussian over the (frozen ViT + prompt-pool) CLS feature;
#     at the start of the next domain we use Mahalanobis distance to that Gaussian to decide
#     whether/how to grow a new head.

import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class L2p(ContinualModel):
    NAME = 'l2p'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--l2p_pull_weight', type=float, default=0.5,
                             help='Weight of the L2P prompt-pool pull-constraint loss.')
        parser.add_argument('--l2p_signature_batches', type=int, default=100,
                             help='#batches used to fit each domain\'s feature Gaussian '
                                  '(used for expert/domain selection at test time).')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(L2p, self).__init__(backbone, loss, args, transform, dataset=dataset)

    # ------------------------------------------------------------------ train ----
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs, pull_loss = self.net(inputs)
        loss_ce = self.loss(outputs, labels)
        loss_tot = loss_ce + self.args.l2p_pull_weight * pull_loss
        loss_tot.backward()
        self.opt.step()
        return [loss_ce.item(), pull_loss.item()]

    # --------------------------------------------------------- domain routing ----
    def myPrediction(self, x, k):
        """Called by utils/training_domain.py:evaluate() with the *guessed* domain index."""
        with torch.no_grad():
            return self.net.myprediction(x, k)

    @torch.no_grad()
    def cal_expert_dist(self, x):
        """
        Called by utils/training_domain.py:evaluate() to choose, for each test loader,
        which domain/head best matches a handful of its samples — no task-id involved.
        Returns one distance per domain seen so far (lower = more likely match), exactly
        like models/LEAR.py:cal_expert_dist.
        """
        feat, _ = self.net.forward_with_prompt(x)
        distances = []
        for dist in self.net.distributions:
            delta = feat - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt((delta @ inv_cov @ delta.T).diagonal().clamp(min=0))
            distances.append(mahalanobis.mean().item())
        return distances

    @torch.no_grad()
    def _fit_domain_signature(self, dataset, num_batches):
        """Fit a Gaussian over the current domain's (frozen ViT + prompt-pool) CLS feature."""
        train_loader = dataset.train_loader
        train_iter = iter(train_loader)
        feats = []
        pbar = tqdm(range(num_batches),
                    desc=f"[l2p] fitting domain signature for domain {self.current_task + 1}",
                    disable=False, mininterval=0.5)
        for _ in pbar:
            try:
                data = next(train_iter)
            except StopIteration:
                break
            x = data[0].to(self.device)
            feat, _ = self.net.forward_with_prompt(x)
            feats.append(feat)
            pbar.update()
        pbar.close()
        feats = torch.cat(feats, dim=0)
        mu = feats.mean(dim=0)
        # small ridge term for numerical stability of the covariance inverse
        sigma = torch.cov(feats.T) + 1e-4 * torch.eye(feats.shape[1], device=feats.device)
        return MultivariateNormal(mu, sigma)

    # ------------------------------------------------------------- task hooks ----
    def begin_task(self, dataset) -> None:
        if self.current_task > 0:
            distances = self._cal_expert_dist_from_loader(dataset.train_loader, num_batches=50)
            min_idx = int(torch.tensor(distances).argmin().item())
            self.net.CreateNewExper(min_idx, dataset.N_CLASSES)
        self.opt = self.get_optimizer()

    def end_task(self, dataset) -> None:
        dist = self._fit_domain_signature(dataset, self.args.l2p_signature_batches)
        self.net.distributions.append(dist)
        # freeze the head we just trained; the prompt pool stays trainable (shared/global,
        # exactly as in vanilla L2P)
        for p in self.net.classifierArr[self.net.c_expert].parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _cal_expert_dist_from_loader(self, loader, num_batches=50):
        train_iter = iter(loader)
        sums = [0.0] * len(self.net.distributions)
        count = 0
        for _ in range(num_batches):
            try:
                data = next(train_iter)
            except StopIteration:
                break
            x = data[0].to(self.device)
            d = self.cal_expert_dist(x)
            sums = [s + v for s, v in zip(sums, d)]
            count += 1
        return [s / max(count, 1) for s in sums]