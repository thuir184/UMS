# Adapted for MDIL. Pair with backbone/ranpac.py, run with --model ranpac --backbone ranpac.
#
# Unlike L2P/DualPrompt/CODA-Prompt, RanPAC's classifier is solved in closed form
# (ridge regression), not by gradient descent — so `observe()` just accumulates
# statistics, and `end_task()` calls `self.net.solve_ridge()`.

import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class Ranpac(ContinualModel):
    NAME = 'ranpac'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--ranpac_signature_batches', type=int, default=100,
                             help='#batches used to fit each domain\'s feature Gaussian.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Ranpac, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self._current_num_classes = None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        # no gradient-based training: the ViT and random projection are frozen, the
        # classifier is solved in closed form at end_task(); here we just accumulate
        # the sufficient statistics for that closed-form solution.
        with torch.no_grad():
            phi = self.net.random_features(inputs)
            self.net.accumulate_stats(phi, labels, self._current_num_classes)
        return 0.0

    def myPrediction(self, x, k):
        with torch.no_grad():
            return self.net.myprediction(x, k)

    @torch.no_grad()
    def cal_expert_dist(self, x):
        phi = self.net.random_features(x)
        distances = []
        for dist in self.net.distributions:
            delta = phi - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt((delta @ inv_cov @ delta.T).diagonal().clamp(min=0))
            distances.append(mahalanobis.mean().item())
        return distances

    @torch.no_grad()
    def _fit_domain_signature(self, dataset, num_batches):
        train_iter = iter(dataset.train_loader)
        feats = []
        pbar = tqdm(range(num_batches),
                    desc=f"[ranpac] fitting domain signature for domain {self.current_task + 1}",
                    disable=False, mininterval=0.5)
        for _ in pbar:
            try:
                data = next(train_iter)
            except StopIteration:
                break
            x = data[0].to(self.device)
            feats.append(self.net.random_features(x))
            pbar.update()
        pbar.close()
        feats = torch.cat(feats, dim=0)
        mu = feats.mean(dim=0)
        # random-projected features live in a much higher-dim space (proj_dim, e.g. 4000);
        # add a slightly larger ridge term for numerical stability of the covariance inverse
        sigma = torch.cov(feats.T) + 1e-2 * torch.eye(feats.shape[1], device=feats.device)
        return MultivariateNormal(mu, sigma)

    def begin_task(self, dataset) -> None:
        self._current_num_classes = dataset.N_CLASSES
        if self.current_task > 0:
            distances = self._cal_expert_dist_from_loader(dataset.train_loader, num_batches=50)
            min_idx = int(torch.tensor(distances).argmin().item())
            self.net.CreateNewExper(min_idx, dataset.N_CLASSES)
        # RanPAC has no trainable parameters (ViT + random projection are frozen, the
        # classifier is closed-form) -> get_optimizer() will safely return None here.
        self.opt = self.get_optimizer()

    def end_task(self, dataset) -> None:
        self.net.solve_ridge()
        dist = self._fit_domain_signature(dataset, self.args.ranpac_signature_batches)
        self.net.distributions.append(dist)

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