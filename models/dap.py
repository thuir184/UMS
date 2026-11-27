# Adapted for MDIL. Pair with backbone/dap.py, run with --model dap --backbone dap.
# Interface identical to models/l2p.py.

import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class Dap(ContinualModel):
    NAME = 'dap'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--dap_signature_batches', type=int, default=100,
                             help='#batches used to fit each domain\'s feature Gaussian.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Dap, self).__init__(backbone, loss, args, transform, dataset=dataset)

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
        distances = []
        for expert_idx, dist in enumerate(self.net.distributions):
            feat = self.net.forward_with_prompt(x, expert_idx=expert_idx)
            delta = feat - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt((delta @ inv_cov @ delta.T).diagonal().clamp(min=0))
            distances.append(mahalanobis.mean().item())
        return distances

    @torch.no_grad()
    def _fit_domain_signature(self, dataset, num_batches):
        train_iter = iter(dataset.train_loader)
        feats = []
        pbar = tqdm(range(num_batches),
                    desc=f"[dap] fitting domain signature for domain {self.current_task + 1}",
                    disable=False, mininterval=0.5)
        for _ in pbar:
            try:
                data = next(train_iter)
            except StopIteration:
                break
            x = data[0].to(self.device)
            feats.append(self.net.forward_with_prompt(x, expert_idx=self.net.c_expert))
            pbar.update()
        pbar.close()
        feats = torch.cat(feats, dim=0)
        mu = feats.mean(dim=0)
        sigma = torch.cov(feats.T) + 1e-4 * torch.eye(feats.shape[1], device=feats.device)
        return MultivariateNormal(mu, sigma)

    def begin_task(self, dataset) -> None:
        if self.current_task > 0:
            distances = self._cal_expert_dist_from_loader(dataset.train_loader, num_batches=50)
            min_idx = int(torch.tensor(distances).argmin().item())
            self.net.CreateNewExper(min_idx, dataset.N_CLASSES)
        self.opt = self.get_optimizer()

    def end_task(self, dataset) -> None:
        dist = self._fit_domain_signature(dataset, self.args.dap_signature_batches)
        self.net.distributions.append(dist)
        for p in self.net.classifierArr[self.net.c_expert].parameters():
            p.requires_grad = False
        for p in self.net.generators[self.net.c_expert].parameters():
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