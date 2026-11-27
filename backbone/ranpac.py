# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
# Follows the same conventions as backbone/LEAR.py, but note RanPAC's classifier is NOT
# trained by gradient descent — it is solved in closed form (ridge regression) at the end
# of every domain. See models/ranpac.py for the observe()/end_task() logic that fills in
# `self.G[i]` / `self.Q[i]` and then calls `solve_ridge(i)`.
#
# Original method: RanPAC: Random Projections and Pre-trained Models for Continual
# Learning, McDonnell et al., NeurIPS 2023.
#
# SIMPLIFICATION vs. the original paper: we skip the optional "first-session adaptation"
# (briefly fine-tuning the backbone on the first domain) and keep the ViT fully frozen
# throughout, i.e. this is the "RanPAC w/o FSA" variant. The random-projection + ridge
# -regression classifier itself follows the paper.

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class RanPAC(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(RanPAC, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768
        self.proj_dim = 4000   # random-feature dimension M (paper default order of magnitude)
        self.ridge_lambda = 1e2

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # fixed (non-trainable) random projection, shared across all domains
        self.register_buffer('proj_W', torch.randn(self.embed_dim, self.proj_dim) / (self.embed_dim ** 0.5))

        # per-domain accumulated Gram/cross-correlation matrices for ridge regression
        # (kept as plain python attributes, NOT nn.Parameter — no gradient involved)
        self.G = [torch.zeros(self.proj_dim, self.proj_dim, device=self.device)]
        self.Q = [torch.zeros(self.proj_dim, num_classes, device=self.device)]
        self.W = [torch.zeros(self.proj_dim, num_classes, device=self.device)]  # solved classifier weight

        self.distributions = []
        self.c_expert = 0

    def to(self, device, **kwargs):
        self.device = device
        self.G = [g.to(device) for g in self.G]
        self.Q = [q.to(device) for q in self.Q]
        self.W = [w.to(device) for w in self.W]
        return super().to(device, **kwargs)

    @torch.no_grad()
    def random_features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen ViT CLS feature -> fixed random projection -> ReLU (RanPAC's Phi(.))."""
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        feat = self.vit.forward_features(processed_x)[:, 0, :]
        return F.relu(feat @ self.proj_W)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        phi = self.random_features(x)
        out = phi @ self.W[self.c_expert]
        if return_features:
            return out, phi
        return out

    @torch.no_grad()
    def accumulate_stats(self, phi: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """Accumulate G = Phi^T Phi and Q = Phi^T Y for the CURRENT domain."""
        y_onehot = F.one_hot(labels, num_classes=num_classes).float()
        self.G[self.c_expert] += phi.T @ phi
        self.Q[self.c_expert] += phi.T @ y_onehot

    def solve_ridge(self):
        """Closed-form ridge regression for the current domain's classifier weight."""
        M = self.proj_dim
        I = torch.eye(M, device=self.device)
        self.W[self.c_expert] = torch.linalg.solve(self.G[self.c_expert] + self.ridge_lambda * I,
                                                     self.Q[self.c_expert])

    def CreateNewExper(self, idx: int, num_classes: int) -> None:
        self.G.append(torch.zeros(self.proj_dim, self.proj_dim, device=self.device))
        self.Q.append(torch.zeros(self.proj_dim, num_classes, device=self.device))
        self.W.append(torch.zeros(self.proj_dim, num_classes, device=self.device))
        self.c_expert = len(self.W) - 1
        print(f'[backbone/ranpac] created ridge-regression head for domain {self.c_expert + 1} '
              f'({num_classes} classes); nearest previous domain was {idx + 1}')

    def myprediction(self, x: torch.Tensor, index: int) -> torch.Tensor:
        with torch.no_grad():
            phi = self.random_features(x)
            return phi @ self.W[index]


@register_backbone("ranpac")
def RanPAC_backbone(num_classes):
    return RanPAC(num_classes)