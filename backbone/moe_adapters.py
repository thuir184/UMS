# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
# Follows the same conventions as backbone/LEAR.py.
#
# Original method: Boosting Continual Learning of Vision-Language Models via
# Mixture-of-Experts Adapters (MoE-Adapters), Yu et al., CVPR 2024. The official method
# uses a CLIP backbone (image+text encoders) with per-layer MoE adapters and an
# autonomous router.
#
# SIMPLIFICATION vs. the original paper: to stay consistent with the other 5 files in
# this set (and avoid pulling in a full CLIP text encoder + contrastive loss), we use the
# same frozen ViT-B/16 image encoder as the other backbones, with ONE bottleneck adapter
# expert per domain applied to the pooled CLS feature (rather than per-layer routing
# throughout the network), plus a genuinely-learned router (see below). This keeps
# MoE-Adapters' defining idea — a growing set of experts selected by a learned,
# input-conditioned router, with NO task-id oracle at test time — while dropping the
# CLIP/text-encoder/per-layer-routing specifics.

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class Adapter(nn.Module):
    """Bottleneck adapter: down-project -> GELU -> up-project, added residually."""

    def __init__(self, embed_dim: int = 768, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class MoeAdapters(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(MoeAdapters, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # one adapter expert per domain, grown via CreateNewExper
        self.adapters = nn.ModuleList([Adapter(self.embed_dim).to(self.device)])
        # router: one binary "does this feature belong to domain i" head per domain,
        # concatenated at inference to form a growing multi-class router — trained with
        # the ground-truth domain id (known at train time) as an auxiliary task.
        self.router_heads = nn.ModuleList([nn.Linear(self.embed_dim, 1, device=self.device)])

        self.classifierArr = nn.ModuleList([nn.Linear(self.embed_dim, num_classes, device=self.device)])
        self.distributions = []  # kept for interface parity; MoE-Adapters routes via `router_heads` instead
        self.c_expert = 0

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    @torch.no_grad()
    def _frozen_feat(self, processed_x):
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        return self.vit.forward_features(processed_x)[:, 0, :]

    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """[B, num_domains_so_far] logits used both as an auxiliary training signal
        (against the ground-truth domain id) and, at test time, to pick an expert
        without any task-id oracle."""
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        feat = self._frozen_feat(processed_x)
        return torch.cat([h(feat) for h in self.router_heads], dim=1)

    def forward_with_adapter(self, x: torch.Tensor, expert_idx: int = None):
        expert_idx = self.c_expert if expert_idx is None else expert_idx
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        feat = self._frozen_feat(processed_x)
        return self.adapters[expert_idx](feat)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.forward_with_adapter(x)
        out = self.classifierArr[self.c_expert](feat)
        router_logits = self.router_logits(x)
        if return_features:
            return out, router_logits, feat
        return out, router_logits

    def CreateNewExper(self, idx: int, num_classes: int) -> None:
        import copy
        new_adapter = copy.deepcopy(self.adapters[idx])  # init from nearest domain's adapter
        self.adapters.append(new_adapter)
        self.router_heads.append(nn.Linear(self.embed_dim, 1, device=self.device))
        new_head = nn.Linear(self.embed_dim, num_classes, device=self.device)
        self.classifierArr.append(new_head)
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/moe_adapters] created adapter+router-head+classifier for domain '
              f'{self.c_expert + 1} ({num_classes} classes), adapter cloned from nearest domain {idx + 1}')

    def myprediction(self, x: torch.Tensor, index: int) -> torch.Tensor:
        with torch.no_grad():
            feat = self.forward_with_adapter(x, expert_idx=index)
            return self.classifierArr[index](feat)


@register_backbone("moe_adapters")
def MoeAdapters_backbone(num_classes):
    return MoeAdapters(num_classes)