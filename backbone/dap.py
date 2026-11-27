# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
# Follows the same conventions as backbone/LEAR.py.
#
# Original method: Generating Instance-level Prompts for Rehearsal-free Continual
# Learning (DAP), Jung et al., ICCV 2023. DAP generates a prompt PER INSTANCE (conditioned
# on that instance's own frozen features) rather than selecting from a fixed pool — this
# is precisely the "domain-adaptive" idea this file keeps.
#
# SIMPLIFICATION vs. the original paper: the official DAP generator is a small
# transformer/MLP mixing multiple intermediate ViT features; here the generator is a
# single 2-layer MLP conditioned on the frozen CLS feature only (still instance-level and
# still domain-adaptive, but architecturally lighter).

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class DAPGenerator(nn.Module):
    """Maps a per-instance frozen CLS feature to a small set of prompt tokens."""

    def __init__(self, embed_dim: int = 768, prompt_len: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_len * embed_dim),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        b = query.shape[0]
        return self.net(query).view(b, self.prompt_len, self.embed_dim)


class DAP(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(DAP, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # one instance-conditioned prompt generator per domain, grown via CreateNewExper
        self.generators = nn.ModuleList([DAPGenerator(embed_dim=self.embed_dim).to(self.device)])

        self.classifierArr = nn.ModuleList([nn.Linear(self.embed_dim, num_classes, device=self.device)])
        self.distributions = []
        self.c_expert = 0

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    @torch.no_grad()
    def _frozen_query(self, processed_x):
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        return self.vit.forward_features(processed_x)[:, 0, :]

    def forward_with_prompt(self, x: torch.Tensor, expert_idx: int = None):
        expert_idx = self.c_expert if expert_idx is None else expert_idx
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        query = self._frozen_query(processed_x)
        instance_prompt = self.generators[expert_idx](query)  # [B, prompt_len, embed_dim]

        tokens = self.vit.patch_embed(processed_x)
        cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.vit.pos_embed
        tokens = torch.cat((tokens[:, :1, :], instance_prompt, tokens[:, 1:, :]), dim=1)

        for block in self.vit.blocks:
            tokens = block(tokens)
        tokens = self.vit.norm(tokens)
        return tokens[:, 0, :]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.forward_with_prompt(x)
        out = self.classifierArr[self.c_expert](feat)
        if return_features:
            return out, feat
        return out

    def CreateNewExper(self, idx: int, num_classes: int) -> None:
        """Init the new domain's generator by cloning the nearest previous domain's
        weights — DAP's paper motivates transfer from the most related past domain."""
        import copy
        new_generator = copy.deepcopy(self.generators[idx])
        self.generators.append(new_generator)
        new_head = nn.Linear(self.embed_dim, num_classes, device=self.device)
        self.classifierArr.append(new_head)
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/dap] created generator+head for domain {self.c_expert + 1} '
              f'({num_classes} classes), cloned from nearest domain {idx + 1}')

    def myprediction(self, x: torch.Tensor, index: int) -> torch.Tensor:
        with torch.no_grad():
            feat = self.forward_with_prompt(x, expert_idx=index)
            return self.classifierArr[index](feat)


@register_backbone("dap")
def DAP_backbone(num_classes):
    return DAP(num_classes)