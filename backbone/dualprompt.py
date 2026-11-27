# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
# Follows the same conventions as backbone/LEAR.py (MammothBackbone, classifierArr,
# CreateNewExper, myprediction, distributions).
#
# Original method: DualPrompt: Complementary Prompting for Rehearsal-free Continual
# Learning, Wang et al., ECCV 2022.
#
# SIMPLIFICATION vs. the original paper: DualPrompt inserts the "general" prompt at
# transformer layers 1-2 and the "expert" prompt at layers 3-5 (prefix-tuning style,
# injected mid-network). For implementation robustness within this codebase we instead
# prepend BOTH prompt groups as extra tokens right after the CLS token, before block 0
# (same insertion point as backbone/l2p.py). This keeps the general-vs-expert split (the
# core idea of the method) but not the exact per-layer placement. If you need the exact
# per-layer placement, port it from the official DualPrompt / Mammoth implementation.

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class DualPrompt(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(DualPrompt, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768
        self.g_len = 5    # general prompt length
        self.e_len = 20   # expert (per-domain) prompt length

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # general prompt: shared across ALL domains, always trainable
        self.general_prompt = nn.Parameter(torch.randn(self.g_len, self.embed_dim, device=self.device) * 0.02)

        # expert prompt: one per domain, grown via CreateNewExper
        self.expert_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(self.e_len, self.embed_dim, device=self.device) * 0.02)
        ])

        self.classifierArr = nn.ModuleList([nn.Linear(self.embed_dim, num_classes, device=self.device)])
        self.distributions = []
        self.c_expert = 0

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def forward_with_prompt(self, x: torch.Tensor, expert_idx: int = None):
        expert_idx = self.c_expert if expert_idx is None else expert_idx
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)

        tokens = self.vit.patch_embed(processed_x)
        cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.vit.pos_embed

        b = tokens.shape[0]
        gen = self.general_prompt.unsqueeze(0).expand(b, -1, -1)
        exp = self.expert_prompts[expert_idx].unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat((tokens[:, :1, :], gen, exp, tokens[:, 1:, :]), dim=1)

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
        """Init the new domain's expert prompt by cloning the nearest previous domain's."""
        new_prompt = nn.Parameter(self.expert_prompts[idx].detach().clone())
        self.expert_prompts.append(new_prompt)
        new_head = nn.Linear(self.embed_dim, num_classes, device=self.device)
        self.classifierArr.append(new_head)
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/dualprompt] created head+expert-prompt for domain {self.c_expert + 1} '
              f'({num_classes} classes), cloned from nearest domain {idx + 1}')

    def myprediction(self, x: torch.Tensor, index: int) -> torch.Tensor:
        with torch.no_grad():
            feat = self.forward_with_prompt(x, expert_idx=index)
            return self.classifierArr[index](feat)


@register_backbone("dualprompt")
def DualPrompt_backbone(num_classes):
    return DualPrompt(num_classes)