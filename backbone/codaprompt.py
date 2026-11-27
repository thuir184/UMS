# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
# Follows the same conventions as backbone/LEAR.py.
#
# Original method: CODA-Prompt: COntinual Decomposed Attention-based Prompting for
# Rehearsal-Free Continual Learning, Smith et al., CVPR 2023.
#
# SIMPLIFICATION vs. the original paper: the official CODA-Prompt uses a learnable
# attention vector A_k per component together with an orthogonality-inducing loss on the
# component pool. Here the pool is combined with a plain softmax-attention over
# cosine-similarity to the (frozen ViT) query, WITHOUT the orthogonality regularizer, to
# keep the file self-contained and robust. This keeps CODA-Prompt's core idea — a smooth,
# differentiable, task-agnostic combination of prompt components, as opposed to L2P's hard
# top-k selection — but drops the exact orthogonality loss term. Add it back
# (`sum(|| P_i^T P_j ||_F for i != j)`) if you need to match the paper's numbers exactly.

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class CodaComponentPool(nn.Module):
    """
    `n_components` learnable (key, attention-vector, prompt) triplets, shared/global and
    continually trained across ALL domains (task-agnostic by construction, no domain id
    needed at train OR test time).
    """

    def __init__(self, embed_dim: int = 768, n_components: int = 100, prompt_len: int = 8):
        super().__init__()
        self.n_components = n_components
        self.prompt_len = prompt_len
        self.keys = nn.Parameter(torch.randn(n_components, embed_dim) * 0.02)
        self.attn = nn.Parameter(torch.randn(n_components, embed_dim) * 0.02)
        self.prompts = nn.Parameter(torch.randn(n_components, prompt_len, embed_dim) * 0.02)

    def forward(self, query: torch.Tensor):
        """
        Args:
            query: [B, embed_dim] CLS feature from the frozen ViT.
        Returns:
            combined_prompt: [B, prompt_len, embed_dim]
        """
        q = query.unsqueeze(1) * self.attn.unsqueeze(0)          # [B, n_components, embed_dim]
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.keys, dim=-1).unsqueeze(0)          # [1, n_components, embed_dim]
        weights = F.softmax((q * k).sum(-1) / (self.attn.shape[-1] ** 0.5), dim=-1)  # [B, n_components]
        combined_prompt = torch.einsum('bn,nld->bld', weights, self.prompts)  # [B, prompt_len, embed_dim]
        return combined_prompt


class CodaPrompt(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(CodaPrompt, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.component_pool = CodaComponentPool(embed_dim=self.embed_dim)

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

    def forward_with_prompt(self, x: torch.Tensor):
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        query = self._frozen_query(processed_x)
        combined_prompt = self.component_pool(query)

        tokens = self.vit.patch_embed(processed_x)
        cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.vit.pos_embed
        tokens = torch.cat((tokens[:, :1, :], combined_prompt, tokens[:, 1:, :]), dim=1)

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
        new_head = nn.Linear(self.embed_dim, num_classes, device=self.device)
        self.classifierArr.append(new_head)
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/coda_prompt] created head for domain {self.c_expert + 1} '
              f'({num_classes} classes); nearest previous domain was {idx + 1}')

    def myprediction(self, x: torch.Tensor, index: int) -> torch.Tensor:
        with torch.no_grad():
            feat = self.forward_with_prompt(x)
            return self.classifierArr[index](feat)


@register_backbone("codaprompt")
def CodaPrompt_backbone(num_classes):
    return CodaPrompt(num_classes)