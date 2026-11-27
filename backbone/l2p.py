# Adapted for Multi-Domain Incremental Learning (MDIL) — varying #classes per domain.
# Follows the exact conventions of backbone/LEAR.py from this repo:
#   - MammothBackbone + @register_backbone
#   - a growable ModuleList of per-domain classifier heads (`classifierArr`)
#   - `CreateNewExper(idx, num_classes)` to expand a new head when a new domain arrives
#   - `myprediction(x, index)` to run inference through a specific domain's head
#
# Original method: Learning to Prompt for Continual Learning (L2P), Wang et al., CVPR 2022.
# MDIL adaptation: the (frozen ViT + prompt pool) trunk is SHARED and kept training across
# domains exactly as in vanilla L2P (task-agnostic prompt selection). What changes for MDIL
# is the *classification head*: instead of one joint head over a fixed global label space
# (the class-il assumption), each domain gets its own head sized to that domain's own
# local number of classes (`dataset.N_CLASSES`), stored in `classifierArr`, mirroring how
# `backbone/LEAR.py` keeps one classifier per expert in `classifierArr`.

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class L2PPromptPool(nn.Module):
    """
    Learning-to-Prompt pool: `pool_size` learnable (key, prompt) pairs. For a batch of
    query vectors (CLS feature of the *frozen* ViT), the top-`top_k` prompts whose keys
    are most similar (cosine) to the query are selected and prepended to the patch tokens.
    This part is intentionally task-agnostic: it requires no domain id, at train OR test time,
    which is exactly what makes L2P/DualPrompt/CODA-Prompt attractive for MDIL/domain-IL.
    """

    def __init__(self, embed_dim: int = 768, pool_size: int = 20, prompt_len: int = 5, top_k: int = 5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_len = prompt_len
        self.top_k = top_k
        self.keys = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, embed_dim) * 0.02)

    def forward(self, query: torch.Tensor):
        """
        Args:
            query: [B, embed_dim] CLS features from the frozen ViT (no grad).
        Returns:
            selected_prompts: [B, top_k * prompt_len, embed_dim]
            pull_loss: scalar tensor, the L2P "pull constraint" (encourages selected keys
                       to be close to their queries)
            topk_idx: [B, top_k] indices of the selected prompts (kept for inspection/debug)
        """
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        sim = q @ k.t()  # [B, pool_size]
        topk_sim, topk_idx = sim.topk(self.top_k, dim=-1)  # [B, top_k]
        selected = self.prompts[topk_idx]  # [B, top_k, prompt_len, embed_dim]
        selected = selected.flatten(1, 2)  # [B, top_k * prompt_len, embed_dim]
        pull_loss = (1.0 - topk_sim).mean()
        return selected, pull_loss, topk_idx


class L2P(MammothBackbone):
    """
    Frozen ViT-B/16 trunk + L2P prompt pool + one classifier head per domain.
    """

    def __init__(self, num_classes: int) -> None:
        super(L2P, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768

        self.vit_process = transforms.Compose([transforms.Resize(224)])

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.prompt_pool = L2PPromptPool(embed_dim=self.embed_dim)

        # one head per domain, grown lazily via CreateNewExper — mirrors LEAR's classifierArr
        self.classifierArr = nn.ModuleList([nn.Linear(self.embed_dim, num_classes, device=self.device)])
        # per-domain feature distribution, used ONLY at test time to guess which domain
        # a sample belongs to (no task-id oracle) — same mechanism as backbone/LEAR.py's
        # `distributions` + Mahalanobis distance, computed in models/l2p.py.
        self.distributions = []
        self.c_expert = 0  # index of the domain/head currently being trained

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    @torch.no_grad()
    def _frozen_query(self, processed_x: torch.Tensor) -> torch.Tensor:
        """CLS feature from the fully frozen ViT — used only to query the prompt pool."""
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        return self.vit.forward_features(processed_x)[:, 0, :]

    def forward_with_prompt(self, x: torch.Tensor):
        """
        Runs: resize -> query prompt pool with frozen CLS feature -> prepend selected
        prompts to patch tokens -> run through ViT blocks -> return CLS feature.
        """
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        query = self._frozen_query(processed_x)

        selected_prompts, pull_loss, _ = self.prompt_pool(query)

        tokens = self.vit.patch_embed(processed_x)
        cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.vit.pos_embed
        tokens = torch.cat((tokens[:, :1, :], selected_prompts, tokens[:, 1:, :]), dim=1)

        for block in self.vit.blocks:
            tokens = block(tokens)
        tokens = self.vit.norm(tokens)
        feat = tokens[:, 0, :]
        return feat, pull_loss

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat, pull_loss = self.forward_with_prompt(x)
        out = self.classifierArr[self.c_expert](feat)
        if return_features:
            return out, pull_loss, feat
        return out, pull_loss

    def CreateNewExper(self, idx: int, num_classes: int) -> None:
        """
        Called once at the start of every domain after the first. `idx` is the index of
        the *nearest* previously-seen domain (chosen by models/l2p.py via Mahalanobis
        distance) — kept as an argument for interface parity with backbone/LEAR.py, but
        unlike LEAR's `fcArr`, L2P's prompt pool is shared/global, so there is nothing to
        clone from the nearest expert here — we only grow a fresh classifier head.
        """
        new_head = nn.Linear(self.embed_dim, num_classes, device=self.device)
        self.classifierArr.append(new_head)
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/l2p] created head for domain {self.c_expert + 1} '
              f'({num_classes} classes); nearest previous domain was {idx + 1}')

    def myprediction(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Inference using domain `index`'s own head (used by models/l2p.py:myPrediction)."""
        with torch.no_grad():
            feat, _ = self.forward_with_prompt(x)
            return self.classifierArr[index](feat)


@register_backbone("l2p")
def L2P_backbone(num_classes):
    return L2P(num_classes)