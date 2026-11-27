# Ported from the OFFICIAL implementation (SOYO-main/models/soyo_vit.py) and adapted for
# Multi-Domain Incremental Learning (varying #classes per domain) + this repo's
# conventions (MammothBackbone, classifierArr, CreateNewExper, myprediction).
#
# Kept FAITHFUL to the official code (models/soyo_vit.py: `ViT_Prompts`, `soyo_vit`):
#  - `img_prompt`: one learnable prompt TOKEN SET per domain (official: `prompt_pool[i]`,
#    an `nn.Linear(embed_dim, prompt_length, bias=False)` whose `.weight` IS the prompt —
#    kept exactly, see `_build_prompt` below), inserted once, right after the CLS token,
#    before block 0 (official `ViT_Prompts.forward`).
#  - `down_pool[i][l]` / `up_pool[i][l]`: one bottleneck adapter PER DOMAIN PER LAYER,
#    applied as `x = x + up_pool[l](down_pool[l](x))` after every block's usual
#    attention+MLP (official `ViT_Prompts.forward`, the `up_pool`/`down_pool` loop).
#  - `classifier[i]`: one linear head per domain (official already does this — the
#    official benchmarks (CDDB/DomainNet/core50) just happen to use the SAME #classes for
#    every domain; here `classifierArr` heads are simply sized per-domain instead).
#  - `interface(image, selection)` (official): runs EVERY domain's (prompt, adapters,
#    head) on the batch and gathers, per SAMPLE, the output of the domain predicted by
#    the SOYO domain classifier — this is exactly this repo's `myprediction(x, index)`
#    contract generalized to per-sample `index`; kept as `interface`, and `myprediction`
#    is a thin per-batch-uniform-index wrapper around it for interface compliance.
#
# The GMC (Gaussian Mixture Compressor) + DFR (Domain Feature Resampler) live in
# models/soyo.py (they operate on stored features, not on the network), exactly mirroring
# the official split between models/soyo_vit.py (network) and methods/soyo.py (training
# loop) + utils/soyo_utils.py (GMM code, ported near-verbatim with sklearn).

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


class SOYO(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(SOYO, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768
        self.prompt_length = 10   # official default (configs/*_soyo_vit.yaml: prompt_length)
        self.hidden_dim = 16      # official default (configs/*_soyo_vit.yaml: hidden_dim) —
                                   # was wrongly set to 64 before; the bigger bottleneck made
                                   # the per-layer adapters much harder to optimize with the
                                   # same LR/epoch budget, which is one contributor to the
                                   # underfitting you observed.
        self.n_layers = 12

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

        # one prompt + one set of per-layer adapters + one head per domain, grown via
        # CreateNewExper (official: prompt_pool / down_pool / up_pool / classifier, all
        # nn.ModuleList indexed by domain)
        self.prompt_pool = nn.ModuleList([self._build_prompt()])
        self.down_pool = nn.ModuleList([self._build_down()])
        self.up_pool = nn.ModuleList([self._build_up()])
        self.classifierArr = nn.ModuleList([nn.Linear(self.embed_dim, num_classes, device=self.device)])

        self.gmms = []            # GMC store: one (means, covariances, weights) GMM per domain
        self.distributions = []   # unused here (SOYO routes via its own trained domain
                                   # classifier — see models/soyo.py — not Mahalanobis;
                                   # kept only for interface parity with the other files)
        self.c_expert = 0

    def _build_prompt(self):
        # official: `nn.Linear(image_dim, prompt_length, bias=False)`, `.weight` used as
        # the actual prompt tensor of shape [prompt_length, embed_dim]
        return nn.Linear(self.embed_dim, self.prompt_length, bias=False, device=self.device)

    def _build_down(self):
        layers = nn.ModuleList([nn.Linear(self.embed_dim, self.hidden_dim, device=self.device)
                                 for _ in range(self.n_layers)])
        for lin in layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        return layers

    def _build_up(self):
        layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.embed_dim, device=self.device)
                                 for _ in range(self.n_layers)])
        for lin in layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        return layers

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    @torch.no_grad()
    def extract_vector(self, x) -> torch.Tensor:
        """Frozen, prompt-free CLS feature — used both for GMC/DFR (models/soyo.py) and
        as the SOYO domain classifier's input (official: `extract_vector`)."""
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        feat = self.image_encoder(processed_x)
        return F.normalize(feat, dim=-1)

    def _vit_prompts_forward(self, x, expert_idx: int) -> torch.Tensor:
        """Official `ViT_Prompts.forward`: patch-embed -> insert this domain's prompt
        right after CLS -> run blocks, adding this domain's bottleneck-adapter residual
        after every block -> final CLS feature."""
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        v = self.image_encoder
        z = v.patch_embed(processed_x)
        cls_token = v.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls_token, z), dim=1) + v.pos_embed

        img_prompt = self.prompt_pool[expert_idx].weight  # [prompt_length, embed_dim]
        img_prompt = img_prompt.unsqueeze(0).expand(z.shape[0], -1, -1)
        z = torch.cat([z[:, :1, :], img_prompt, z[:, 1:, :]], dim=1)

        down = self.down_pool[expert_idx]
        up = self.up_pool[expert_idx]
        for i, block in enumerate(v.blocks):
            z = block(z)
            z = z + up[i](down[i](z))
        z = v.norm(z[:, 0])
        return z

    def forward(self, x, return_features=False):
        feat = self._vit_prompts_forward(x, self.c_expert)
        out = self.classifierArr[self.c_expert](feat)
        if return_features:
            return out, feat
        return out

    def interface(self, x, selection: torch.Tensor) -> torch.Tensor:
        """Official `soyo_vit.interface`: run every domain seen so far and gather, per
        SAMPLE, the output of the domain predicted by the SOYO domain classifier."""
        n_domains = len(self.classifierArr)
        feats = torch.stack([self._vit_prompts_forward(x, i) for i in range(n_domains)], dim=0)
        logits_all = torch.stack([self.classifierArr[i](feats[i]) for i in range(n_domains)], dim=0)
        arange = torch.arange(x.shape[0], device=x.device)
        return logits_all[selection, arange]

    def CreateNewExper(self, idx: int, num_classes: int) -> None:
        self.prompt_pool.append(self._build_prompt())
        self.down_pool.append(self._build_down())
        self.up_pool.append(self._build_up())
        self.classifierArr.append(nn.Linear(self.embed_dim, num_classes, device=self.device))
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/soyo] created prompt+adapters+head for domain {self.c_expert + 1} '
              f'({num_classes} classes) (official SOYO always inits fresh, no cloning; '
              f'`idx`={idx} kept only for logging/interface parity)')

    def myprediction(self, x, index):
        with torch.no_grad():
            feat = self._vit_prompts_forward(x, index)
            return self.classifierArr[index](feat)


@register_backbone("soyo")
def SOYO_backbone(num_classes):
    return SOYO(num_classes)