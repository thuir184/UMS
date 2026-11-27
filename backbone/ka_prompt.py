# Ported from the OFFICIAL implementation (ICML2025-KA-Prompt-main/models/kaprompt_vit.py,
# utils/prompt_pool.py) and adapted for Multi-Domain Incremental Learning (varying
# #classes per domain) + this repo's conventions (MammothBackbone, classifierArr,
# CreateNewExper, myprediction).
#
# Kept FAITHFUL to the official code:
#  - true prefix-tuning: expert/general prompts are split in half and injected as extra
#    Key / Value tokens inside attention (see `_forward_attn`, ported near-verbatim from
#    kaprompt_vit.py:forward_attn/forward_block).
#  - general prompt at layers [0, 1] (`layer_g`), shared/continually trained across ALL
#    domains, exactly as in the official code.
#  - expert prompt pool at layers [2..11] (`layer_e`, official default `e_s=2, e_e=11`),
#    ONE prompt (of length `prompt_num=20`) per domain (official default `prompt_per_task=1`).
#  - `greedy_init` (utils/prompt_pool.py + kaprompt_vit.py) ported near-verbatim: when a
#    new domain arrives, its expert prompt/key are initialized via the official
#    coverage-style greedy search over the existing prompt pool (not a naive
#    nearest-neighbour clone).
#  - `forward_specific_prompt` (official kaprompt_vit.py) is reused for `myprediction`,
#    since it already implements exactly "run the ViT with one fixed, externally chosen,
#    expert prompt" — precisely this repo's `myprediction(x, index)` contract.
#
# DROPPED vs. the official code (out of scope for this port / not needed for MDIL
# benchmarking): EMA smoothing of the general prompt (`use_ema`/`use_ema_c`), the
# auxiliary old-model distillation loss (`aux_weight`, `forward_vit_old`), prompt
# shuffling, and disk save/load of the prompt pool. The official code also uses a SINGLE
# shared classifier head (its own DIL benchmarks — CDDB, DomainNet, ImageNet-R/C — all
# keep the same label space across domains); here `classifierArr` grows one head per
# domain instead, which is this repo's standard MDIL adaptation used for every other
# integrated method.

import copy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import MammothBackbone, register_backbone


def greedy_init(vit, loader, prompt_list, key_list, new_idx, query_pos, device):
    """Ported near-verbatim from the official kaprompt_vit.py:greedy_init, simplified to
    prompt_per_task=1 (the official default): initializes the new domain's single expert
    prompt/key by greedily picking the existing prompt that best "covers" the new
    domain's own query distribution (a submodular-style coverage criterion), rather than
    just cloning the single nearest one. Returns the source index (or None for the first
    domain) — does NOT mutate `prompt_list`/`key_list` itself, the caller appends."""
    if new_idx == 0:
        return None
    all_keys = torch.stack(list(key_list[:new_idx]))
    all_keys = F.normalize(all_keys, dim=-1)

    sim_scores = []
    for data in loader:
        x = data[0].to(device)
        with torch.no_grad():
            processed_x = vit['process'](x)
            if processed_x.size(1) == 1:
                processed_x = processed_x.expand(-1, 3, -1, -1)
            reps = vit['model'].forward_features(processed_x)  # [B, N, embed_dim]
            query = reps[:, query_pos, :] if query_pos >= 0 else reps[:, 1:, :].mean(dim=1)
            query = F.normalize(query, dim=-1)
            sim_scores.append(all_keys @ query.T)
    sim_scores = (torch.cat(sim_scores, dim=1) + 1) / 2  # [new_idx, N_samples] in [0, 1]

    # single-slot greedy pick (prompt_per_task = 1): the old prompt whose query-coverage
    # is highest overall
    overall_w = sim_scores.sum(1)
    return int(torch.argmax(overall_w).item())


class KAPrompt(MammothBackbone):
    def __init__(self, num_classes: int) -> None:
        super(KAPrompt, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.embed_dim = 768
        self.n_layers = 12
        self.layer_g = [0, 1]                       # official default
        self.layer_e = list(range(2, 12))            # official default (e_s=2, e_e=11)
        self.n_e_layers = len(self.layer_e)
        self.prompt_num = 20                         # official default (E-prompt length)
        self.gprompt_num = 4                         # official default (general prompt length)
        self.topN = 1                                # official default
        self.query_pos = 1                           # official default

        self.vit_process = transforms.Compose([transforms.Resize(224)])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # general prompt: shared/global, continually trained (official: `self.general_prompt`)
        self.general_prompt = nn.ParameterList([
            nn.Parameter(torch.empty(self.gprompt_num, self.embed_dim, device=self.device).uniform_(-1, 1))
            for _ in self.layer_g
        ])

        # expert prompt pool + keys: one entry per domain, grown via CreateNewExper
        # (official: PromptPool.prompt_list / key_list, per layer in `layer_e`)
        self.prompt_list = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.empty(self.prompt_num, self.embed_dim, device=self.device).uniform_(-1, 1))
            ]) for _ in range(self.n_e_layers)
        ])
        self.key_list = nn.ParameterList([
            nn.Parameter(nn.init.uniform_(torch.randn(self.embed_dim, device=self.device)))
        ])

        self.classifierArr = nn.ModuleList([nn.Linear(self.embed_dim, num_classes, device=self.device)])
        self.distributions = []  # for Mahalanobis-based domain routing, see models/ka_prompt.py
        self.c_expert = 0

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    # ---------------------------------------------------------- prefix-tuned attention ----
    def _forward_attn(self, block, x, prompts):
        """Ported near-verbatim from kaprompt_vit.py:forward_attn — splits `prompts` in
        half, uses the first half as extra Key tokens and the second half as extra Value
        tokens (true prefix-tuning), while Query stays computed from the un-prompted x."""
        attn = block.attn
        b, n, c = x.shape
        qkv_q = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, _, _ = qkv_q.unbind(0)

        half = prompts.shape[1] // 2
        qkv_k = torch.cat([x[:, :1, :], prompts[:, :half, :], x[:, 1:, :]], dim=1)
        qkv_k = attn.qkv(qkv_k).reshape(b, -1, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        _, k, _ = qkv_k.unbind(0)
        qkv_v = torch.cat([x[:, :1, :], prompts[:, half:, :], x[:, 1:, :]], dim=1)
        qkv_v = attn.qkv(qkv_v).reshape(b, -1, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        _, _, v = qkv_v.unbind(0)

        q, k = attn.q_norm(q), attn.k_norm(k)
        q = q * attn.scale
        a = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        a = attn.attn_drop(a)
        x = (a @ v).transpose(1, 2).reshape(b, n, c)
        x = attn.proj(x)
        x = attn.proj_drop(x)
        return x

    def _forward_block(self, layer, x, prompts):
        block = self.vit.blocks[layer]
        if prompts is not None:
            x = x + block.drop_path1(block.ls1(self._forward_attn(block, block.norm1(x), prompts)))
        else:
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def _get_prompts(self, layer, b, e_idx):
        if layer in self.layer_g:
            g_i = self.layer_g.index(layer)
            return self.general_prompt[g_i].unsqueeze(0).expand(b, -1, -1)
        if layer in self.layer_e:
            e_i = self.layer_e.index(layer)
            return self.prompt_list[e_i][e_idx].unsqueeze(0).expand(b, -1, -1)
        return None

    def forward_specific_prompt(self, x, expert_idx: int):
        """Ported from the official kaprompt_vit.py:forward_specific_prompt — runs the
        ViT with ONE fixed, externally-chosen expert prompt (used by this repo's
        `myprediction(x, index)` contract)."""
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        z = self.vit.patch_embed(processed_x)
        cls_token = self.vit.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls_token, z), dim=1) + self.vit.pos_embed
        b = z.shape[0]
        for layer in range(self.n_layers):
            prompts = self._get_prompts(layer, b, expert_idx)
            z = self._forward_block(layer, z, prompts)
        z = self.vit.norm(z)
        return z[:, 0, :]

    def forward(self, x, return_features=False):
        feat = self.forward_specific_prompt(x, self.c_expert)
        out = self.classifierArr[self.c_expert](feat)
        if return_features:
            return out, feat
        return out

    @torch.no_grad()
    def frozen_feature(self, x):
        processed_x = self.vit_process(x)
        if processed_x.size(1) == 1:
            processed_x = processed_x.expand(-1, 3, -1, -1)
        return F.normalize(self.vit.forward_features(processed_x)[:, 0, :], dim=-1)

    # ------------------------------------------------------------------- growing ----
    def register_first_domain(self) -> None:
        """Domain 0 uses the prompt/key/head created in __init__ directly (index 0)."""
        self.c_expert = 0

    def CreateNewExper(self, greedy_loader, num_classes: int) -> None:
        new_idx = len(self.key_list)
        # official: greedy_init picks the existing prompt with the best query-coverage
        # and clones it as the new domain's starting point (Phase 1 of the paper)
        source_idx = greedy_init(
            {'process': self.vit_process, 'model': self.vit}, greedy_loader,
            self.prompt_list, self.key_list, new_idx, self.query_pos, self.device)
        if source_idx is not None:
            for e_i in range(self.n_e_layers):
                self.prompt_list[e_i].append(nn.Parameter(self.prompt_list[e_i][source_idx].detach().clone()))
            self.key_list.append(nn.Parameter(self.key_list[source_idx].detach().clone()))
        self.classifierArr.append(nn.Linear(self.embed_dim, num_classes, device=self.device))
        self.c_expert = len(self.classifierArr) - 1
        print(f'[backbone/ka_prompt] configured expert prompt for domain {self.c_expert + 1} '
              f'({num_classes} classes) via official greedy_init'
              + (f' (source domain {source_idx + 1})' if source_idx is not None else ' (first domain)'))

    def myprediction(self, x, index):
        with torch.no_grad():
            feat = self.forward_specific_prompt(x, index)
            return self.classifierArr[index](feat)


@register_backbone("ka_prompt")
def KAPrompt_backbone(num_classes):
    return KAPrompt(num_classes)