# Ported from the official ICML2025-KA-Prompt-main/models/kaprompt_vit.py (observe()/
# init_opt()) and adapted for MDIL. Pair with backbone/ka_prompt.py, run with
# --model ka_prompt --backbone ka_prompt.
#
# `--pw` (distance penalty weight) matches the official default (0.5): the official loss
# is CE + pw * mean(sum(distance)) where `distance` is `1 - cosine_sim` between the query
# and the (topN-)selected key(s) — this pulls the selected key toward the query, exactly
# like L2P's "pull constraint". With topN=1 and training always using the CURRENT
# domain's own (single) key (see forward below), `distance` reduces to `1 - cos(query,
# key[c_expert])`.
#
# Domain routing at TEST time (cal_expert_dist/myPrediction) reuses this repo's
# Mahalanobis convention (see models/LEAR.py) — the official code's own similarity-based
# retrieval already picks prompts in a domain-agnostic way *within* a shared, fixed-size
# classifier, but our MDIL setting additionally needs to pick which per-domain
# `classifierArr` head to read logits from (see the backbone docstring for why).

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class KaPrompt(ContinualModel):
    NAME = 'ka_prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--pw', type=float, default=0.5, help='Distance penalty weight (official default).')
        parser.add_argument('--ka_signature_batches', type=int, default=100)
        parser.add_argument('--ka_greedy_batches', type=int, default=50,
                             help='#batches used for the official greedy_init query-coverage search.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(KaPrompt, self).__init__(backbone, loss, args, transform, dataset=dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        query = self.net.frozen_feature(inputs)
        key = F.normalize(self.net.key_list[self.net.c_expert], dim=-1)
        distance = 1.0 - (query * key.unsqueeze(0)).sum(-1)  # official: `distance` term

        outputs = self.net(inputs)
        loss_ce = self.loss(outputs, labels)
        loss_tot = loss_ce + self.args.pw * distance.mean()
        loss_tot.backward()
        self.opt.step()
        return [loss_ce.item(), distance.mean().item()]

    def myPrediction(self, x, k):
        with torch.no_grad():
            return self.net.myprediction(x, k)

    @torch.no_grad()
    def cal_expert_dist(self, x):
        distances = []
        for expert_idx, dist in enumerate(self.net.distributions):
            feat = self.net.forward_specific_prompt(x, expert_idx)
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
                    desc=f"[ka_prompt] fitting domain signature for domain {self.current_task + 1}",
                    disable=False, mininterval=0.5)
        for _ in pbar:
            try:
                data = next(train_iter)
            except StopIteration:
                break
            x = data[0].to(self.device)
            feats.append(self.net.forward_specific_prompt(x, self.net.c_expert))
            pbar.update()
        pbar.close()
        feats = torch.cat(feats, dim=0)
        mu = feats.mean(dim=0)
        sigma = torch.cov(feats.T) + 1e-4 * torch.eye(feats.shape[1], device=feats.device)
        return MultivariateNormal(mu, sigma)

    def begin_task(self, dataset) -> None:
        if self.current_task == 0:
            self.net.register_first_domain()
        else:
            # official greedy_init needs a data loader to compute query-coverage
            self.net.CreateNewExper(dataset.train_loader, dataset.N_CLASSES)
        self.opt = self.get_optimizer()

    def end_task(self, dataset) -> None:
        dist = self._fit_domain_signature(dataset, self.args.ka_signature_batches)
        self.net.distributions.append(dist)
        for p in self.net.classifierArr[self.net.c_expert].parameters():
            p.requires_grad = False
        for e_i in range(self.net.n_e_layers):
            self.net.prompt_list[e_i][self.net.c_expert].requires_grad = False
        self.net.key_list[self.net.c_expert].requires_grad = False
        # general_prompt stays trainable and shared across all domains, as in the official code