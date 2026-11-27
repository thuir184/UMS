# Adapted for MDIL. Pair with backbone/moe_adapters.py, run with --model moe_adapters --backbone moe_adapters.
#
# Unlike the other 4 files (which reuse a Mahalanobis-distance-over-frozen-features
# routing signal, like models/LEAR.py), MoE_Adapters routes via its own learned router
# (`self.net.router_heads`), trained with an auxiliary cross-entropy loss against the
# ground-truth domain id (known at train time, per the "no task-id at TEST time only"
# rule) — this is closer to the spirit of the original paper's autonomous router.

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class MoeAdapters(ContinualModel):
    NAME = 'moe_adapters'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--moe_router_weight', type=float, default=1.0,
                             help='Weight of the auxiliary router cross-entropy loss.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MoeAdapters, self).__init__(backbone, loss, args, transform, dataset=dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs, router_logits = self.net(inputs)
        loss_ce = self.loss(outputs, labels)
        domain_target = torch.full((inputs.shape[0],), self.current_task,
                                    dtype=torch.long, device=self.device)
        loss_router = F.cross_entropy(router_logits, domain_target)
        loss_tot = loss_ce + self.args.moe_router_weight * loss_router
        loss_tot.backward()
        self.opt.step()
        return [loss_ce.item(), loss_router.item()]

    def myPrediction(self, x, k):
        with torch.no_grad():
            return self.net.myprediction(x, k)

    @torch.no_grad()
    def cal_expert_dist(self, x):
        """Lower = more likely match, to stay consistent with the Mahalanobis convention
        used elsewhere in this harness: we return the negative log-softmax of the
        router's own logits for each domain seen so far."""
        router_logits = self.net.router_logits(x)  # [B, num_domains_so_far]
        neg_log_probs = -F.log_softmax(router_logits, dim=1).mean(dim=0)  # [num_domains_so_far]
        return neg_log_probs.tolist()

    @torch.no_grad()
    def _pick_nearest_domain(self, loader, num_batches=50):
        train_iter = iter(loader)
        sums = None
        count = 0
        for _ in range(num_batches):
            try:
                data = next(train_iter)
            except StopIteration:
                break
            x = data[0].to(self.device)
            d = torch.tensor(self.cal_expert_dist(x))
            sums = d if sums is None else sums + d
            count += 1
        avg = sums / max(count, 1)
        return int(avg.argmin().item())

    def begin_task(self, dataset) -> None:
        if self.current_task > 0:
            min_idx = self._pick_nearest_domain(dataset.train_loader, num_batches=50)
            self.net.CreateNewExper(min_idx, dataset.N_CLASSES)
        self.opt = self.get_optimizer()

    def end_task(self, dataset) -> None:
        for p in self.net.classifierArr[self.net.c_expert].parameters():
            p.requires_grad = False
        for p in self.net.adapters[self.net.c_expert].parameters():
            p.requires_grad = False
        # router heads for THIS domain stay trainable in principle, but since this
        # domain's data is gone we simply freeze them too to avoid drift on future tasks
        for p in self.net.router_heads[self.net.c_expert].parameters():
            p.requires_grad = False