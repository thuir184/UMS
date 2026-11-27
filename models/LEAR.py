# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from backbone.utils.hsic import hsic
import torch.nn.functional as F
import random
import copy
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from backbone.bilora import BiLORA_MoE, BiLORA_Focal, BiLoRA_InverseMoE
from models.utils.quantum_task_router import QuantumRLSTaskRouter
from tqdm import tqdm

def parse_apply_to(apply_to_val, total_blocks=12):
    """Parses the --apply-to argument into a sorted list of block indices.

    Supports:
        - None -> [9, 10, 11]
        - 'all' or 'ALL' -> [0, 1, ..., total_blocks - 1]
        - '9,10,11' or '9, 10, 11' -> [9, 10, 11]
        - list/tuple of ints -> list of ints
    """
    if apply_to_val is None:
        return [9, 10, 11]
    if isinstance(apply_to_val, (list, tuple)):
        return [int(x) for x in apply_to_val]
    s = str(apply_to_val).strip()
    if s.lower() == 'all':
        return list(range(total_blocks))
    parts = s.replace(' ', ',').split(',')
    blocks = [int(p.strip()) for p in parts if p.strip()]
    for b in blocks:
        if b < 0 or b >= total_blocks:
            raise ValueError(f"Invalid block index {b} in apply_to='{apply_to_val}'. Must be between 0 and {total_blocks - 1}.")
    return sorted(list(set(blocks)))


class LEAR(ContinualModel):
    NAME = 'LEAR'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        existing = [action.dest for action in parser._actions]
        if 'apply_to' not in existing and 'apply-to' not in existing:
            parser.add_argument('--apply-to', '--apply_to', type=str, default="9,10,11",
                                help="ViT blocks to apply FocalExpert modulation ('all' for all 12 blocks, or comma-separated indices e.g. '9,10,11')")
        if 'ums_in_loop' not in existing and 'ums-in-loop' not in existing:
            from utils import binary_to_boolean_type
            parser.add_argument('--ums-in-loop', '--ums_in_loop', nargs='?', const=True, default=False,
                                type=binary_to_boolean_type, help="Enable UMS-in-the-loop training where sparse coefficients are projected through UMS operator during forward pass")
        if 'ums_refresh_every' not in existing and 'ums-refresh-every' not in existing:
            parser.add_argument('--ums_refresh_every', '--ums-refresh-every', type=int, default=1,
                                help="Epoch interval to refresh candidate basis in UMS-in-the-loop")
        if 'task_retrieval' not in existing and 'task-retrieval' not in existing:
            parser.add_argument('--task-retrieval', '--task_retrieval', type=str, default="mahalanobis",
                                choices=["mahalanobis", "resonance", "quantum_rls"],
                                help="Task retrieval strategy: Mahalanobis, UMS resonance, or fixed quantum features with recursive ridge")
        if 'quantum_n_qubits' not in existing:
            parser.add_argument('--quantum-n-qubits', '--quantum_n_qubits', type=int, default=8,
                                help="Number of qubits for the built-in quantum RLS task router (2-16)")
        if 'quantum_layers' not in existing:
            parser.add_argument('--quantum-layers', '--quantum_layers', type=int, default=2,
                                help="Number of data-reuploading RY/RZ + ring-CNOT layers")
        if 'quantum_ridge' not in existing:
            parser.add_argument('--quantum-ridge', '--quantum_ridge', type=float, default=1e-3,
                                help="L2 regularization for the recursive ridge task head")
        if 'quantum_shots' not in existing:
            parser.add_argument('--quantum-shots', '--quantum_shots', type=int, default=0,
                                help="Measurement shots per observable; 0 uses exact state-vector expectations")
        if 'quantum_seed' not in existing:
            parser.add_argument('--quantum-seed', '--quantum_seed', type=int, default=None,
                                help="Seed for the fixed orthogonal reducer (defaults to the run seed)")
        if 'quantum_balance_tasks' not in existing:
            from utils import binary_to_boolean_type
            parser.add_argument(
                '--quantum-balance-tasks', '--quantum_balance_tasks',
                nargs='?', const=True, default=True, type=binary_to_boolean_type,
                help="Give every task equal total weight in the recursive ridge objective",
            )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LEAR, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.train_loader_size = None
        self.trainable_blocks = parse_apply_to(getattr(self.args, 'apply_to', '9,10,11'))
        print(f"[INFO] Trainable ViT blocks set to: {self.trainable_blocks} (total: {len(self.trainable_blocks)}/12)")
        self.iter = 0

        # Task retrieval strategy.  All implementations remain available so an
        # experiment can be switched without reverting code or changing UMS.
        self.task_retrieval = str(getattr(self.args, 'task_retrieval', 'mahalanobis')).lower()
        if self.task_retrieval not in ['mahalanobis', 'resonance', 'quantum_rls']:
            raise ValueError(
                f"Unknown task retrieval strategy: '{self.task_retrieval}'. "
                "Must be 'mahalanobis', 'resonance', or 'quantum_rls'."
            )
        print(f"[INFO] Task retrieval strategy: {self.task_retrieval.upper()}")

        if self.task_retrieval == 'quantum_rls':
            quantum_seed = getattr(self.args, 'quantum_seed', None)
            if quantum_seed is None:
                quantum_seed = int(getattr(self.args, 'seed', 0))
            self.net.quantum_task_router = QuantumRLSTaskRouter(
                input_dim=getattr(self.net, 'model_dim', 768),
                n_tasks=int(getattr(self.args, 'n_tasks', 10)),
                n_qubits=int(getattr(self.args, 'quantum_n_qubits', 8)),
                n_layers=int(getattr(self.args, 'quantum_layers', 2)),
                ridge=float(getattr(self.args, 'quantum_ridge', 1e-3)),
                seed=int(quantum_seed),
                shots=int(getattr(self.args, 'quantum_shots', 0)),
                balance_tasks=bool(getattr(self.args, 'quantum_balance_tasks', True)),
            )
            q_router = self.net.quantum_task_router
            print(
                f"[INFO] Quantum-RLS router initialized: qubits={q_router.feature_map.n_qubits}, "
                f"layers={q_router.feature_map.n_layers}, features={q_router.feature_dim}, "
                f"shots={q_router.feature_map.shots}, ridge={q_router.ridge}, "
                f"balance_tasks={q_router.balance_tasks}"
            )

        # UMS-in-the-loop: forward pass projection through candidate UMS operator
        self.ums_in_loop = bool(getattr(self.args, 'ums_in_loop', False))
        self.ums_refresh_every = int(getattr(self.args, 'ums_refresh_every', 1))
        if self.ums_in_loop:
            print(f"[INFO] UMS-in-the-loop ENABLED: forward pass projects sparse coefficients through candidate UMS operator (refresh_every={self.ums_refresh_every} epochs)")

        self.init_bilora()
        self.opt = self.get_optimizer()


    def end_task(self, dataset) -> None:
        # calculate distribution for Mahalanobis routing if selected
        if self.task_retrieval == 'mahalanobis':
            train_loader = dataset.train_loader
            num_choose = len(train_loader)
            with torch.no_grad():
                train_iter = iter(train_loader)

                pbar = tqdm(train_iter, total=num_choose,
                            desc=f"Calculate distribution for task {self.current_task + 1}",
                            disable=False, mininterval=0.5)

                fc_features_list = []

                count = 0
                while count < num_choose:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break

                    x = data[0]
                    x = x.to(self.device)

                    processX = self.net.vitProcess(x)
                    if processX.size(1) == 1:
                        processX = processX.expand(-1, 3, -1, -1)
                    
                    class_token = self.net.extract_feature(processX)

                    fc_features_list.append(self.net.fcArr[self.current_task](class_token))

                    count += 1
                    pbar.update()

                pbar.close()
                fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
                mu = torch.mean(fc_features, dim=0)
                sigma = torch.cov(fc_features.T) + 1e-4 * torch.eye(fc_features.shape[1], device=fc_features.device)
                self.net.distributions.append(MultivariateNormal(mu, sigma))
        elif self.task_retrieval == 'quantum_rls':
            train_loader = dataset.train_loader
            num_choose = len(train_loader)
            router = self.net.quantum_task_router
            with torch.no_grad():
                train_iter = iter(train_loader)
                pbar = tqdm(
                    train_iter,
                    total=num_choose,
                    desc=f"Update Quantum-RLS router for task {self.current_task + 1}",
                    disable=False,
                    mininterval=0.5,
                )
                count = 0
                while count < num_choose:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break

                    x = data[0].to(self.device)
                    processX = self.net.vitProcess(x)
                    if processX.size(1) == 1:
                        processX = processX.expand(-1, 3, -1, -1)
                    class_token = self.net.extract_feature(processX)
                    router.update_batch(class_token, self.current_task)

                    count += 1
                    pbar.update()
                pbar.close()

            router_metrics = router.finalize_task(self.current_task)
            print(
                f"[Quantum-RLS] Task {self.current_task + 1} updated: "
                f"samples={int(router_metrics['samples'])}, "
                f"features={int(router_metrics['feature_dim'])}, "
                f"tasks_seen={int(router_metrics['tasks_seen'])}, "
                f"cond(A)={router_metrics['gram_condition']:.3e}"
            )
        else:
            print(f"[INFO] Task retrieval mode is '{self.task_retrieval}' - skipping Gaussian distribution calculation.")

        #deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False

        # Fold task into UMS Consolidator
        self.turn_off_saving_experts()

        # The outer training loop saves after evaluation.  Avoid a second,
        # pre-evaluation serialization when final-only checkpointing is used.
        if (not getattr(self.args, 'disable_task_checkpoints', False)
                and not getattr(self.args, 'checkpoint_final_only', False)):
            self.save_task_checkpoint(task_id=self.current_task)

    def save_task_checkpoint(self, task_id=None, checkpoint_dir=None, eval_results=None, loss_history=None, extra_metadata=None):
        from utils.task_checkpointing import save_task_checkpoint
        t_id = self.current_task if task_id is None else task_id
        ckpt_dir = getattr(self.args, 'checkpoint_dir', 'checkpoints') if checkpoint_dir is None else checkpoint_dir
        metadata = dict(extra_metadata or {})
        metadata.setdefault('task_retrieval', self.task_retrieval)
        if self.task_retrieval == 'quantum_rls':
            metadata['quantum_router'] = self.net.quantum_task_router.get_analysis_metrics()
        return save_task_checkpoint(
            model=self,
            task_id=t_id,
            base_dir=ckpt_dir,
            optimizer=getattr(self, 'opt', None),
            eval_results=eval_results,
            loss_history=loss_history,
            extra_metadata=metadata
        )

    def load_task_checkpoint(self, checkpoint_path_or_dir, task_id=None):
        from utils.task_checkpointing import load_task_checkpoint
        return load_task_checkpoint(
            checkpoint_path_or_task_dir=checkpoint_path_or_dir,
            model=self,
            task_id=task_id,
            optimizer=getattr(self, 'opt', None),
            device=self.device
        )
        
    def begin_task(self, dataset, threshold=0) -> None:
        min_idx = 0
        if self.current_task > 0:
            self.net.CreateNewExper(min_idx, dataset.N_CLASSES)

        if self.ums_in_loop:
            for i in self.trainable_blocks:
                attn = getattr(self.net.local_vitmodel.blocks[i], 'attn', None)
                if attn is not None and hasattr(attn, 'refresh_basis'):
                    attn.refresh_basis(self.current_task)

        self.opt = self.get_optimizer()

    def begin_epoch(self, epoch: int, dataset: ContinualDataset) -> None:
        super().begin_epoch(epoch, dataset)
        if self.ums_in_loop and epoch > 0:
            # Refresh basis periodically (except last epoch to freeze basis for fine-tuning)
            if (epoch % self.ums_refresh_every == 0) and (epoch < self.args.n_epochs - 1):
                for i in self.trainable_blocks:
                    attn = getattr(self.net.local_vitmodel.blocks[i], 'attn', None)
                    if attn is not None and hasattr(attn, 'refresh_basis'):
                        attn.refresh_basis(self.current_task)

    def get_parameters(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        return params

    def set_eval_task(self, index):
        for i in self.trainable_blocks:
            self.net.local_vitmodel.blocks[i].attn.set_force_task(index)

    def clear_eval_task(self):
        for i in self.trainable_blocks:
            self.net.local_vitmodel.blocks[i].attn.clear_force_task()

    def myPrediction(self, x, k):
        with torch.no_grad():
            # single persistent model: force the UMS readout for task k, then use the
            # in-memory per-task bottleneck and classifier
            self.set_eval_task(k)
            local_features = self.net.forward_fusion(x)
            fc_features = self.net.fcArr[k](local_features)
            out = self.net.classifierArr[k](fc_features)
            self.clear_eval_task()
        return out

    def get_active_focal_experts(self):
        experts = []
        for i in self.trainable_blocks:
            attn = getattr(self.net.local_vitmodel.blocks[i], 'attn', None)
            if attn is not None and hasattr(attn, 'get_focal_experts'):
                experts.extend(attn.get_focal_experts())
        return experts

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs, _ = self.net(inputs, return_features=True)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        return loss.item()

    def cal_expert_dist_mahalanobis(self, x):
        x = x.to(self.device)
        processX = self.net.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)

        class_token = self.net.extract_feature(processX)
        distances = [0.0] * len(self.net.fcArr)
        for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
            fc_feature = fc(class_token)
            delta = fc_feature - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt(delta @ inv_cov @ delta.T).diagonal()
            distances[t] += mahalanobis.mean().item()
        return distances

    def cal_expert_dist_resonance(self, x):
        """Option 1: UMS Bilinear Coordinate Resonance.

        Leverages shared orthonormal bases (P, Q), Cayley rotation R_o, and task coordinates z_t.
        Computes cosine distance: 1.0 - cosine_similarity(phi(x), z_t), where:
            phi(x) = (class_token @ R_o @ P) * (class_token @ R_o @ Q)
        Aggregates across all active FocalExpert UMS Consolidators.
        """
        x = x.to(self.device)
        processX = self.net.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)

        with torch.no_grad():
            class_token = self.net.extract_feature(processX).detach()  # [B, D]
            n_tasks_seen = len(self.net.fcArr)
            if n_tasks_seen == 0:
                return []

            experts = self.get_active_focal_experts()
            total_distances = torch.zeros(n_tasks_seen, device=class_token.device, dtype=torch.float32)
            valid_experts = 0

            for expert in experts:
                ums = getattr(expert, 'ums', None)
                if ums is None:
                    continue
                r_cur = ums.r_cur
                n_folded = len(ums.z_store)
                if r_cur == 0 or n_folded == 0:
                    continue

                tasks_eval = min(n_tasks_seen, n_folded)
                if tasks_eval == 0:
                    continue

                R = ums.rotation().detach()
                P = ums.P[:, :r_cur].detach()
                Q = ums.Q[:, :r_cur].detach()

                # Rotated bases in representation space: A = R @ P, B = R @ Q
                A = torch.matmul(R, P).to(dtype=class_token.dtype)      # [D, r_cur]
                B_mat = torch.matmul(R, Q).to(dtype=class_token.dtype)  # [D, r_cur]

                # Project class_token into bases
                a = torch.matmul(class_token, A)      # [B, r_cur]
                b = torch.matmul(class_token, B_mat)  # [B, r_cur]
                phi = a * b                           # [B, r_cur] (Bilinear UMS embedding)
                phi_norm = F.normalize(phi, p=2, dim=-1)  # [B, r_cur]

                # Stack task coordinates z_t
                z_list = [ums.z_store[t][:r_cur].detach().to(device=class_token.device, dtype=class_token.dtype)
                          for t in range(tasks_eval)]
                Z = torch.stack(z_list, dim=0)        # [tasks_eval, r_cur]
                Z_norm = F.normalize(Z, p=2, dim=-1)  # [tasks_eval, r_cur]

                # Batch cosine similarity: [B, tasks_eval]
                sim = torch.matmul(phi_norm, Z_norm.t())
                # Mean over batch: [tasks_eval]
                mean_sim = sim.mean(dim=0)
                # Cosine distance: 1.0 - mean_sim
                expert_dist = 1.0 - mean_sim

                total_distances[:tasks_eval] += expert_dist
                valid_experts += 1

            if valid_experts > 0:
                avg_dist = total_distances / float(valid_experts)
                return avg_dist.cpu().tolist()
            else:
                return [0.0] * n_tasks_seen

    def cal_expert_dist_quantum_rls(self, x):
        """Return negative mean RLS task scores as distances for evaluation."""
        x = x.to(self.device)
        processX = self.net.vitProcess(x)
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)
        with torch.no_grad():
            class_token = self.net.extract_feature(processX).detach()
            return self.net.quantum_task_router.distances(class_token)

    def cal_expert_dist(self, x):
        if self.task_retrieval == 'resonance':
            return self.cal_expert_dist_resonance(x)
        if self.task_retrieval == 'quantum_rls':
            return self.cal_expert_dist_quantum_rls(x)
        return self.cal_expert_dist_mahalanobis(x)

    def init_bilora(self):
        n_tasks = getattr(self.args, 'n_tasks', 10)
        seed = getattr(self.args, 'seed', None)
        if seed is not None:
            from backbone.bilora import set_bilora_seed
            set_bilora_seed(seed)
        print(f"[INFO] Initializing BiLORA Focal with UMS (n_tasks={n_tasks}, seed={seed})")
        ums_cfg = {
            'basis_rank': self.args.ums_basis_rank,
            'svd_rank': self.args.ums_svd_rank,
            'tau': self.args.ums_tau,
            'rot_steps': self.args.ums_rot_steps,
            'rot_lr': getattr(self.args, 'ums_rot_lr', 0.005),
            'head_lr': getattr(self.args, 'ums_head_lr', 0.05),
            'rot_lambda_diag': getattr(self.args, 'ums_rot_lambda_diag', 0.1),
            'rot_lambda_anchor': getattr(self.args, 'ums_rot_lambda_anchor', 0.5),
            'rot_lambda_support_mse': getattr(self.args, 'ums_rot_lambda_support_mse', 1.0),
            'rot_lambda_support_bce': getattr(self.args, 'ums_rot_lambda_support_bce', 0.1),
            'rot_lambda_gaussian': getattr(self.args, 'ums_rot_lambda_gaussian', 0.01),
            'n_frq': getattr(self.args, 'n_frq', 576),
            'ums_in_loop': self.ums_in_loop,
            'seed': seed,
        }
        modulate_q = getattr(self.args, 'modulate_q', True)
        modulate_proj = getattr(self.args, 'modulate_proj', True)
        for i in range(12):
            if i in self.trainable_blocks:
                orig_attn = self.net.local_vitmodel.blocks[i].attn
                num_heads = getattr(orig_attn, 'num_heads', 12)
                qkv_bias = (orig_attn.qkv.bias is not None) if (hasattr(orig_attn, 'qkv') and hasattr(orig_attn.qkv, 'bias')) else True
                qk_scale = getattr(orig_attn, 'scale', None)

                new_attn = BiLORA_Focal(
                    dim=768, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    n_frq=self.args.n_frq, n_tasks=n_tasks,
                    ums_cfg=ums_cfg, layer_name=f"Block{i}",
                    modulate_q=modulate_q,
                    modulate_proj=modulate_proj,
                    seed=seed)

                # Copy pretrained weights and biases from the original ViT attention module
                new_attn.copy_weights_from(orig_attn)
                if hasattr(orig_attn, 'qkv') and orig_attn.qkv is not None and hasattr(orig_attn.qkv, 'weight'):
                    new_attn.to(device=orig_attn.qkv.weight.device, dtype=orig_attn.qkv.weight.dtype)

                self.net.local_vitmodel.blocks[i].attn = new_attn
                for param in self.net.local_vitmodel.blocks[i].parameters():
                    param.requires_grad = False
                for focal_attr in ['focal_q', 'focal_k', 'focal_v', 'focal_proj']:
                    expert = getattr(self.net.local_vitmodel.blocks[i].attn, focal_attr, None)
                    if expert is not None and hasattr(expert, 'coeff'):
                        expert.coeff.requires_grad = True
        print(f"[INFO] Initialized BiLORA Focal with UMS (n_tasks={n_tasks}, modulate_q={modulate_q}, modulate_proj={modulate_proj}, seed={seed})")

    def turn_off_saving_experts(self):
        print(f"\n================================================================================")
        print(f"[UMS] STARTING CONSOLIDATION FOR TASK {self.current_task} (Blocks: {self.trainable_blocks})")
        print(f"================================================================================")
        for i in range(12):
            if i in self.trainable_blocks:
                self.net.local_vitmodel.blocks[i].attn.end_task(self.current_task)
        print(f"================================================================================")
        print(f"[UMS] FINISHED CONSOLIDATION FOR TASK {self.current_task}")
        print(f"================================================================================\n")
