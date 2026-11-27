import math
import logging
from functools import partial
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn.init as init
from tqdm import tqdm
GLOBAL_BILORA_SEED = 0

def set_bilora_seed(seed: int):
    """Sets the global random seed for BiLORA and regenerates frequency permutation."""
    global GLOBAL_BILORA_SEED, PERMUTATION
    GLOBAL_BILORA_SEED = int(seed)
    g = torch.Generator().manual_seed(int(seed))
    PERMUTATION = torch.randperm(768 * 768, generator=g).tolist()
    print(f"[INFO] BiLORA seed set to {seed}, regenerated frequency PERMUTATION (sample[0]={PERMUTATION[0]})")

def get_bilora_seed():
    """Returns current global BiLORA seed."""
    return GLOBAL_BILORA_SEED

def get_permutation(dim=768, seed=None):
    """Gets or generates deterministic frequency permutation matching seed."""
    global PERMUTATION
    effective_seed = seed if seed is not None else get_bilora_seed()
    if dim == 768 and seed is None and PERMUTATION is not None:
        return PERMUTATION
    g = torch.Generator().manual_seed(int(effective_seed))
    perm = torch.randperm(dim * dim, generator=g).tolist()
    if dim == 768 and (seed is None or seed == GLOBAL_BILORA_SEED):
        PERMUTATION = perm
    return perm

# Initialize default permutation with GLOBAL_BILORA_SEED
PERMUTATION = torch.randperm(768 * 768, generator=torch.Generator().manual_seed(GLOBAL_BILORA_SEED)).tolist()

def init_weights_kaiming(m):
    # Kiểm tra nếu module là lớp Linear
    if isinstance(m, nn.Linear):
        # Khởi tạo Weight bằng Kaiming Normal
        # nonlinearity='relu' giúp PyTorch tính toán độ lệch chuẩn tối ưu
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
        # Khởi tạo Bias về 0 (để tránh làm lệch tín hiệu ban đầu)
        if m.bias is not None:
            init.constant_(m.bias, 0)

class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task > -0.5:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v
    

def FFT_SHIFT(matrix):
        m_clone = matrix.clone()
        m,n = m_clone.shape
        m = int(m / 2)
        n = int(n / 2)

        for i in range(m):
            for j in range(n):
                m_clone[i][j] = matrix[m+i][n+j]
                m_clone[m+i][n+j] = matrix[i][j]
                m_clone[m+i][j] = matrix[i][j+n]
                m_clone[i][j+n] = matrix[m+i][j]
        return m_clone        

class Attention_LoRA_FFT_Separate(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, indices=None):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)

        self.n_frq = n_frq
        self.coef_k = nn.Parameter(torch.randn(self.n_frq), requires_grad=True).to(self.qkv.weight.device)
        self.coef_v = nn.Parameter(torch.randn(self.n_frq), requires_grad=True).to(self.qkv.weight.device)
        self.indices = indices
    
    def select_pos(self, t, dim, seed=None):
        if seed is None:
            seed = get_bilora_seed()
        indices = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(seed+t*10))[:self.n_frq]
        indices = torch.stack([indices // dim, indices % dim], dim=0)
        return indices
    
    def get_delta_w_k(self, alpha=300):
        indices = self.indices
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_k
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
    
    def get_delta_w_v(self, alpha=300):
        indices = self.indices
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_v
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        weight_k = self.get_delta_w_k()
        weight_v = self.get_delta_w_v()
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_LoRA_FFT_Aggregate(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, list_indices=None):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)

        self.n_frq = n_frq
        self.list_indices = list_indices
        self.current_task = len(list_indices)
        self.coef_k = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(self.current_task)]).to(self.qkv.weight.device)
        self.coef_v = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(self.current_task)]).to(self.qkv.weight.device)
        

    
    def get_delta_w_k(self, task, alpha=300):
        indices = self.list_indices[task]
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_k[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
    
    def get_delta_w_v(self, task, alpha=300):
        indices = self.list_indices[task]
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_v[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        weight_k = torch.stack([self.get_delta_w_k(t) for t in range(self.current_task)], dim=0).sum(dim=0)
        weight_v = torch.stack([self.get_delta_w_v(t) for t in range(self.current_task)], dim=0).sum(dim=0)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BiLoRA_Manager(object):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000, device="cuda", mode="separate", **kwargs):
        self.dim = dim
        self.device = device
        self.n_tasks = n_tasks
        self.n_frq = n_frq
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.r = r
        seed = kwargs.get('seed', get_bilora_seed())
        self.full_permutation = torch.randperm(
            dim * dim, 
            generator=torch.Generator().manual_seed(int(seed))
        )
        self.indices = [self.select_pos(t).to(self.device) for t in range(n_tasks)]
        self.weights = []
        self.mode = mode

    def get_bilora_attn_separate(self, task):
        indices = self.indices[task]
        return Attention_LoRA_FFT_Separate(dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                  qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                                  r=self.r, n_tasks=self.n_tasks, n_frq=self.n_frq, indices=indices).to(self.device)
    
    def get_bilora_attn_aggregate(self, task):
        list_indices = self.indices[:(task + 1)]
        return Attention_LoRA_FFT_Aggregate(dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                  qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                                  r=self.r, n_tasks=self.n_tasks, n_frq=self.n_frq, list_indices=list_indices).to(self.device)
        
    
    def get_bilora_attn(self, task):
        if self.mode == "separate":
            return self.get_bilora_attn_separate(task=task)
        elif self.mode == "aggregate":
            return self.get_bilora_attn_aggregate(task=task)
        
    def select_pos(self, t):
        """
        t: index của tập dữ liệu (ví dụ 0, 1, 2... 9)
        """
        start = t * self.n_frq
        end = (t + 1) * self.n_frq
        indices = self.full_permutation[start:end]
        indices_2d = torch.stack([indices // self.dim, indices % self.dim], dim=0)
        return indices_2d

    def save_bilora_attn(self, bilora_attn: Attention_LoRA):
        saved_bilora_attn = copy.deepcopy(bilora_attn)
        self.weights.append(saved_bilora_attn)
        

class BilinearSupportHead(nn.Module):
    """Linear layer W_z mapping diag(z) (R x R) -> support logits (dim x dim).
    Logits = W_L @ diag(z) @ W_R^T + Bias.
    Followed by Sigmoid for binary classification of non-zero entries in S_target.
    """
    def __init__(self, dim, basis_rank):
        super().__init__()
        self.dim = dim
        self.basis_rank = basis_rank
        self.W_L = nn.Parameter(torch.empty(dim, basis_rank))
        self.W_R = nn.Parameter(torch.empty(dim, basis_rank))
        self.bias = nn.Parameter(torch.zeros(dim, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_L)
        nn.init.xavier_uniform_(self.W_R)
        nn.init.zeros_(self.bias)

    def forward(self, diag_z):
        if diag_z.dim() == 1:
            diag_z = torch.diag(diag_z)
        logits = self.W_L @ diag_z @ self.W_R.t() + self.bias
        return logits

    def predict_mask_prob(self, diag_z):
        return torch.sigmoid(self.forward(diag_z))


class UMSConsolidator(nn.Module):
    """Unified Modulation Subspace (UMS).

    Folds each per-task sparse coordinate matrix S_t into ONE consolidated operator:
        S_t --(truncated SVD + residual test)--> online shared basis (P, Q) --> coordinates z_t
              --> a shared orthogonal rotation R_o (Cayley parametrization) aligns the operator.
    Readout for a task is
        w_t = iFFT( R_o P diag(z_t) Q^T R_o^T ).real * alpha.
    Only (P, Q, W_rot, z_t) persist; the per-task matrix S_t is discarded after folding.

    Anti-interference: basis coordinates added for a new task are private to that task,
    while early (shared) coordinates form the common subspace; canonical blocks are
    therefore (near) disjoint by construction. R_o is the single learned global
    orientation that minimizes reconstruction across tasks.
    """

    def __init__(self, dim, basis_rank=48, svd_rank=8, tau=1e-3, rot_steps=15,
                 rot_lr=0.005, head_lr=0.05, alpha=300., name="UMS",
                 rot_lambda_diag=0.1, rot_lambda_anchor=0.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.name = name
        self.R = int(basis_rank)
        self.r = int(svd_rank)
        self.tau = float(tau)
        self.rot_steps = int(rot_steps)
        self.rot_lr = float(rot_lr)
        self.head_lr = float(kwargs.get('head_lr', head_lr))
        self.alpha = float(alpha)
        self.rot_lambda_diag = float(kwargs.get('rot_lambda_diag', rot_lambda_diag))
        self.rot_lambda_anchor = float(kwargs.get('rot_lambda_anchor', rot_lambda_anchor))
        self.rot_lambda_support_mse = float(kwargs.get('rot_lambda_support_mse', 1.0))
        self.rot_lambda_support_bce = float(kwargs.get('rot_lambda_support_bce', 0.1))
        self.rot_lambda_gaussian = float(kwargs.get('rot_lambda_gaussian', 0.01))
        self.n_frq = int(kwargs.get('n_frq', 576))

        # Pos weight for binary classification: pos_weight = (total_entries - pos_entries) / pos_entries
        # Exact calculation directly from configuration at initialization
        total_entries = self.dim * self.dim
        pos_entries = self.n_frq
        pos_weight_val = float((total_entries - pos_entries) / max(pos_entries, 1))
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val], dtype=torch.float32))

        # Linear network W_z for support classification (dim x dim from diag(z))
        self.support_head = BilinearSupportHead(dim=self.dim, basis_rank=self.R)
        for p in self.support_head.parameters():
            p.requires_grad = False

        self.r_cur = 0
        self._readout_logged = set()
        self.fold_history = []
        self._last_singular_values = []
        self._last_refinement_metrics = {}
        # CPU float16 references are diagnostics only. They let us measure the
        # cumulative drift of every old decoded modulation from the moment it
        # was consolidated without consuming GPU memory.
        self._reference_spectral_readouts = []

        # shared orthonormal bases, grown online
        self.register_buffer('P', torch.zeros(dim, self.R))
        self.register_buffer('Q', torch.zeros(dim, self.R))
        # Cayley rotation generator; A = W - W^T (skew), R_o = (I - A/2)^{-1}(I + A/2)
        self.W_rot = nn.Parameter(torch.zeros(dim, dim), requires_grad=False)
        # one coordinate vector per folded task
        self.z_store = nn.ParameterList()

    # ---- rotation ----
    def rotation(self):
        A = self.W_rot - self.W_rot.t()
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I - A / 2.0, I + A / 2.0)

    # ---- basis expansion (incremental orthonormalization of residual directions) ----
    def _expand_basis(self, left_res, right_res):
        r_old = self.r_cur
        @torch.no_grad()
        def _add(buf, vecs):
            if vecs is None or vecs.shape[1] == 0:
                return
            if self.r_cur > 0:
                B = buf[:, :self.r_cur]
                vecs = vecs - B @ (B.t() @ vecs)
            q, _ = torch.linalg.qr(vecs)
            free = self.R - self.r_cur
            take = min(free, q.shape[1])
            if take > 0:
                buf[:, self.r_cur:self.r_cur + take] = q[:, :take]
        # grow both sides together, one QR round
        _add(self.P, left_res)
        _add(self.Q, right_res)
        self.r_cur = int((self.P.norm(dim=0) > 1e-8).sum().item())
        added = self.r_cur - r_old
        print(f"[{self.name}] Basis expansion: {r_old} -> {self.r_cur}/{self.R} (+{added} cols added, remaining free={self.R - self.r_cur})")

    @torch.no_grad()
    def get_candidate_basis(self, T):
        """Returns candidate orthonormal basis (P_cand, Q_cand) for task canonical matrix T.
        Combines current basis (if r_cur > 0) with top residual SVD directions, capped at self.R.
        Does NOT modify internal self.P and self.Q buffers.
        """
        r_cur = self.r_cur
        device = self.P.device
        dtype = self.P.dtype
        if r_cur > 0:
            Bp, Bq = self.P[:, :r_cur].clone(), self.Q[:, :r_cur].clone()
            Up = T - Bp @ (Bp.t() @ T)
            Vp = T - (T @ Bq) @ Bq.t()
        else:
            Bp, Bq = None, None
            Up, Vp = T, T

        Ur, Dr, Vhr = torch.linalg.svd(Up, full_matrices=False)
        r = min(self.r, Ur.shape[1])
        left_res, right_in = Ur[:, :r], Vhr[:r, :].t()
        if r_cur > 0:
            left_res = left_res - Bp @ (Bp.t() @ left_res)
            right_in = right_in - Bq @ (Bq.t() @ right_in)

        # Orthonormalize residual components via QR
        q_p, _ = torch.linalg.qr(left_res)
        q_q, _ = torch.linalg.qr(right_in)
        free = self.R - r_cur
        take = max(0, min(free, self.r, q_p.shape[1], q_q.shape[1]))
        q_p = q_p[:, :take]
        q_q = q_q[:, :take]

        if r_cur > 0 and take > 0:
            P_cand = torch.cat([Bp, q_p], dim=1)
            Q_cand = torch.cat([Bq, q_q], dim=1)
        elif r_cur > 0:
            P_cand = Bp
            Q_cand = Bq
        elif take > 0:
            P_cand = q_p
            Q_cand = q_q
        else:
            P_cand = torch.empty((self.dim, 0), device=device, dtype=dtype)
            Q_cand = torch.empty((self.dim, 0), device=device, dtype=dtype)
        return P_cand, Q_cand

    # ---- encode one canonical-frame task into (basis, coordinates) ----
    def _encode(self, task_id, T, verbose=True, allow_expand=True):
        r_cur = self.r_cur
        if r_cur > 0:
            Bp, Bq = self.P[:, :r_cur], self.Q[:, :r_cur]
            Up = T - Bp @ (Bp.t() @ T)
            Vp = T - (T @ Bq) @ Bq.t()
        else:
            Up, Vp = T, T
        # residual singular structure
        Ur, Dr, Vhr = torch.linalg.svd(Up, full_matrices=False)
        r = min(self.r, Ur.shape[1])
        left_res, d_r, right_in = Ur[:, :r], Dr[:r], Vhr[:r, :].t()
        if r_cur > 0:
            right_in = right_in - Bq @ (Bq.t() @ right_in)
        energy = (left_res * d_r).pow(2).sum().item()
        self._last_singular_values = [float(s.item()) for s in Dr[:min(8, len(Dr))]]

        if verbose:
            top_svals = ", ".join([f"{s.item():.3e}" for s in Dr[:min(4, len(Dr))]])
            print(f"[{self.name}] Task {task_id} Residual SVD: top_σ=[{top_svals}], energy={energy:.3e} (threshold τ={self.tau:.3e})")

        if allow_expand:
            if energy > self.tau and self.r_cur < self.R:
                if verbose:
                    print(f"[{self.name}] Energy ({energy:.3e}) > τ ({self.tau:.3e}) -> Expanding shared basis...")
                self._expand_basis(left_res * d_r, right_in * d_r)
            elif verbose and energy <= self.tau:
                print(f"[{self.name}] Energy ({energy:.3e}) <= τ ({self.tau:.3e}) -> Reusing existing subspace (0 cols added).")
            elif verbose and self.r_cur >= self.R:
                print(f"[{self.name}] Basis capacity reached limit ({self.r_cur}/{self.R}) -> Projecting into existing subspace.")

        # coordinates = diagonal of P^T T Q
        r_cur = self.r_cur
        z_full = torch.zeros(self.R, device=T.device, dtype=T.dtype)
        if r_cur > 0:
            Zm = self.P[:, :r_cur].t() @ T @ self.Q[:, :r_cur]
            z_full[:r_cur] = torch.diagonal(Zm)
        z = nn.Parameter(z_full, requires_grad=False)
        if task_id < len(self.z_store):
            self.z_store[task_id] = z
        else:
            assert task_id == len(self.z_store), "tasks must be folded in order"
            self.z_store.append(z)
        return energy

    # ---- refine the shared rotation to fit the current task ----
    def _refine_rotation(self, S_target, task_id, r_old=None, R_prev=None, P_prev=None, Q_prev=None):
        if self.rot_steps <= 0 or self.r_cur == 0:
            self._last_refinement_metrics = {
                'steps': 0,
                'anchor_enabled': False,
                'initial_total_loss': None,
                'final_total_loss': None,
                'final_unweighted_losses': {},
                'refinement_trace': [],
            }
            return self._last_refinement_metrics
        if R_prev is None and task_id > 0:
            R_prev = self.rotation().detach()
        if r_old is None:
            r_old = 0 if task_id == 0 else self.r_cur

        self.W_rot.requires_grad_(True)
        for p in self.support_head.parameters():
            p.requires_grad_(True)

        try:
            with torch.enable_grad():
                z = self.z_store[task_id].detach()
                r_cur = self.r_cur
                opt = torch.optim.Adam([
                    {'params': [self.W_rot], 'lr': self.rot_lr},
                    {'params': self.support_head.parameters(), 'lr': self.head_lr},
                ])
                init_loss, final_loss = None, None
                last_support_mse, last_support_bce, last_gaussian = 0.0, 0.0, 0.0
                last_off_diag, last_drift = 0.0, 0.0
                refinement_trace = []

                has_anchor = (task_id > 0) and (r_old > 0) and (R_prev is not None)
                if has_anchor:
                    P_old = P_prev if P_prev is not None else self.P[:, :r_old].detach()
                    Q_old = Q_prev if Q_prev is not None else self.Q[:, :r_old].detach()
                    # detach() alone is a view and would follow the in-place Adam
                    # updates. The drift anchor needs independent old snapshots.
                    W_L_old = self.support_head.W_L[:, :r_old].detach().clone()
                    W_R_old = self.support_head.W_R[:, :r_old].detach().clone()

                pos_mask = (S_target != 0)
                target_binary = pos_mask.to(dtype=S_target.dtype)
                diag_z = torch.diag(z)

                for step in range(self.rot_steps):
                    def closure():
                        nonlocal last_support_mse, last_support_bce, last_gaussian, last_off_diag, last_drift
                        opt.zero_grad()
                        R = self.rotation()

                        # 1) Binary classification loss for non-zero support prediction (W_z on diag(z) + Sigmoid)
                        logits_support = self.support_head(diag_z)
                        loss_support_bce = F.binary_cross_entropy_with_logits(
                            logits_support, target_binary, pos_weight=self.pos_weight
                        )

                        # 2) Task-specific reconstruction and Gaussian prior on true non-zero entries
                        C = self.P @ diag_z @ self.Q.t()
                        recon = R @ C @ R.t()

                        if pos_mask.any():
                            # MSE loss calculated ONLY on positions where S_target is truly non-zero
                            loss_support_mse = (S_target[pos_mask] - recon[pos_mask]).pow(2).mean()
                            # Gaussian prior loss: ensure prediction entries at non-zero positions follow N(0, 1)
                            pred_support = recon[pos_mask]
                            mu_pred = pred_support.mean()
                            std_pred = torch.sqrt(pred_support.var(unbiased=False) + 1e-8)
                            loss_gaussian = mu_pred.pow(2) + (std_pred - 1.0).pow(2)
                        else:
                            loss_support_mse = torch.zeros((), device=S_target.device, dtype=S_target.dtype)
                            loss_gaussian = torch.zeros((), device=S_target.device, dtype=S_target.dtype)

                        # 3) Off-diagonal suppression loss (joint diagonalization)
                        T = R.t() @ S_target @ R
                        Zm = self.P[:, :r_cur].t() @ T @ self.Q[:, :r_cur]
                        diag_Zm = torch.diagonal(Zm)
                        off_diag_Zm = Zm - torch.diag_embed(diag_Zm)
                        denom_off_diag = max(r_cur * (r_cur - 1), 1)
                        loss_off_diag = off_diag_Zm.pow(2).sum() / denom_off_diag

                        # 4) Subspace drift constraint (anchor rotation and support head to previous basis)
                        if has_anchor:
                            diff_P = (R - R_prev) @ P_old
                            diff_Q = (R - R_prev) @ Q_old
                            loss_rot_drift = (diff_P.pow(2).sum() + diff_Q.pow(2).sum()) / (2.0 * r_old)
                            diff_WL = self.support_head.W_L[:, :r_old] - W_L_old
                            diff_WR = self.support_head.W_R[:, :r_old] - W_R_old
                            loss_head_drift = (diff_WL.pow(2).sum() + diff_WR.pow(2).sum()) / (2.0 * r_old)
                            loss_drift = loss_rot_drift + loss_head_drift
                        else:
                            loss_drift = torch.zeros((), device=S_target.device, dtype=S_target.dtype)

                        total_loss = (
                            self.rot_lambda_support_mse * loss_support_mse
                            + self.rot_lambda_support_bce * loss_support_bce
                            + self.rot_lambda_gaussian * loss_gaussian
                            + self.rot_lambda_diag * loss_off_diag
                            + (self.rot_lambda_anchor * loss_drift if has_anchor else 0.0)
                        )

                        last_support_mse = loss_support_mse.item()
                        last_support_bce = loss_support_bce.item()
                        last_gaussian = loss_gaussian.item()
                        last_off_diag = loss_off_diag.item()
                        last_drift = loss_drift.item() if has_anchor else 0.0

                        total_loss.backward()
                        return total_loss

                    loss_val = opt.step(closure).item()
                    if init_loss is None:
                        init_loss = loss_val
                    final_loss = loss_val

                    grad_norms = {
                        'W_rot': float(self.W_rot.grad.norm().item()) if self.W_rot.grad is not None else 0.0,
                        'W_L': float(self.support_head.W_L.grad.norm().item()) if self.support_head.W_L.grad is not None else 0.0,
                        'W_R': float(self.support_head.W_R.grad.norm().item()) if self.support_head.W_R.grad is not None else 0.0,
                        'bias': float(self.support_head.bias.grad.norm().item()) if self.support_head.bias.grad is not None else 0.0,
                    }
                    refinement_trace.append({
                        'step': int(step + 1),
                        'total_loss': float(loss_val),
                        'unweighted': {
                            'value': float(last_support_mse),
                            'mask': float(last_support_bce),
                            'moment': float(last_gaussian),
                            'off_diagonal': float(last_off_diag),
                            'drift': float(last_drift),
                        },
                        'weighted': {
                            'value': float(self.rot_lambda_support_mse * last_support_mse),
                            'mask': float(self.rot_lambda_support_bce * last_support_bce),
                            'moment': float(self.rot_lambda_gaussian * last_gaussian),
                            'off_diagonal': float(self.rot_lambda_diag * last_off_diag),
                            'drift': float(self.rot_lambda_anchor * last_drift) if has_anchor else 0.0,
                        },
                        'gradient_norms': grad_norms,
                    })

                    print(f"[{self.name}] [Rotation Step {step + 1:2d}/{self.rot_steps}] "
                          f"Loss: {loss_val:.4e} "
                          f"(mse_supp={last_support_mse:.4e}, bce_supp={last_support_bce:.4e}, "
                          f"gauss={last_gaussian:.4e}, off_diag={last_off_diag:.4e}, drift={last_drift:.4e})")
        finally:
            self.W_rot.requires_grad_(False)
            for p in self.support_head.parameters():
                p.requires_grad_(False)

        with torch.no_grad():
            R = self.rotation()
            # Re-project coordinates under the refined rotation R
            T_refined = R.t() @ S_target @ R
            Zm = self.P[:, :r_cur].t() @ T_refined @ self.Q[:, :r_cur]
            self.z_store[task_id].data[:r_cur] = torch.diagonal(Zm)
            z = self.z_store[task_id].detach()

            C = self.P @ torch.diag(z) @ self.Q.t()
            recon = R @ C @ R.t()
            probs = self.support_head.predict_mask_prob(torch.diag(z))
            recon_gated = torch.where(probs >= 0.5, recon, torch.zeros_like(recon))
            rel_err = ((S_target - recon_gated).norm() / (S_target.norm() + 1e-12)).item()
            raw_rel_err = ((S_target - recon).norm() / (S_target.norm() + 1e-12)).item()
        print(f"[{self.name}] Rotation refined ({self.rot_steps} steps): "
              f"Loss {init_loss:.3e} -> {final_loss:.3e} "
              f"(mse_supp={last_support_mse:.3e}, bce_supp={last_support_bce:.3e}, gauss={last_gaussian:.3e}, off_diag={last_off_diag:.3e}, drift={last_drift:.3e}), "
              f"rel_recon_err={rel_err:.2%} (raw={raw_rel_err:.2%})")
        self._last_refinement_metrics = {
            'steps': int(self.rot_steps),
            'anchor_enabled': bool(has_anchor),
            'initial_total_loss': float(init_loss),
            'final_total_loss': float(final_loss),
            'final_unweighted_losses': {
                'value': float(last_support_mse),
                'mask': float(last_support_bce),
                'moment': float(last_gaussian),
                'off_diagonal': float(last_off_diag),
                'drift': float(last_drift),
            },
            'refinement_trace': refinement_trace,
        }
        return self._last_refinement_metrics

    @torch.no_grad()
    def _fold_diagnostics(self, S_target, task_id):
        """Compute mechanism diagnostics for a single folded task."""
        z = self.z_store[task_id].detach()
        R = self.rotation()
        C = self.P @ torch.diag(z) @ self.Q.t()
        recon = R @ C @ R.t()
        probs = self.support_head.predict_mask_prob(torch.diag(z))
        pred_mask = probs >= 0.5
        target_mask = S_target != 0
        recon_gated = torch.where(pred_mask, recon, torch.zeros_like(recon))
        eps = 1e-12

        tp = int((pred_mask & target_mask).sum().item())
        fp = int((pred_mask & ~target_mask).sum().item())
        fn = int((~pred_mask & target_mask).sum().item())
        tn = int((~pred_mask & ~target_mask).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)
        iou = tp / max(tp + fp + fn, 1)

        r_cur = self.r_cur
        T = R.t() @ S_target @ R
        Zm = self.P[:, :r_cur].t() @ T @ self.Q[:, :r_cur]
        off = Zm - torch.diag_embed(torch.diagonal(Zm))
        off_ratio = float((off.pow(2).sum() / (Zm.pow(2).sum() + eps)).item())

        if target_mask.any():
            pred_values = recon[target_mask]
            target_values = S_target[target_mask]
            pred_mean = float(pred_values.mean().item())
            pred_std = float(torch.sqrt(pred_values.var(unbiased=False) + 1e-8).item())
            target_mean = float(target_values.mean().item())
            target_std = float(torch.sqrt(target_values.var(unbiased=False) + 1e-8).item())
        else:
            pred_mean = pred_std = target_mean = target_std = 0.0

        return {
            'raw_relative_recon_error': float(((S_target - recon).norm() / (S_target.norm() + eps)).item()),
            'gated_relative_recon_error': float(((S_target - recon_gated).norm() / (S_target.norm() + eps)).item()),
            'support_threshold': 0.5,
            'empty_support_convention': 'zero',
            'support_precision': float(precision),
            'support_recall': float(recall),
            'support_f1': float(f1),
            'support_iou': float(iou),
            'support_false_positive_rate': float(fp / max(fp + tn, 1)),
            'support_false_negative_rate': float(fn / max(fn + tp, 1)),
            'support_counts': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'off_diagonal_energy_ratio': off_ratio,
            'pred_support_mean': pred_mean,
            'pred_support_std': pred_std,
            'target_support_mean': target_mean,
            'target_support_std': target_std,
            'prior_mean_abs_error': float(abs(pred_mean)),
            'prior_std_abs_error': float(abs(pred_std - 1.0)),
            'target_mean_abs_error': float(abs(pred_mean - target_mean)),
            'target_std_abs_error': float(abs(pred_std - target_std)),
        }

    @torch.no_grad()
    def _spectral_readout(self, task_id):
        z = self.z_store[task_id].detach()
        R = self.rotation()
        C = self.P @ torch.diag(z) @ self.Q.t()
        spectral = R @ C @ R.t()
        probs = self.support_head.predict_mask_prob(torch.diag(z))
        return torch.where(probs >= 0.5, spectral, torch.zeros_like(spectral))

    # ---- public: fold one task ----
    def fold(self, task_id, S_t):
        S = S_t.detach().to(self.P.device, dtype=self.P.dtype)
        s_norm = S.norm().item()
        nnz = int((S != 0).sum().item())
        sparsity = (1.0 - nnz / S.numel()) * 100
        print(f"\n>>> [{self.name}] START FOLDING TASK {task_id} (||S_{task_id}||_F={s_norm:.2f}, nnz={nnz}/{S.numel()}, sparsity={sparsity:.1f}%)")
        r_before = self.r_cur
        P_anchor = self.P[:, :r_before].clone().detach() if r_before > 0 else None
        Q_anchor = self.Q[:, :r_before].clone().detach() if r_before > 0 else None
        # deterministic encoding (basis/coordinates carry no graph)
        with torch.no_grad():
            R_init = self.rotation().detach()
            T = R_init.t() @ S @ R_init                       # canonical frame
            self._encode(task_id, T, verbose=True, allow_expand=True)
            pre_refine = self._fold_diagnostics(S, task_id)
        # only the shared rotation is learned (with multi-component loss and backward anchor)
        refinement = self._refine_rotation(
            S, task_id, r_old=r_before, R_prev=R_init,
            P_prev=P_anchor, Q_prev=Q_anchor,
        )
        with torch.no_grad():
            R = self.rotation()                      # re-encode after rotation
            T = R.t() @ S @ R
            energy = self._encode(task_id, T, verbose=False, allow_expand=False)

        z = self.z_store[task_id]
        with torch.no_grad():
            C = self.P @ torch.diag(z) @ self.Q.t()
            recon = R @ C @ R.t()
            probs = self.support_head.predict_mask_prob(torch.diag(z))
            recon_gated = torch.where(probs >= 0.5, recon, torch.zeros_like(recon))
            rel_err = ((S - recon_gated).norm() / (S.norm() + 1e-12)).item()
            raw_rel_err = ((S - recon).norm() / (S.norm() + 1e-12)).item()
            post_refine = self._fold_diagnostics(S, task_id)

            old_task_drift = {}
            for old_task_id, reference_cpu in enumerate(self._reference_spectral_readouts):
                current_old = self._spectral_readout(old_task_id)
                reference = reference_cpu.to(device=current_old.device, dtype=current_old.dtype)
                drift = (current_old - reference).norm() / (reference.norm() + 1e-12)
                old_task_drift[f'task_{old_task_id}'] = float(drift.item())

            current_reference = self._spectral_readout(task_id).detach().to(
                device='cpu', dtype=torch.float16
            ).clone()
            if task_id < len(self._reference_spectral_readouts):
                self._reference_spectral_readouts[task_id] = current_reference
            else:
                self._reference_spectral_readouts.append(current_reference)

        metrics = {
            'task_id': int(task_id),
            'tasks_folded': len(self.z_store),
            'basis_rank_used': self.r_cur,
            'basis_capacity': self.R,
            'added_basis_cols': int(self.r_cur - r_before),
            'residual_energy': float(energy),
            'block_coherence': float(self.coherence()),
            'relative_recon_error': float(rel_err),
            'raw_relative_recon_error': float(raw_rel_err),
            'pre_refinement': pre_refine,
            'post_refinement': post_refine,
            'refinement': copy.deepcopy(refinement),
            'old_task_modulation_drift': old_task_drift,
            'avg_old_task_modulation_drift': float(sum(old_task_drift.values()) / len(old_task_drift)) if old_task_drift else 0.0,
            'max_old_task_modulation_drift': float(max(old_task_drift.values())) if old_task_drift else 0.0,
            'drift_reference_dtype': 'float16_cpu',
            'z_norm': float(z.norm().item()),
            's_norm': float(s_norm),
            'sparsity': float(sparsity),
            'nnz': int(nnz),
            'residual_singular_values': list(getattr(self, '_last_singular_values', [])),
        }
        self.fold_history.append(metrics)
        print(f"<<< [{self.name}] FINISHED TASK {task_id}: basis={self.r_cur}/{self.R} ({self.r_cur/self.R*100:.1f}%), "
              f"||z_{task_id}||_2={z.norm().item():.3f}, rel_recon_err={rel_err:.2%} (raw={raw_rel_err:.2%}), coherence={metrics['block_coherence']:.3e}\n")
        return metrics

    # ---- canonical-block coherence (diagnostic) ----
    def coherence(self):
        n = len(self.z_store)
        if n < 2 or self.r_cur == 0:
            return torch.zeros((), device=self.P.device)
        blocks = []
        for u in range(n):
            C = self.P @ torch.diag(self.z_store[u]) @ self.Q.t()
            blocks.append(C / (C.norm() + 1e-12))
        tot = 0.0
        cnt = 0
        for u in range(n):
            for v in range(u + 1, n):
                tot = tot + (blocks[u].t() @ blocks[v]).pow(2).sum().sqrt()
                cnt += 1
        return tot / max(cnt, 1)

    # ---- public: dense modulation for one task, shape (E, E) ----
    def readout(self, task_id):
        z = self.z_store[task_id]
        R = self.rotation()
        C = self.P @ torch.diag(z) @ self.Q.t()
        S_view = R @ C @ R.t()
        with torch.no_grad():
            probs = self.support_head.predict_mask_prob(torch.diag(z))
            S_view = torch.where(probs >= 0.5, S_view, torch.zeros_like(S_view))
        w = torch.fft.ifft2(S_view, dim=(-2, -1)).real * self.alpha
        if task_id not in self._readout_logged:
            self._readout_logged.add(task_id)
            print(f"[{self.name}] Readout Task {task_id}: in-memory modulation restored (||w_{task_id}||_F={w.norm().item():.2f}, alpha={self.alpha})")
        return w

    # ---- analytical metrics extraction for research & improvement ----
    def get_analysis_metrics(self, task_id=None):
        r = self.r_cur
        device = self.P.device
        with torch.no_grad():
            if r > 0:
                p_act = self.P[:, :r]
                q_act = self.Q[:, :r]
                eye_r = torch.eye(r, device=device)
                p_ortho_err = (p_act.t() @ p_act - eye_r).norm().item()
                q_ortho_err = (q_act.t() @ q_act - eye_r).norm().item()
            else:
                p_ortho_err = 0.0
                q_ortho_err = 0.0

            R_o = self.rotation()
            eye_dim = torch.eye(self.dim, device=device)
            rot_ortho_err = (R_o.t() @ R_o - eye_dim).norm().item()
            rot_dev = (R_o - eye_dim).norm().item()

            z_stats = {}
            for t_idx, z in enumerate(self.z_store):
                z_vec = z.detach()
                l2 = z_vec.norm().item()
                l1 = z_vec.abs().sum().item()
                active_dims = int((z_vec.abs() > 1e-4).sum().item())
                sparsity = float((z_vec.abs() <= 1e-5).sum().item() / max(len(z_vec), 1) * 100)
                vals, idxs = torch.topk(z_vec.abs(), min(5, len(z_vec)))
                top_comps = [{"index": int(i), "value": float(z_vec[i].item())} for i in idxs]
                z_stats[f"task_{t_idx}"] = {
                    "l2_norm": float(l2),
                    "l1_norm": float(l1),
                    "active_dimensions": active_dims,
                    "sparsity_pct": sparsity,
                    "top_components": top_comps,
                }

            z_similarities = {}
            n_tasks = len(self.z_store)
            if n_tasks >= 2:
                for i in range(n_tasks):
                    for j in range(i + 1, n_tasks):
                        zi = self.z_store[i].detach()
                        zj = self.z_store[j].detach()
                        denom = (zi.norm() * zj.norm()).item()
                        sim = (zi @ zj).item() / (denom + 1e-12) if denom > 0 else 0.0
                        z_similarities[f"task_{i}_vs_task_{j}"] = float(sim)

            coherence_matrix = {}
            if n_tasks >= 2 and r > 0:
                blocks = []
                for u in range(n_tasks):
                    C_u = self.P @ torch.diag(self.z_store[u]) @ self.Q.t()
                    blocks.append(C_u / (C_u.norm() + 1e-12))
                for u in range(n_tasks):
                    for v in range(u + 1, n_tasks):
                        coh = (blocks[u].t() @ blocks[v]).pow(2).sum().sqrt().item()
                        coherence_matrix[f"task_{u}_vs_task_{v}"] = float(coh)

        return {
            "name": self.name,
            "dim": self.dim,
            "basis_capacity_R": self.R,
            "basis_rank_used_r_cur": self.r_cur,
            "basis_capacity_used_pct": float(self.r_cur / max(self.R, 1) * 100),
            "tasks_folded": len(self.z_store),
            "basis_orthonormality_error_P": float(p_ortho_err),
            "basis_orthonormality_error_Q": float(q_ortho_err),
            "rotation_orthogonality_error": float(rot_ortho_err),
            "rotation_deviation_from_identity": float(rot_dev),
            "overall_block_coherence": float(self.coherence().item()),
            "pairwise_block_coherence": coherence_matrix,
            "coordinates_stats": z_stats,
            "coordinates_cross_similarity": z_similarities,
            "fold_history": copy.deepcopy(self.fold_history),
        }

    # ---- export state dict & tensors ----
    def export_state(self, task_id=None, include_readout=True):
        metrics = self.get_analysis_metrics(task_id=task_id)
        with torch.no_grad():
            tensors = {
                "P": self.P.detach().cpu(),
                "Q": self.Q.detach().cpu(),
                "P_active": self.P[:, :self.r_cur].detach().cpu() if self.r_cur > 0 else torch.empty(self.dim, 0),
                "Q_active": self.Q[:, :self.r_cur].detach().cpu() if self.r_cur > 0 else torch.empty(self.dim, 0),
                "W_rot": self.W_rot.detach().cpu(),
                "R_o": self.rotation().detach().cpu(),
                "z_store": torch.stack([z.detach().cpu() for z in self.z_store]) if len(self.z_store) > 0 else torch.empty(0, self.R),
                "support_head_W_L": self.support_head.W_L.detach().cpu(),
                "support_head_W_R": self.support_head.W_R.detach().cpu(),
                "support_head_bias": self.support_head.bias.detach().cpu(),
            }
            if include_readout and len(self.z_store) > 0:
                t_target = task_id if task_id is not None and task_id < len(self.z_store) else len(self.z_store) - 1
                tensors["w_readout"] = self.readout(t_target).detach().cpu()

        return {
            "name": self.name,
            "config": {
                "dim": self.dim,
                "basis_rank": self.R,
                "svd_rank": self.r,
                "tau": self.tau,
                "rot_steps": self.rot_steps,
                "rot_lr": self.rot_lr,
                "head_lr": self.head_lr,
                "alpha": self.alpha,
                "rot_lambda_diag": self.rot_lambda_diag,
                "rot_lambda_anchor": self.rot_lambda_anchor,
                "rot_lambda_support_mse": self.rot_lambda_support_mse,
                "rot_lambda_support_bce": self.rot_lambda_support_bce,
                "rot_lambda_gaussian": self.rot_lambda_gaussian,
                "n_frq": self.n_frq,
                "pos_weight": float(self.pos_weight.item()),
                "r_cur": self.r_cur,
                "tasks_folded": len(self.z_store),
            },
            "metrics": metrics,
            "tensors": tensors,
        }

    # ---- save UMS to disk: .pt tensor weights and .json analysis diagnostics ----
    def save_state(self, save_dir, task_id=None, include_readout=True):
        import json
        import os
        os.makedirs(save_dir, exist_ok=True)
        exported = self.export_state(task_id=task_id, include_readout=include_readout)

        pt_path = os.path.join(save_dir, f"{self.name}_ums.pt")
        torch.save({
            "name": exported["name"],
            "config": exported["config"],
            "tensors": exported["tensors"],
        }, pt_path)

        json_path = os.path.join(save_dir, f"{self.name}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": exported["name"],
                "config": exported["config"],
                "metrics": exported["metrics"],
            }, f, indent=2)

        return pt_path, json_path

    # ---- load UMS state from checkpoint ----
    def load_state(self, state_dict_or_path, device=None):
        if isinstance(state_dict_or_path, str):
            state = torch.load(state_dict_or_path, map_location=device or self.P.device)
        else:
            state = state_dict_or_path

        tensors = state.get("tensors", state)
        if "P" in tensors:
            self.P.copy_(tensors["P"].to(self.P.device))
        if "Q" in tensors:
            self.Q.copy_(tensors["Q"].to(self.Q.device))
        if "W_rot" in tensors:
            self.W_rot.data.copy_(tensors["W_rot"].to(self.W_rot.device))
        if "support_head_W_L" in tensors:
            self.support_head.W_L.data.copy_(tensors["support_head_W_L"].to(self.support_head.W_L.device))
        if "support_head_W_R" in tensors:
            self.support_head.W_R.data.copy_(tensors["support_head_W_R"].to(self.support_head.W_R.device))
        if "support_head_bias" in tensors:
            self.support_head.bias.data.copy_(tensors["support_head_bias"].to(self.support_head.bias.device))

        if "r_cur" in state.get("config", {}):
            self.r_cur = int(state["config"]["r_cur"])
        else:
            self.r_cur = int((self.P.norm(dim=0) > 1e-8).sum().item())

        if "z_store" in tensors:
            z_data = tensors["z_store"]
            self.z_store = nn.ParameterList()
            for z_vec in z_data:
                self.z_store.append(nn.Parameter(z_vec.to(self.P.device), requires_grad=False))


class MoE(nn.Module):
    def __init__(self, dim, num_experts, n_frq, n_tasks, topk, device=None,
                 use_expert_weights=False, ums_cfg=None, name="MoE", **kwargs):
        super().__init__()
        self.dim = dim
        self.name = name
        ums_cfg = dict(ums_cfg or {})
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_frq = n_frq
        self.n_tasks = n_tasks
        self.topk = topk
        self.num_experts = num_experts
        self.use_expert_weights = use_expert_weights
        self.full_permutation = PERMUTATION.copy()
        self.router = nn.Linear(dim, num_experts).to(self.device)
        self.coeff = nn.Parameter(torch.randn(num_experts, n_frq, device=self.device), requires_grad=True)
        self.expert_weights = torch.zeros((self.topk, ), device=self.device, dtype=torch.float32)
        self.create_bilora_indices()
        self.flag_to_save_lora_indices = False
        self.lora_indices_across_tasks = []
        self.initial_noise_prob = 0.5
        self.current_epoch = 0
        self.register_buffer('expert_mask', torch.zeros(num_experts, device=self.device))
        self.task_expert_history = {}

        # UMS: one consolidated modulation operator; force_task is set only at eval readout
        ums_cfg.setdefault('name', self.name)
        ums_cfg.setdefault('n_frq', self.n_frq)
        self.ums = UMSConsolidator(dim=dim, **ums_cfg).to(self.device)
        self.force_task = None

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, 'coeff'):
            self.device = self.coeff.device
        return self

    def create_bilora_indices(self):
        target_device = self.coeff.device if hasattr(self, 'coeff') else self.device
        list_indices = [torch.tensor(self.full_permutation[t * self.n_frq : (t + 1) * self.n_frq], device=target_device) for t in range(self.num_experts)]
        self.register_buffer('list_indices', torch.stack(list_indices, dim=0), persistent=False)

    def build_task_sparse_matrix(self, expert_ids):
        """Pre-iFFT sparse coordinate matrix S_t owned by the given experts."""
        S = torch.zeros(self.dim, self.dim, device=self.coeff.device, dtype=self.coeff.dtype)
        ids = torch.as_tensor(expert_ids, device=self.coeff.device, dtype=torch.long)
        flat_idx = self.list_indices[ids].reshape(-1)
        vals = self.coeff[ids].reshape(-1)
        S.view(-1).index_add_(0, flat_idx, vals)
        return S

    def set_force_task(self, task):
        self.force_task = None if task is None else int(task)

    def clear_force_task(self):
        self.force_task = None

    def vectorized_forward(self, batch_expert_weights, batch_expert_indices, alpha=300):
        B, topk = batch_expert_indices.shape
        device = self.coeff.device
        dim = self.dim
        all_selected_indices = self.list_indices[batch_expert_indices]
        current_coeffs = self.coeff[batch_expert_indices]
        if self.use_expert_weights:
            weighted_coeffs = current_coeffs * batch_expert_weights.unsqueeze(-1)
            flat_values = weighted_coeffs.view(-1)
        else:
            flat_values = current_coeffs.view(-1)
        aggregated_freq_mask = torch.zeros(B, dim, dim, device=device, dtype=current_coeffs.dtype)
        batch_offsets = torch.arange(B, device=device).view(B, 1, 1) * (dim * dim)
        flat_indices = (all_selected_indices + batch_offsets).view(-1)
        aggregated_freq_mask.view(-1).index_put_(
            (flat_indices,), 
            flat_values, 
            accumulate=True
        )

        final_output = torch.fft.ifft2(aggregated_freq_mask, dim=(-2, -1)).real * alpha
        return final_output
    
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        print("[INFO] Current epoch: ", self.current_epoch)
    
    def forward(self, cls_token, **kwargs):
        # UMS readout: bypass the router and return the consolidated task modulation
        if self.force_task is not None:
            w = self.ums.readout(self.force_task)
            return w.unsqueeze(0).expand(cls_token.shape[0], -1, -1)

        router_logits = self.router(cls_token)

        batch_size = router_logits.size(0)
        decay_noise_prob = self.initial_noise_prob / (self.current_epoch + 1)
        noise_mask = (torch.rand(batch_size, 1, device=router_logits.device) < decay_noise_prob).float()
        
        # Tạo Gaussian noise cùng kích thước với router_logits
        noise = torch.randn_like(router_logits)
        
        # Cộng nhiễu có chọn lọc: chỉ cộng vào các dòng được mask chọn
        router_logits = router_logits + (noise * noise_mask)
        
        masked_router_logits = router_logits + self.expert_mask
        batch_expert_weights, batch_expert_indices = torch.topk(F.softmax(masked_router_logits, dim=-1), self.topk, dim=-1)
        if self.flag_to_save_lora_indices:
            self.lora_indices_across_tasks.append(batch_expert_indices.detach())
        # batch_expert_weights = batch_expert_weights / batch_expert_weights.sum(dim=-1, keepdim=True)
        self.expert_weights = batch_expert_weights.sum(dim=-1)
        alpha = 300
        outputs = self.vectorized_forward(batch_expert_weights, batch_expert_indices, alpha=alpha)
        return outputs

    def end_task(self, task_id=None):
        assert len(self.lora_indices_across_tasks) > 0, "[ERROR] LoRA indices across tasks does not contains anything"
        all_indices = torch.cat(self.lora_indices_across_tasks, dim=0).view(-1)
        expert_counts = torch.bincount(all_indices, minlength=self.num_experts)
        sorted_counts, sorted_expert_indices = torch.sort(expert_counts, descending=True)
        num_active_experts = (sorted_counts > 0).sum().item()
        allocated_limit = self.num_experts // self.n_tasks
        M = min(num_active_experts, allocated_limit)
        M = max(1, M) if num_active_experts > 0 else 0
        top_M_experts = sorted_expert_indices[:M].tolist()
        fold_metrics = None
        s_norm, nnz, sparsity = 0.0, 0, 0.0
        fold_id = task_id if task_id is not None else len(self.ums.z_store)
        if top_M_experts:
            indices_to_mask = torch.tensor(top_M_experts, dtype=torch.long, device=self.expert_mask.device)
            self.expert_mask[indices_to_mask] = float('-inf')
            print(f"[{self.name}] Masked {len(top_M_experts)} active experts (active: {num_active_experts}/{self.num_experts}, quota: {allocated_limit}): {top_M_experts[:10]}{'...' if len(top_M_experts) > 10 else ''}")
            # UMS folding: build this task's sparse matrix and consolidate it
            S_t = self.build_task_sparse_matrix(top_M_experts)
            s_norm = S_t.norm().item()
            nnz = int((S_t != 0).sum().item())
            sparsity = (1.0 - nnz / S_t.numel()) * 100
            fold_metrics = self.ums.fold(fold_id, S_t)

        self.task_expert_history[int(fold_id)] = {
            'task_id': int(fold_id),
            'allocated_experts': top_M_experts,
            'expert_counts': expert_counts.detach().cpu().tolist(),
            'num_active_experts': int(num_active_experts),
            'allocated_limit': int(allocated_limit),
            'S_norm': float(s_norm),
            'S_sparsity': float(sparsity),
            'S_nnz': int(nnz),
            'ums_metrics': fold_metrics
        }
        self.lora_indices_across_tasks.clear()

    # ---- MoE & expert analytical metrics ----
    def get_analysis_metrics(self, task_id=None):
        import numpy as np
        with torch.no_grad():
            masked_count = int((self.expert_mask == float('-inf')).sum().item())
            remaining_count = self.num_experts - masked_count
            mask_pct = float(masked_count / max(self.num_experts, 1) * 100)

            coeff_norms = self.coeff.norm(dim=1).detach().cpu().tolist()
            norms_arr = np.array(coeff_norms)
            norm_stats = {
                "mean": float(norms_arr.mean()),
                "std": float(norms_arr.std()),
                "min": float(norms_arr.min()),
                "max": float(norms_arr.max()),
            }

            router_norm = float(self.router.weight.norm().item())
            router_bias_norm = float(self.router.bias.norm().item()) if self.router.bias is not None else 0.0

        return {
            "name": self.name,
            "dim": self.dim,
            "num_experts": self.num_experts,
            "n_frq": self.n_frq,
            "topk": self.topk,
            "n_tasks": self.n_tasks,
            "masked_experts_count": masked_count,
            "remaining_experts_count": remaining_count,
            "expert_mask_pct": mask_pct,
            "expert_coeff_norms_summary": norm_stats,
            "expert_coeff_norms": coeff_norms,
            "router_weight_norm": router_norm,
            "router_bias_norm": router_bias_norm,
            "task_history": copy.deepcopy(getattr(self, 'task_expert_history', {})),
        }

    # ---- export MoE state and parameters ----
    def export_state(self):
        metrics = self.get_analysis_metrics()
        with torch.no_grad():
            tensors = {
                "router_weight": self.router.weight.detach().cpu(),
                "router_bias": self.router.bias.detach().cpu() if self.router.bias is not None else None,
                "coeff": self.coeff.detach().cpu(),
                "expert_mask": self.expert_mask.detach().cpu(),
            }
            if hasattr(self, 'list_indices') and self.list_indices is not None:
                tensors["list_indices"] = self.list_indices.detach().cpu()

        return {
            "name": self.name,
            "config": {
                "dim": self.dim,
                "num_experts": self.num_experts,
                "n_frq": self.n_frq,
                "topk": self.topk,
                "n_tasks": self.n_tasks,
                "use_expert_weights": self.use_expert_weights,
            },
            "metrics": metrics,
            "tensors": tensors,
        }

    # ---- save MoE and experts to disk ----
    def save_state(self, save_dir, task_id=None, include_readout=True, **kwargs):
        import json
        import os
        os.makedirs(save_dir, exist_ok=True)
        exported = self.export_state()

        pt_path = os.path.join(save_dir, f"{self.name}_moe.pt")
        torch.save({
            "name": exported["name"],
            "config": exported["config"],
            "tensors": exported["tensors"],
        }, pt_path)

        json_path = os.path.join(save_dir, f"{self.name}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": exported["name"],
                "config": exported["config"],
                "metrics": exported["metrics"],
            }, f, indent=2)

        return pt_path, json_path

    # ---- load MoE state from checkpoint ----
    def load_state(self, state_dict_or_path, device=None):
        if isinstance(state_dict_or_path, str):
            state = torch.load(state_dict_or_path, map_location=device or self.device)
        else:
            state = state_dict_or_path

        tensors = state.get("tensors", state)
        if "router_weight" in tensors:
            self.router.weight.data.copy_(tensors["router_weight"].to(self.router.weight.device))
        if "router_bias" in tensors and tensors["router_bias"] is not None and self.router.bias is not None:
            self.router.bias.data.copy_(tensors["router_bias"].to(self.router.bias.device))
        if "coeff" in tensors:
            self.coeff.data.copy_(tensors["coeff"].to(self.coeff.device))
        if "expert_mask" in tensors:
            self.expert_mask.copy_(tensors["expert_mask"].to(self.expert_mask.device))


class InverseMoE(nn.Module):
    def __init__(self, dim, num_experts=20, n_frq=3000, topk=5, device="cuda", **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.n_frq = n_frq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.topk = topk
        seed = kwargs.get('seed', get_bilora_seed())
        self.full_permutation = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(int(seed)))
        self.create_bilora_indices()
        self.router = nn.Linear(dim, num_experts)
        
    def create_bilora_indices(self):
        list_indices = [torch.tensor(self.full_permutation[t * self.n_frq : (t + 1) * self.n_frq], device=self.device) for t in range(self.num_experts)]
        self.list_indices = torch.stack(list_indices, dim=0)
    
    def forward(self, cls_token: torch.Tensor, **kwargs):
        B = cls_token.size(0)
        N = self.dim
        router_logits = self.router(cls_token)
        _, batch_expert_indices = torch.topk(F.softmax(router_logits, dim=-1), self.topk, dim=-1)
        all_selected_indices = self.list_indices[batch_expert_indices]
        indices_flat = all_selected_indices.view(B, -1).long()
        binary_matrix_flat = torch.zeros(B, N * N, device=self.device)
        binary_matrix_flat.scatter_(dim=1, index=indices_flat, value=1.0)
        return binary_matrix_flat.view(B, N, N)


class FocalExpert(nn.Module):
    """Task-partitioned frequency expert without routing.
    
    At init, the full frequency space (n_frq_total positions) is partitioned into n_tasks
    non-overlapping subsets. Task t trains only coeff[t*n_frq_per_task : (t+1)*n_frq_per_task].
    At end_task, the task sparse matrix S_t is built from its subset and folded into UMS.
    """
    def __init__(self, dim, n_frq_total, n_tasks, device=None, ums_cfg=None, name="FocalExpert", **kwargs):
        super().__init__()
        self.dim = dim
        self.name = name
        self.n_tasks = n_tasks
        self.n_frq_total = n_frq_total
        self.n_frq_per_task = n_frq_total // n_tasks
        ums_cfg = dict(ums_cfg or {})
        self.ums_in_loop = bool(kwargs.get('ums_in_loop', ums_cfg.get('ums_in_loop', False)))
        self._cached_G = None
        self._cached_P_cand = None
        self._cached_Q_cand = None
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Partition frequency indices: (n_tasks, n_frq_per_task), flat indices into dim*dim
        expert_seed = kwargs.get('seed', ums_cfg.get('seed', get_bilora_seed()))
        if expert_seed is not None:
            full_perm = get_permutation(dim=dim, seed=expert_seed)[:n_frq_total]
        else:
            full_perm = get_permutation(dim=dim)[:n_frq_total]
        list_indices = [
            torch.tensor(full_perm[t * self.n_frq_per_task : (t + 1) * self.n_frq_per_task], dtype=torch.long)
            for t in range(n_tasks)
        ]
        self.register_buffer('list_indices', torch.stack(list_indices, dim=0))  # (n_tasks, n_frq_per_task)
        
        # All coefficients in one parameter vector; task t uses slice [t*n_frq_per_task : (t+1)*n_frq_per_task]
        self.coeff = nn.Parameter(torch.randn(n_frq_total, device=self.device), requires_grad=True)
        
        # UMS Consolidator
        ums_cfg.setdefault('name', self.name)
        ums_cfg.setdefault('n_frq', self.n_frq_per_task)
        self.ums = UMSConsolidator(dim=dim, **ums_cfg).to(self.device)
        
        self.force_task = None
        self.task_assignment_history = {}
    
    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, 'coeff'):
            self.device = self.coeff.device
        if hasattr(self, '_cached_G') and self._cached_G is not None:
            self._cached_G = fn(self._cached_G)
        return self
    
    def build_task_sparse_matrix(self, task_id):
        """Build sparse frequency matrix S_t from the fixed subset of task_id."""
        S = torch.zeros(self.dim, self.dim, device=self.coeff.device, dtype=self.coeff.dtype)
        flat_idx = self.list_indices[task_id].to(self.coeff.device)
        vals = self.coeff[task_id * self.n_frq_per_task : (task_id + 1) * self.n_frq_per_task]
        S.view(-1).index_add_(0, flat_idx, vals)
        return S
    
    def set_force_task(self, task):
        self.force_task = None if task is None else int(task)
    
    def clear_force_task(self):
        self.force_task = None

    @torch.no_grad()
    def refresh_basis(self, task_id=None):
        """Refreshes the candidate basis and caches projection matrix G for UMS-in-the-loop."""
        if not self.ums_in_loop:
            return
        if task_id is None:
            task_id = min(len(self.ums.z_store), self.n_tasks - 1)

        # 1. Build current task sparse matrix
        R = self.ums.rotation().detach()
        S_t = self.build_task_sparse_matrix(task_id).to(device=R.device, dtype=R.dtype)
        if S_t.norm() < 1e-8:
            return

        # 2. Canonical frame under current rotation
        T = R.t() @ S_t @ R

        # 3. Get candidate basis from UMS
        P_cand, Q_cand = self.ums.get_candidate_basis(T)
        if P_cand.shape[1] == 0:
            return

        # 4. Form rotated basis: A = R @ P_cand, B = R @ Q_cand
        A = R @ P_cand
        B = R @ Q_cand

        # 5. Extract support indices
        flat_idx = self.list_indices[task_id].to(R.device)
        rows = flat_idx // self.dim
        cols = flat_idx % self.dim

        # 6. G = A[rows, :] * B[cols, :] (n_frq_per_task, K)
        G = A[rows, :] * B[cols, :]
        self._cached_G = G.detach()
        self._cached_P_cand = P_cand.detach()
        self._cached_Q_cand = Q_cand.detach()

        # Log diagnostic reconstruction error on support
        vals = self.coeff[task_id * self.n_frq_per_task : (task_id + 1) * self.n_frq_per_task].to(device=R.device, dtype=R.dtype)
        s = vals
        z_proj = G.t() @ s
        v_hat = G @ z_proj
        rel_err = ((s - v_hat).norm() / (s.norm() + 1e-12)).item()
        print(f"[{self.name}] Task {task_id} Basis Refreshed: candidate rank={P_cand.shape[1]}/{self.ums.R}, in-loop support recon err: {rel_err:.2%}")

    def end_task(self, task_id):
        """Build S_t from task's fixed subset and fold into UMS. No routing needed."""
        fold_id = int(task_id)
        S_t = self.build_task_sparse_matrix(fold_id)
        s_norm = S_t.norm().item()
        nnz = int((S_t != 0).sum().item())
        sparsity = (1.0 - nnz / S_t.numel()) * 100
        fold_metrics = self.ums.fold(fold_id, S_t)
        self.task_assignment_history[fold_id] = {
            'task_id': fold_id,
            'n_frq_per_task': self.n_frq_per_task,
            'S_norm': float(s_norm),
            'S_sparsity': float(sparsity),
            'S_nnz': int(nnz),
            'ums_metrics': fold_metrics,
        }
        self._cached_G = None
        self._cached_P_cand = None
        self._cached_Q_cand = None
    
    def forward(self, cls_token, **kwargs):
        """Return per-sample weight matrices (B, dim, dim).
        
        In eval mode (force_task set): use UMS readout for the forced task.
        In training mode: build modulation from current task's coefficients via iFFT.
        """
        if self.force_task is not None:
            w = self.ums.readout(self.force_task)
            return w.unsqueeze(0).expand(cls_token.shape[0], -1, -1)
        
        # Training: current task = number of tasks already folded
        cur_task = min(len(self.ums.z_store), self.n_tasks - 1)
        flat_idx = self.list_indices[cur_task].to(self.coeff.device)
        vals = self.coeff[cur_task * self.n_frq_per_task : (cur_task + 1) * self.n_frq_per_task]

        if self.ums_in_loop and hasattr(self, '_cached_G') and self._cached_G is not None:
            G = self._cached_G.to(device=self.coeff.device, dtype=self.coeff.dtype)
            s = vals

            # Linear projection through UMS subspace on support: v_hat = G @ (G^T @ s)
            z_proj = torch.matmul(G.t(), s)
            v_hat = torch.matmul(G, z_proj)

            F_mat = torch.zeros(self.dim, self.dim, device=self.coeff.device, dtype=self.coeff.dtype)
            F_mat.view(-1).index_add_(0, flat_idx, v_hat)
            w = torch.fft.ifft2(F_mat, dim=(-2, -1)).real * self.ums.alpha
            return w.unsqueeze(0).expand(cls_token.shape[0], -1, -1)

        F_mat = torch.zeros(self.dim, self.dim, device=self.coeff.device, dtype=self.coeff.dtype)
        F_mat.view(-1).index_add_(0, flat_idx, vals)
        w = torch.fft.ifft2(F_mat, dim=(-2, -1)).real * 300
        return w.unsqueeze(0).expand(cls_token.shape[0], -1, -1)
    
    def get_analysis_metrics(self, task_id=None):
        return {
            'name': self.name,
            'dim': self.dim,
            'n_tasks': self.n_tasks,
            'n_frq_total': self.n_frq_total,
            'n_frq_per_task': self.n_frq_per_task,
            'coeff_norm': float(self.coeff.norm().item()),
            'tasks_folded': len(self.ums.z_store),
            'task_history': copy.deepcopy(self.task_assignment_history),
        }

    def save_state(self, save_dir, task_id=None, include_readout=True):
        import os
        os.makedirs(save_dir, exist_ok=True)
        pt_path, json_path = self.ums.save_state(save_dir, task_id=task_id, include_readout=include_readout)
        coeff_path = os.path.join(save_dir, f"{self.name}_coeff.pt")
        torch.save({
            'coeff': self.coeff.detach().cpu(),
            'list_indices': self.list_indices.detach().cpu(),
        }, coeff_path)
        return pt_path, json_path



class BiLORA_Focal(Attention_LoRA):
    """BiLORA with fixed task-partitioned frequency subsets (no Router, no MoE).
    
    At init, the full frequency space is partitioned into n_tasks non-overlapping subsets.
    Each task trains only its own subset of coefficients.
    At end_task, the task sparse matrix S_t is built and folded into UMS.
    Supports modulation on Q, K, V, and Output Projection (proj) via FocalExpert.
    """
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 r=64, n_tasks=10, n_frq=576, ums_cfg=None, layer_name="",
                 modulate_q=True, modulate_proj=True, seed=None, **kwargs):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)
        self.n_tasks = n_tasks
        self.n_frq = n_frq  # n_frq_per_task
        self.n_frq_total = n_frq * n_tasks
        self.layer_name = layer_name
        self.modulate_q = bool(modulate_q)
        self.modulate_proj = bool(modulate_proj)
        ums_cfg = dict(ums_cfg or {})
        
        self.seed = seed if seed is not None else (ums_cfg.get('seed') if ums_cfg else None)
        if self.seed is None:
            self.seed = kwargs.get('seed', get_bilora_seed())
        
        name_q = f"{layer_name}-Query" if layer_name else "Focal-Query"
        name_k = f"{layer_name}-Key" if layer_name else "Focal-Key"
        name_v = f"{layer_name}-Val" if layer_name else "Focal-Val"
        name_proj = f"{layer_name}-Proj" if layer_name else "Focal-Proj"
        
        if self.modulate_q:
            ums_cfg_q = dict(ums_cfg)
            ums_cfg_q.setdefault('name', name_q)
            if self.seed is not None:
                ums_cfg_q['seed'] = self.seed
            self.focal_q = FocalExpert(
                dim=dim, n_frq_total=self.n_frq_total, n_tasks=n_tasks,
                device=self.qkv.weight.device, ums_cfg=ums_cfg_q, name=name_q,
                seed=self.seed
            )
        else:
            self.focal_q = None

        ums_cfg_k = dict(ums_cfg)
        ums_cfg_k.setdefault('name', name_k)
        if self.seed is not None:
            ums_cfg_k['seed'] = self.seed
        ums_cfg_v = dict(ums_cfg)
        ums_cfg_v.setdefault('name', name_v)
        if self.seed is not None:
            ums_cfg_v['seed'] = self.seed
        
        self.focal_k = FocalExpert(
            dim=dim, n_frq_total=self.n_frq_total, n_tasks=n_tasks,
            device=self.qkv.weight.device, ums_cfg=ums_cfg_k, name=name_k,
            seed=self.seed
        )
        self.focal_v = FocalExpert(
            dim=dim, n_frq_total=self.n_frq_total, n_tasks=n_tasks,
            device=self.qkv.weight.device, ums_cfg=ums_cfg_v, name=name_v,
            seed=self.seed
        )

        if self.modulate_proj:
            ums_cfg_proj = dict(ums_cfg)
            ums_cfg_proj.setdefault('name', name_proj)
            if self.seed is not None:
                ums_cfg_proj['seed'] = self.seed
            self.focal_proj = FocalExpert(
                dim=dim, n_frq_total=self.n_frq_total, n_tasks=n_tasks,
                device=self.qkv.weight.device, ums_cfg=ums_cfg_proj, name=name_proj,
                seed=self.seed
            )
        else:
            self.focal_proj = None
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.focal_q is not None:
            self.focal_q.to(*args, **kwargs)
        self.focal_k.to(*args, **kwargs)
        self.focal_v.to(*args, **kwargs)
        if self.focal_proj is not None:
            self.focal_proj.to(*args, **kwargs)
        return self

    def copy_weights_from(self, orig_attn):
        """Copy pretrained qkv and proj weights and biases from the original ViT attention module."""
        if hasattr(orig_attn, 'qkv') and orig_attn.qkv is not None and hasattr(self, 'qkv') and self.qkv is not None:
            if hasattr(orig_attn.qkv, 'weight') and orig_attn.qkv.weight is not None:
                if orig_attn.qkv.weight.shape == self.qkv.weight.shape:
                    self.qkv.weight.data.copy_(orig_attn.qkv.weight.data)
            if getattr(orig_attn.qkv, 'bias', None) is not None and getattr(self.qkv, 'bias', None) is not None:
                if orig_attn.qkv.bias.shape == self.qkv.bias.shape:
                    self.qkv.bias.data.copy_(orig_attn.qkv.bias.data)
        if hasattr(orig_attn, 'proj') and orig_attn.proj is not None and hasattr(self, 'proj') and self.proj is not None:
            if hasattr(orig_attn.proj, 'weight') and orig_attn.proj.weight is not None:
                if orig_attn.proj.weight.shape == self.proj.weight.shape:
                    self.proj.weight.data.copy_(orig_attn.proj.weight.data)
            if getattr(orig_attn.proj, 'bias', None) is not None and getattr(self.proj, 'bias', None) is not None:
                if orig_attn.proj.bias.shape == self.proj.bias.shape:
                    self.proj.bias.data.copy_(orig_attn.proj.bias.data)
        return self
    
    def end_task(self, task_id=None):
        if self.focal_q is not None:
            self.focal_q.end_task(task_id)
        self.focal_k.end_task(task_id)
        self.focal_v.end_task(task_id)
        if self.focal_proj is not None:
            self.focal_proj.end_task(task_id)
    
    def set_force_task(self, task):
        if self.focal_q is not None:
            self.focal_q.set_force_task(task)
        self.focal_k.set_force_task(task)
        self.focal_v.set_force_task(task)
        if self.focal_proj is not None:
            self.focal_proj.set_force_task(task)
    
    def clear_force_task(self):
        if self.focal_q is not None:
            self.focal_q.clear_force_task()
        self.focal_k.clear_force_task()
        self.focal_v.clear_force_task()
        if self.focal_proj is not None:
            self.focal_proj.clear_force_task()

    def refresh_basis(self, task_id=None):
        for expert in self.get_focal_experts():
            expert.refresh_basis(task_id)

    def get_focal_experts(self):
        """Return list of active FocalExpert modules in this layer."""
        experts = []
        if self.focal_q is not None:
            experts.append(self.focal_q)
        if self.focal_k is not None:
            experts.append(self.focal_k)
        if self.focal_v is not None:
            experts.append(self.focal_v)
        if self.focal_proj is not None:
            experts.append(self.focal_proj)
        return experts

    def get_analysis_metrics(self, task_id=None):
        metrics = {
            'layer_name': getattr(self, 'layer_name', 'BiLORA_Focal'),
            'focal_k': self.focal_k.get_analysis_metrics(task_id=task_id),
            'focal_v': self.focal_v.get_analysis_metrics(task_id=task_id),
            'ums_k': self.focal_k.ums.get_analysis_metrics(task_id=task_id),
            'ums_v': self.focal_v.ums.get_analysis_metrics(task_id=task_id),
        }
        if self.focal_q is not None:
            metrics['focal_q'] = self.focal_q.get_analysis_metrics(task_id=task_id)
            metrics['ums_q'] = self.focal_q.ums.get_analysis_metrics(task_id=task_id)
        if self.focal_proj is not None:
            metrics['focal_proj'] = self.focal_proj.get_analysis_metrics(task_id=task_id)
            metrics['ums_proj'] = self.focal_proj.ums.get_analysis_metrics(task_id=task_id)
        return metrics
    
    def save_state(self, ums_dir, task_id=None, include_readout=True):
        if self.focal_q is not None:
            self.focal_q.save_state(ums_dir, task_id=task_id, include_readout=include_readout)
        self.focal_k.save_state(ums_dir, task_id=task_id, include_readout=include_readout)
        self.focal_v.save_state(ums_dir, task_id=task_id, include_readout=include_readout)
        if self.focal_proj is not None:
            self.focal_proj.save_state(ums_dir, task_id=task_id, include_readout=include_readout)
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # NOTE: the modulation is applied to ALL tokens (CLS included). Concatenating the
        # raw CLS input here would add an identity residual to the CLS position only.
        if self.focal_q is not None:
            weight_q = self.focal_q(x[:, 0, :])
            x_new_q = torch.matmul(x, weight_q)
            q = q + x_new_q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        weight_k = self.focal_k(x[:, 0, :])
        weight_v = self.focal_v(x[:, 0, :])
        
        x_new_k = torch.matmul(x, weight_k)
        x_new_v = torch.matmul(x, weight_v)
        k = k + x_new_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + x_new_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.focal_proj is not None:
            weight_proj = self.focal_proj(x[:, 0, :])
            x_new_proj = torch.matmul(attn_out, weight_proj)
            x = self.proj(attn_out) + x_new_proj
        else:
            x = self.proj(attn_out)

        x = self.proj_drop(x)
        return x


class BiLORA_MoE(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=9000, num_experts=32, topk=3, use_expert_weights=False, ums_cfg=None, layer_name="", **kwargs):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)
        self.num_experts= num_experts
        self.n_tasks = n_tasks
        assert num_experts <= (dim * dim) / n_frq, "Number of experts is too large"
        self.topk = topk
        self.n_frq = n_frq
        self.use_expert_weights = use_expert_weights
        self.layer_name = layer_name
        ums_cfg = dict(ums_cfg or {})
        name_k = f"{layer_name}-Key" if layer_name else "MoE-Key"
        name_v = f"{layer_name}-Val" if layer_name else "MoE-Val"
        self.moe_k = MoE(dim=dim, num_experts=num_experts, n_frq=n_frq, n_tasks=n_tasks, topk=topk, device=self.qkv.weight.device, use_expert_weights=self.use_expert_weights, ums_cfg=ums_cfg, name=name_k)
        self.moe_v = MoE(dim=dim, num_experts=num_experts, n_frq=n_frq, n_tasks=n_tasks, topk=topk, device=self.qkv.weight.device, use_expert_weights=self.use_expert_weights, ums_cfg=ums_cfg, name=name_v)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.moe_k.to(*args, **kwargs)
        self.moe_v.to(*args, **kwargs)
        return self
    
    def set_current_epoch(self, epoch):
        self.moe_k.set_current_epoch(epoch)
        self.moe_v.set_current_epoch(epoch)
    
    def turn_on_saving_experts(self):
        self.moe_k.flag_to_save_lora_indices = True 
        self.moe_v.flag_to_save_lora_indices = True
        
    def turn_off_saving_experts(self, task=None):
        self.moe_k.end_task(task)
        self.moe_v.end_task(task)
        self.moe_k.flag_to_save_lora_indices = False
        self.moe_v.flag_to_save_lora_indices = False

    def set_force_task(self, task):
        self.moe_k.set_force_task(task)
        self.moe_v.set_force_task(task)

    def clear_force_task(self):
        self.moe_k.clear_force_task()
        self.moe_v.clear_force_task()

    # ---- BiLORA_MoE analysis & saving ----
    def get_analysis_metrics(self, task_id=None):
        return {
            "layer_name": getattr(self, 'layer_name', 'BiLORA_MoE'),
            "moe_k": self.moe_k.get_analysis_metrics(task_id=task_id),
            "moe_v": self.moe_v.get_analysis_metrics(task_id=task_id),
            "ums_k": self.moe_k.ums.get_analysis_metrics(task_id=task_id),
            "ums_v": self.moe_v.ums.get_analysis_metrics(task_id=task_id),
        }

    def save_state(self, ums_dir, moe_dir, task_id=None):
        self.moe_k.ums.save_state(ums_dir, task_id=task_id)
        self.moe_v.ums.save_state(ums_dir, task_id=task_id)
        self.moe_k.save_state(moe_dir, task_id=task_id)
        self.moe_v.save_state(moe_dir, task_id=task_id)
    
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        weight_k = self.moe_k(x[:, 0, :])
        weight_v = self.moe_v(x[:, 0, :])
        cls_token = x[:, :1, :]      # Kích thước (B, 1, 768)
        patch_tokens = x[:, 1:, :]   # Kích thước (B, 196, 768)
        transformed_patches_k = torch.matmul(patch_tokens, weight_k)
        transformed_patches_v = torch.matmul(patch_tokens, weight_v)
        x_new_k = torch.cat([cls_token, transformed_patches_k], dim=1)
        x_new_v = torch.cat([cls_token, transformed_patches_v], dim=1)
        k = k + x_new_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + x_new_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BiLoRA_InverseMoE(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, r=64, n_tasks=10, num_experts=20):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)
        
        self.moe_k = InverseMoE(dim=dim, num_experts=num_experts, n_frq=r, topk=5, device=self.qkv.weight.device)
        self.moe_v = InverseMoE(dim=dim, num_experts=num_experts, n_frq=r, topk=5, device=self.qkv.weight.device)
    
    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        weight_k = self.moe_k(x[:, 0, :])
        weight_v = self.moe_v(x[:, 0, :])
        k = k + torch.bmm(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + torch.bmm(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
