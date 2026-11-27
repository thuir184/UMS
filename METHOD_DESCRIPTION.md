# FOCAL-UMS: Frequency-Partitioned Orthogonal Consolidation and Unified Modulation Subspace for Exemplar-Free Continual Learning

---

## TABLE OF CONTENTS / MỤC LỤC
1. [English Version: Proposed Method Section](#english-version-proposed-method)
   - 1. Problem Formulation and Preliminaries
   - 2. Overview of the Proposed Architecture
   - 3. Disjoint Frequency-Domain Parameterization (FocalExpert)
   - 4. Closed-Loop Training: UMS-in-the-Loop Projection
   - 5. Post-Task Consolidation: Unified Modulation Subspace (UMS)
     - 5.1. Residual SVD and Incremental Subspace Expansion
     - 5.2. Cayley Optimization on the Stiefel Manifold with Multi-Objective Loss
     - 5.3. Bilinear Support Head for High-Frequency Noise Gating
   - 6. Evaluation and Inference: Task-Agnostic Mahalanobis Routing
   - 7. Theoretical Analysis and Computational Complexity
2. [Bản Tiếng Việt: Mô Tả Chi Tiết Phương Pháp Đề Xuất](#ban-tieng-viet-mo-ta-chi-tiet-phuong-phap)
   - 1. Phát biểu Bài toán và Kiến trúc Cơ sở
   - 2. Tổng quan về Kiến trúc Đề xuất
   - 3. Phân hoạch Tham số trên Miền Tần số Rời rạc (FocalExpert)
   - 4. Quá trình Huấn luyện Khép kín: UMS-in-the-Loop
   - 5. Hợp nhất Tri thức sau mỗi Task: Không gian Điều chế Hợp nhất (UMS)
     - 5.1. SVD phần dư và Mở rộng Cơ sở Trực chuẩn Trực tuyến
     - 5.2. Tối ưu hóa Phép quay Cayley trên Đa tạp Stiefel
     - 5.3. Mạng Phân loại Hỗ trợ Song tuyến tính (Bilinear Support Head)
   - 6. Quá trình Đánh giá và Suy luận: Phân định Miền bằng Khoảng cách Mahalanobis
   - 7. Phân tích Lý thuyết và Độ phức tạp Tính toán

---

# English Version: Proposed Method

## 1. Problem Formulation and Preliminaries

### 1.1 Continual Learning Paradigm
We consider an exemplar-free continual learning setup defined over a sequence of $T$ distinct tasks (domains) $\mathcal{D} = \{\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_T\}$. Each task $\mathcal{T}_t$ is associated with a dataset $\mathcal{D}_t = \{(x_{t, i}, y_{t, i})\}_{i=1}^{N_t}$, where $x_{t, i} \in \mathcal{X}$ is an input image and $y_{t, i} \in \mathcal{Y}_t$ denotes its ground-truth label. 
In our exemplar-free setting, when training on task $\mathcal{T}_t$, data from previous tasks $\mathcal{D}_{<t}$ cannot be retained, stored in memory buffers, or replayed. At inference time, given an arbitrary test sample $x$, the model must correctly identify its originating task domain $t^*$ and produce the corresponding class prediction $\hat{y} \in \mathcal{Y}_{t^*}$ without being provided task identity descriptors at test time (Task-Agnostic / Domain-IL evaluation).

### 1.2 Vision Transformer (ViT) Formulation
The backbone is a Vision Transformer composed of $L = 12$ standard Transformer encoder blocks. Let $D = 768$ denote the embedding dimension. An input image is converted into a sequence of patch embeddings prepended with a class token, yielding a latent sequence $X \in \mathbb{R}^{B \times N \times D}$, where $B$ is the batch size and $N = 197$ is the sequence length.
Inside each multi-head self-attention (MHSA) layer with $H = 12$ heads and head dimension $d = D / H = 64$, the linear projections for Query ($Q$), Key ($K$), Value ($V$), and Output Projection ($\text{Proj}$) are parameterized by weight matrices:
$$W_Q, W_K, W_V, W_{proj} \in \mathbb{R}^{D \times D}$$
Throughout continual training, the pretrained weights $W_Q, W_K, W_V, W_{proj}$ remain strictly **frozen** across all $L$ blocks to preserve generalized visual representations. Plastic adaptation is achieved strictly through parameter-efficient task modulations.

---

## 2. Overview of the Proposed Architecture

To overcome catastrophic forgetting while strictly maintaining high task plasticity without replay buffers, we propose **FOCAL-UMS** (*Frequency-Partitioned Orthogonal Consolidation with Unified Modulation Subspace*). 
Our framework comprises three key components:
1. **Targeted Multi-Layer Modulation**: Modulation is applied exclusively to the upper transformer blocks $\mathcal{L}_{train} = \{6, 7, 8, 9, 10, 11\}$ ($6$ out of $12$ blocks), keeping lower blocks $\{0, 1, 2, 3, 4, 5\}$ entirely frozen to act as domain-invariant feature extractors. Within each targeted block, modulation operates on all four critical linear projections: Query ($Q$), Key ($K$), Value ($V$), and Output Projection ($Proj$).
2. **Disjoint Frequency-Domain Allocation (FocalExpert)**: Instead of allocating unconstrained dense or low-rank adapters in the spatial domain, each task is assigned a mathematically disjoint coordinate subset in the 2D Discrete Fourier Transform (DFT) frequency domain. Mutual interference between tasks during active learning is eliminated by design through orthogonal frequency partitioning.
3. **Unified Modulation Subspace (UMS) with Closed-Loop Projection**: During training, a candidate projection operator aligns the gradient trajectory with a shared subspace (**UMS-in-the-loop**). At the boundary of each task, the sparse frequency matrix is factorized via truncated SVD and integrated into a compact, shared orthonormal basis $(P, Q)$ governed by a smooth Cayley-parameterized rotation on the Stiefel manifold. Raw per-task parameters are completely discarded, leaving only a low-dimensional coordinate vector $z_t \in \mathbb{R}^R$ and a gating support head per task.

---

## 3. Disjoint Frequency-Domain Parameterization (FocalExpert)

### 3.1 Spatial vs. Frequency Modulation
For any linear layer with input $H \in \mathbb{R}^{B \times N \times D}$ and frozen weight $W \in \mathbb{R}^{D \times D}$, our adaptation introduces a task-specific additive matrix $\Delta W_t \in \mathbb{R}^{D \times D}$:
$$H' = H (W + \Delta W_t) = H W + H \Delta W_t$$
For Output Projection, the modulated output is similarly:
$$H_{out} = H_{attn} W_{proj} + H_{attn} \Delta W_{proj, t}$$
The modulation weight $\Delta W_t$ is constructed as the real part of the 2D Inverse Discrete Fourier Transform ($\text{iFFT2}$) of a sparse 2D frequency matrix $S_t \in \mathbb{R}^{D \times D}$:
$$\Delta W_t = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(S_t) \right)$$
where $\alpha = 300.0$ is a fixed scaling factor, and $\mathcal{F}^{-1}(\cdot)$ denotes the 2D inverse Fourier transform:
$$\left[\mathcal{F}^{-1}(S_t)\right]_{j, k} = \frac{1}{D} \sum_{u=0}^{D-1} \sum_{v=0}^{D-1} S_t(u, v) \exp\left( i 2\pi \left(\frac{ju + kv}{D}\right) \right)$$

### 3.2 Orthogonal Frequency Partitioning
The total frequency space contains $D^2 = 768 \times 768 = 589,824$ potential frequency coordinates. We establish a deterministic pseudo-random permutation of all $D^2$ coordinate indices: $\pi: \{0, \dots, D^2-1\} \to \{0, \dots, D-1\} \times \{0, \dots, D-1\}$.
For a continuum of $T = 10$ tasks, we partition the spectrum into $T$ pairwise disjoint subsets $\{\Omega_1, \Omega_2, \dots, \Omega_T\}$:
$$\Omega_t \cap \Omega_{t'} = \emptyset \quad \forall t \neq t', \quad |\Omega_t| = N_{frq} = 5,898$$
The active capacity per task is $N_{frq} = 5,898$ coefficients ($\approx 1.0\%$ of the total frequency space), ensuring high sparsity ($99.0\%$ sparse). Across all $T=10$ tasks, the total utilized frequency capacity is $\sum_{t=1}^{10} |\Omega_t| = 58,980 \approx 10\%$ of the full frequency matrix.
Each task $\mathcal{T}_t$ optimizes strictly its dedicated vector of coefficients $\theta_t \in \mathbb{R}^{N_{frq}}$. The matrix $S_t$ is constructed as:
$$S_t(u, v) = \begin{cases} \theta_{t, m} & \text{if } (u, v) = \pi_{t}(m) \in \Omega_t \\ 0 & \text{otherwise} \end{cases}$$
Because $\Omega_t \cap \Omega_{t'} = \emptyset$, task parameters cannot interfere with or overwrite one another during gradient updates:
$$\frac{\partial S_t(u, v)}{\partial \theta_{t', m}} = 0 \quad \forall t \neq t'$$

---

## 4. Closed-Loop Training: UMS-in-the-Loop Projection

Standard continual adaptation methods suffer from a fundamental discrepancy: a model trains unconstrained task weights during epoch optimization, but subsequently subjects them to post-hoc compression (e.g., SVD truncation or projection), inducing catastrophic reconstruction distortion. 
To eliminate this optimization gap, we introduce **UMS-in-the-Loop**. The optimization trajectory is directly constrained to lie within the candidate consolidated subspace throughout training.

### 4.1 Candidate Subspace Construction
At task initialization and every $E_{refresh} = 2$ epochs during task $\mathcal{T}_t$, the current state of $S_t$ is evaluated against the shared orthonormal bases $P \in \mathbb{R}^{D \times R}$ and $Q \in \mathbb{R}^{D \times R}$ ($R = 768$, current active rank $r_{cur} \le R$) and current rotation matrix $R_o \in \mathcal{SO}(D)$.
1. Compute the canonical representation:
   $$T_t = R_o^T S_t R_o$$
2. Project out existing basis directions:
   $$U_P = (I - P_{:, :r_{cur}} P_{:, :r_{cur}}^T) T_t, \quad V_Q = T_t (I - Q_{:, :r_{cur}} Q_{:, :r_{cur}}^T)$$
3. Compute truncated SVD of the residual:
   $$U_P \approx U_r \Sigma_r V_r^T, \quad U_r \in \mathbb{R}^{D \times r}, \quad V_r \in \mathbb{R}^{D \times r} \quad (r = 72)$$
4. Orthonormalize the residual vectors via QR factorization and concatenate with existing bases to form candidate orthonormal matrices $P_{cand}, Q_{cand} \in \mathbb{R}^{D \times K}$ ($K = r_{cur} + r$):
   $$P_{cand} = [P_{:, :r_{cur}}, \text{qr}(U_r)], \quad Q_{cand} = [Q_{:, :r_{cur}}, \text{qr}(V_r)]$$
5. Map candidate bases back into the rotated reference frame:
   $$A = R_o P_{cand} \in \mathbb{R}^{D \times K}, \quad B = R_o Q_{cand} \in \mathbb{R}^{D \times K}$$

### 4.2 In-Loop Gradient Projection
For any frequency position $(u, v) = \pi_t(m) \in \Omega_t$, its reconstructed value from a diagonal coordinate representation $z \in \mathbb{R}^K$ is:
$$\left[ A \text{diag}(z) B^T \right]_{u, v} = \sum_{k=1}^K A_{u, k} B_{v, k} z_k$$
We define the support projection operator matrix $G \in \mathbb{R}^{N_{frq} \times K}$ whose $m$-th row is given by the Hadamard product:
$$G_{m, :} = A_{u(m), :} \odot B_{v(m), :}, \quad \text{where } (u(m), v(m)) = \pi_t(m)$$
During forward propagation, the raw parameter vector $\theta_t \in \mathbb{R}^{N_{frq}}$ is projected through $G$:
$$z_{proj} = G^T \theta_t \in \mathbb{R}^K$$
$$\hat{v}_t = G z_{proj} = G G^T \theta_t \in \mathbb{R}^{N_{frq}}$$
The matrix $\hat{S}_t$ is populated exclusively using the projected values $\hat{v}_t$:
$$\hat{S}_t(u, v) = \begin{cases} \hat{v}_{t, m} & \text{if } (u, v) = \pi_t(m) \in \Omega_t \\ 0 & \text{otherwise} \end{cases}$$
The modulation weight matrix during training is therefore:
$$\Delta W_t = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(\hat{S}_t) \right)$$
Under this formulation, the task loss $\mathcal{L}_{task} = \mathcal{L}_{CE}(f(X; \Delta W_t), Y)$ backpropagates through the projection operator:
$$\frac{\partial \mathcal{L}_{task}}{\partial \theta_t} = G G^T \frac{\partial \mathcal{L}_{task}}{\partial \hat{v}_t}$$
This guarantees that $\theta_t$ is trained strictly within the span of the candidate consolidated subspace, resulting in zero post-hoc truncation shock.

---

## 5. Post-Task Consolidation: Unified Modulation Subspace (UMS)

At the completion of task $\mathcal{T}_t$, the task matrix $S_t$ must be permanently consolidated into the shared memory structure, after which $S_t$ and $\theta_t$ are safely erased from memory.

### 5.1 Residual SVD and Incremental Subspace Expansion
With the rotation matrix $R_o$, the canonical representation is:
$$T_t = R_o^T S_t R_o$$
If existing basis rank $r_{cur} > 0$, the projected residual matrices are:
$$U_p = T_t - P_{:, :r_{cur}} (P_{:, :r_{cur}}^T T_t)$$
$$V_q = T_t - (T_t Q_{:, :r_{cur}}) Q_{:, :r_{cur}}^T$$
We compute the singular value decomposition of $U_p$:
$$U_p = U_r \Sigma_r V_r^T, \quad \Sigma_r = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$$
The residual reconstruction energy is quantified by:
$$\mathcal{E}(T_t) = \sum_{j=1}^r \sigma_j^2$$
We compare $\mathcal{E}(T_t)$ against an expansion threshold $\tau = 10^{-3}$:
- **Case 1 ($\mathcal{E}(T_t) > \tau$ and $r_{cur} < R$)**: The existing subspace is insufficient to capture task $\mathcal{T}_t$. The residual singular directions $U_r \Sigma_r$ and $(I - Q_{:, :r_{cur}} Q_{:, :r_{cur}}^T) V_r \Sigma_r$ are orthonormalized via QR decomposition and appended to $P$ and $Q$:
  $$P_{:, r_{cur}:r_{cur}+\Delta r} \leftarrow \text{qr}(U_r \Sigma_r), \quad Q_{:, r_{cur}:r_{cur}+\Delta r} \leftarrow \text{qr}(V_r')$$
  The active rank updates as $r_{cur} \leftarrow r_{cur} + \Delta r$, subject to the maximum capacity $R = 768$.
- **Case 2 ($\mathcal{E}(T_t) \le \tau$)**: The existing subspace already spans task $\mathcal{T}_t$ within tolerance $\tau$. No new columns are added ($\Delta r = 0$), preserving maximal subspace capacity for subsequent tasks.

Once $P$ and $Q$ are updated, the task coordinate vector $z_t \in \mathbb{R}^R$ is computed as the exact diagonal projection:
$$z_t = \text{diag}\left( P_{:, :r_{cur}}^T T_t Q_{:, :r_{cur}} \right) \in \mathbb{R}^{r_{cur}}$$
padded with zeros up to dimension $R$.

### 5.2 Cayley Optimization on the Stiefel Manifold
The rotation matrix $R_o \in \mathbb{R}^{D \times D}$ serves as a global transformation that aligns all tasks into an optimal canonical frame where cross-task interference is minimized.
To guarantee that $R_o$ remains strictly orthogonal without expensive Riemannian retract operations, we employ the **Cayley transform**. 
Let $W_{rot} \in \mathbb{R}^{D \times D}$ be an unconstrained parameter matrix. We construct a skew-symmetric matrix $A$:
$$A = W_{rot} - W_{rot}^T \quad \implies \quad A^T = -A$$
The orthogonal rotation matrix $R_o \in \mathcal{SO}(D)$ is parameterized as:
$$R_o = \left( I - \frac{1}{2} A \right)^{-1} \left( I + \frac{1}{2} A \right)$$
By Cayley's theorem, $R_o^T R_o = I$ and $\det(R_o) = +1$ unconditionally.

During post-task consolidation, $W_{rot}$ is refined for $N_{rot} = 15$ steps using Adam ($\text{lr} = 10^{-4}$) under a composite multi-objective loss:
$$\mathcal{L}_{rot} = \lambda_{mse} \mathcal{L}_{mse} + \lambda_{bce} \mathcal{L}_{bce} + \lambda_{gauss} \mathcal{L}_{gauss} + \lambda_{diag} \mathcal{L}_{diag} + \lambda_{anchor} \mathcal{L}_{anchor}$$
where the hyperparameter weights are set to:
$$\lambda_{mse} = 1.0, \quad \lambda_{bce} = 0.1, \quad \lambda_{gauss} = 0.01, \quad \lambda_{diag} = 0.1, \quad \lambda_{anchor} = 2.0$$

The individual loss components are rigorously formulated as follows:
1. **Target Support Reconstruction Loss ($\mathcal{L}_{mse}$)**:
   Given the canonical core $C_t = P \text{diag}(z_t) Q^T$, the reconstructed frequency matrix is $\hat{S}_{rot} = R_o C_t R_o^T$. The MSE is computed strictly over the non-zero support $\Omega_t$:
   $$\mathcal{L}_{mse} = \frac{1}{|\Omega_t|} \sum_{(u, v) \in \Omega_t} \left( S_t(u, v) - [\hat{S}_{rot}]_{u, v} \right)^2$$
2. **Support Binary Cross-Entropy Loss ($\mathcal{L}_{bce}$)**:
   Supervises the Bilinear Support Head (defined in Section 5.3) to classify non-zero vs. zero coordinates:
   $$\mathcal{L}_{bce} = -\frac{1}{D^2} \sum_{u, v} \left[ w_{pos} \cdot Y_{u, v} \log \sigma(\Lambda_{u, v}) + (1 - Y_{u, v}) \log (1 - \sigma(\Lambda_{u, v})) \right]$$
   where $Y_{u, v} = \mathbb{I}((u, v) \in \Omega_t)$, $\Lambda = W_L \text{diag}(z_t) W_R^T + B$, and $w_{pos} = \frac{D^2 - N_{frq}}{N_{frq}} \approx 99.0$.
3. **Gaussian Prior Regularization ($\mathcal{L}_{gauss}$)**:
   Enforces that the values of the reconstructed frequency coefficients on $\Omega_t$ match a standard normal distribution $\mathcal{N}(0, 1)$, preventing runaway gradient explosion:
   $$\mu_{pred} = \frac{1}{|\Omega_t|} \sum_{(u, v) \in \Omega_t} [\hat{S}_{rot}]_{u, v}, \quad \sigma_{pred} = \sqrt{\frac{1}{|\Omega_t|} \sum_{(u, v) \in \Omega_t} ([\hat{S}_{rot}]_{u, v} - \mu_{pred})^2 + \epsilon}$$
   $$\mathcal{L}_{gauss} = \mu_{pred}^2 + (\sigma_{pred} - 1.0)^2$$
4. **Off-Diagonal Suppression / Joint Diagonalization ($\mathcal{L}_{diag}$)**:
   Forces the transformed representation $Z_t = P_{:, :r_{cur}}^T (R_o^T S_t R_o) Q_{:, :r_{cur}}$ to be purely diagonal, eliminating off-diagonal energy that causes cross-task coordinate crosstalk:
   $$\mathcal{L}_{diag} = \frac{1}{r_{cur}(r_{cur} - 1)} \sum_{i \neq j} [Z_t]_{i, j}^2$$
5. **Subspace Anchor Drift Loss ($\mathcal{L}_{anchor}$)**:
   Prevents the newly updated rotation $R_o$ and support head from corrupting the subspace orientations already established for previous tasks $\mathcal{T}_{<t}$:
   $$\mathcal{L}_{anchor} = \frac{1}{2 r_{old}} \left( \| (R_o - R_{prev}) P_{old} \|_F^2 + \| (R_o - R_{prev}) Q_{old} \|_F^2 \right) + \frac{1}{2 r_{old}} \left( \| W_L^{old} - W_{L, prev}^{old} \|_F^2 + \| W_R^{old} - W_{R, prev}^{old} \|_F^2 \right)$$

### 5.3 Bilinear Support Head for High-Frequency Noise Gating
Because the inverse Fourier transform $\mathcal{F}^{-1}$ integrates over all $D \times D$ coordinates, small reconstruction leakage on the $99\%$ zero-valued frequency positions would accumulate into severe spatial domain noise upon transformation. 
To eliminate this artifact, we introduce the **Bilinear Support Head** $\mathcal{H}_{supp}$, parameterized by matrices $W_L, W_R \in \mathbb{R}^{D \times R}$ and bias $B \in \mathbb{R}^{D \times D}$.
Given the task coordinate diagonal matrix $\text{diag}(z_t)$, the head computes coordinate-wise logits:
$$\Lambda(z_t) = W_L \text{diag}(z_t) W_R^T + B \in \mathbb{R}^{D \times D}$$
The support probability mask is obtained via the elementwise sigmoid function:
$$M_t = \sigma(\Lambda(z_t)) \in [0, 1]^{D \times D}$$
At consolidation and during readout, the reconstructed matrix $\hat{S}_t = R_o (P \text{diag}(z_t) Q^T) R_o^T$ is subjected to hard threshold gating:
$$\hat{S}_{gated, t}(u, v) = \begin{cases} \hat{S}_t(u, v) & \text{if } M_t(u, v) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$
The spatial modulation matrix restored from the consolidated memory is:
$$W_t = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(\hat{S}_{gated, t}) \right)$$

---

## 6. Evaluation and Inference: Task-Agnostic Mahalanobis Routing

At test time, the model is evaluated in a task-agnostic setting: an input image $x$ arrives without task identity. The evaluation process consists of two stages: **Domain Routing** followed by **Targeted Prediction**.

```
Input x ───► Frozen ViT (Feature Extractor) ───► Class Token h
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      ▼                                                                   ▼
       Linear Bottleneck fcArr[0]                                          Linear Bottleneck fcArr[k]
                      │                                                                   │
             Features z^(0)                                                      Features z^(k)
                      │                                                                   │
    Mahalanobis d_M(z^(0), N(mu_0, Sigma_0))                            Mahalanobis d_M(z^(k), N(mu_k, Sigma_k))
                      └─────────────────────────────────┬─────────────────────────────────┘
                                                        ▼
                                           Argmin k* (Predicted Task)
                                                        │
                                                        ▼
                                    Readout UMS Modulation: W_{k*}
                                    Apply to ViT Blocks [6, 7, 8, 9, 10, 11]
                                                        │
                                                        ▼
                                        Forward Pass with Modulation
                                                        │
                                                        ▼
                                            classifierArr[k*](...) ───► Class Prediction c*
```

### 6.1 Task Distribution Profiling (End-of-Task Registration)
At the end of training task $\mathcal{T}_t$, the frozen ViT backbone (acting as a fixed feature extractor) extracts normalized class tokens $h(x) \in \mathbb{R}^D$ across all training samples $x \in \mathcal{D}_t$.
The features are projected through the task-specific linear bottleneck $\text{fcArr}_t: \mathbb{R}^D \to \mathbb{R}^{d_{fc}}$ ($d_{fc} = 500$):
$$z_{t, i} = \text{fcArr}_t(h(x_{t, i})) \in \mathbb{R}^{d_{fc}}$$
We compute and store the empirical mean vector $\mu_t \in \mathbb{R}^{d_{fc}}$ and covariance matrix $\Sigma_t \in \mathbb{R}^{d_{fc} \times d_{fc}}$:
$$\mu_t = \frac{1}{N_t} \sum_{i=1}^{N_t} z_{t, i}$$
$$\Sigma_t = \frac{1}{N_t - 1} \sum_{i=1}^{N_t} (z_{t, i} - \mu_t)(z_{t, i} - \mu_t)^T + \epsilon_I I_{d_{fc}}$$
where $\epsilon_I = 10^{-4}$ is a shrinkage parameter ensuring strict positive definiteness and invertible covariance conditioning.

### 6.2 Inference Task Routing
Given a test query batch $X_{test} \in \mathbb{R}^{B \times C \times H \times W}$, we extract feature representations for each observed candidate task $k \in \{1, \dots, t\}$:
$$Z^{(k)} = \text{fcArr}_k(h(X_{test})) \in \mathbb{R}^{B \times d_{fc}}$$
The Mahalanobis distance of sample $b$ to the distribution of task $k$ is given by:
$$d_M(Z_b^{(k)}, \mathcal{N}(\mu_k, \Sigma_k)) = \sqrt{ (Z_b^{(k)} - \mu_k)^T \Sigma_k^{-1} (Z_b^{(k)} - \mu_k) }$$
The optimal task domain index $k^*$ is identified by minimizing the expected batch distance:
$$k^* = \arg\min_{k \in \{1, \dots, t\}} \frac{1}{B} \sum_{b=1}^B d_M(Z_b^{(k)}, \mathcal{N}(\mu_k, \Sigma_k))$$

### 6.3 Restoring Modulation and Producing Predictions
Once $k^*$ is identified:
1. Each modulated attention module restores its modulation weight $\Delta W_{k^*}$ using the single consolidated operator:
   $$\hat{S}_{k^*} = \left[ R_o (P \text{diag}(z_{k^*}) Q^T) R_o^T \right] \odot \mathbb{I}(\sigma(W_L \text{diag}(z_{k^*}) W_R^T + B) \ge 0.5)$$
   $$\Delta W_{k^*} = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(\hat{S}_{k^*}) \right)$$
2. The input is forwarded through the modulated ViT model to obtain adapted feature representation $f_{k^*}(X_{test}) \in \mathbb{R}^{B \times D}$.
3. The final class prediction is obtained through the $k^*$-th classification head:
   $$\hat{y} = \text{classifierArr}_{k^*}\left( \text{fcArr}_{k^*}(f_{k^*}(X_{test})) \right)$$
   $$\hat{c} = \arg\max_{c \in \mathcal{Y}_{k^*}} \hat{y}_c$$

---

## 7. Theoretical Analysis and Computational Complexity

### 7.1 Memory Footprint and Storage Complexity
Let $D = 768$, $R = 768$, $d_{fc} = 500$, and $T = 10$ tasks. 
- In traditional exemplar-free CL, storing $T$ independent sets of LoRA parameters or full models incurs $\mathcal{O}(T \cdot 2 D r_{lora} \cdot L)$ overhead.
- Under **FOCAL-UMS**:
  1. The shared bases $P, Q \in \mathbb{R}^{D \times R}$ require $2 \cdot D \cdot R = 2 \cdot 768^2 \approx 1.18 \times 10^6$ parameters per modulated matrix.
  2. The Cayley generator $W_{rot} \in \mathbb{R}^{D \times D}$ requires $D^2 \approx 0.59 \times 10^6$ parameters.
  3. The Bilinear Support Head requires $2 \cdot D \cdot R + D^2 \approx 1.77 \times 10^6$ parameters.
  4. Crucially, **for each additional task $t$, the incremental storage cost is strictly $R = 768$ scalar coordinates** ($z_t \in \mathbb{R}^R$), plus the linear classifier weights ($\mathcal{O}(d_{fc} \cdot |\mathcal{Y}_t|)$).
  The parameter growth rate per task is:
  $$\frac{\partial \text{Storage}}{\partial T} = R \ll D \times D$$
  representing an asymptotic compression ratio exceeding $\mathbf{99.8\%}$ compared to storing separate full rank matrices ($768$ floats vs. $589,824$ floats per matrix).

### 7.2 Strict Orthogonal Protection against Catastrophic Forgetting
Catastrophic forgetting in linear representation learning is driven by the inner product interference between task update directions: $\Delta W_t^T \Delta W_{t'} \neq 0$.
In FOCAL-UMS, non-interference is guaranteed at dual levels:
1. **Frequency Domain Orthogonality**:
   $$\langle S_t, S_{t'} \rangle_F = \text{Tr}(S_t^T S_{t'}) = \sum_{(u, v)} S_t(u, v) S_{t'}(u, v) = 0 \quad (\text{since } \Omega_t \cap \Omega_{t'} = \emptyset)$$
   By Parseval's theorem, the discrete Fourier transform is an isometry up to scaling:
   $$\langle \mathcal{F}^{-1}(S_t), \mathcal{F}^{-1}(S_{t'}) \rangle_F = \frac{1}{D^2} \langle S_t, S_{t'} \rangle_F = 0$$
   Hence, the spatial modulation matrices $\Delta W_t$ and $\Delta W_{t'}$ are strictly orthogonal in the Frobenius inner-product space.
2. **Subspace Coordinate Separation**:
   In the consolidated UMS representation, task coordinate vectors $z_t$ are supported on incremental segments of the basis:
   $$\text{supp}(z_t) \subseteq \{r_{cur}^{(t-1)} + 1, \dots, r_{cur}^{(t)}\}$$
   Because columns of $P$ and $Q$ are mutually orthonormal ($P^T P = I_R$, $Q^T Q = I_R$), the cross-task reconstruction interference between task representations is bounded by the residual threshold:
   $$\| P \text{diag}(z_t) Q^T \cdot (P \text{diag}(z_{t'}) Q^T)^T \|_F \le \mathcal{O}(\tau)$$
   ensuring near-zero interference between consolidated tasks.

---
---

# Bản Tiếng Việt: Mô Tả Chi Tiết Phương Pháp Đề Xuất

## 1. Phát biểu Bài toán và Kiến trúc Cơ sở

### 1.1 Bài toán Học Liên tục Không Lưu trữ Mẫu (Exemplar-Free Continual Learning)
Chúng tôi nghiên cứu bài toán Học liên tục không lưu trữ dữ liệu cũ (Exemplar-Free Continual Learning) qua một chuỗi gồm $T$ tác vụ (hoặc miền dữ liệu - domains) tuần tự $\mathcal{D} = \{\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_T\}$. Mỗi tác vụ $\mathcal{T}_t$ bao gồm tập dữ liệu huấn luyện $\mathcal{D}_t = \{(x_{t, i}, y_{t, i})\}_{i=1}^{N_t}$, trong đó $x_{t, i} \in \mathcal{X}$ là ảnh đầu vào và $y_{t, i} \in \mathcal{Y}_t$ là nhãn tương ứng.
Trong thiết lập này, khi mô hình học tác vụ hiện tại $\mathcal{T}_t$, toàn bộ dữ liệu của các tác vụ trong quá khứ $\mathcal{D}_{<t}$ **hoàn toàn không được phép lưu trữ** trong bộ nhớ đệm (không dùng replay buffer / rehearsal). Tại thời điểm suy luận (Inference), mô hình tiếp nhận mẫu kiểm thử $x$ bất kỳ mà không được cung cấp ID tác vụ (Task-Agnostic / Domain-IL), đòi hỏi hệ thống phải tự động nhận diện miền dữ liệu $t^*$ và đưa ra nhãn dự đoán chính xác $\hat{y} \in \mathcal{Y}_{t^*}$.

### 1.2 Kiến trúc Vision Transformer (ViT)
Mô hình nền tảng sử dụng kiến trúc Vision Transformer (ViT-Base) bao gồm $L = 12$ khối Transformer encoder tiêu chuẩn với số chiều vector đặc trưng $D = 768$. Mỗi ảnh đầu vào được chia thành chuỗi các patch tokens kết hợp cùng class token tạo thành tensor biểu diễn $X \in \mathbb{R}^{B \times N \times D}$ ($B$ là batch size, $N = 197$).
Trong mỗi khối Multi-Head Self-Attention (MHSA) có $H = 12$ attention heads ($d = D / H = 64$), các phép chiếu tuyến tính Query ($Q$), Key ($K$), Value ($V$) và Output Projection ($Proj$) được tham số hóa bởi các ma trận trọng số:
$$W_Q, W_K, W_V, W_{proj} \in \mathbb{R}^{D \times D}$$
Trong suốt toàn bộ vòng đời huấn luyện liên tục, các ma trận trọng số tiền huấn luyện $W_Q, W_K, W_V, W_{proj}$ trên tất cả $12$ khối ViT được **đóng băng tuyệt đối (freeze)** nhằm bảo toàn tri thức biểu diễn thị giác tổng quát. Quá trình thích ứng với tác vụ mới chỉ được thực hiện thông qua các tham số điều chế (modulation) hiệu quả cao.

---

## 2. Tổng quan về Kiến trúc Đề xuất

Để giải quyết triệt để hiện tượng quên lãng thảm khốc (catastrophic forgetting) mà vẫn duy trì khả năng tiếp thu tri thức mới (plasticity) trong điều kiện không lưu mẫu, phương pháp **FOCAL-UMS** (*Frequency-Partitioned Orthogonal Consolidation with Unified Modulation Subspace*) được xây dựng dựa trên 3 trụ cột toán học:
1. **Điều chế Cục bộ Đa tầng**: Chỉ tác động vào $6$ khối Transformer trên cùng $\mathcal{L}_{train} = \{6, 7, 8, 9, 10, 11\}$, giữ đông cứng hoàn toàn $6$ khối dưới $\{0, 1, 2, 3, 4, 5\}$ đóng vai trò trích xuất đặc trưng bất biến. Trong các khối được chọn, cả $4$ thành phần $Q, K, V$ và $Proj$ đều được điều chế độc lập.
2. **Phân hoạch Không gian Tần số Rời rạc (FocalExpert)**: Thay vì gán các adapter dày đặc trong không gian không gian ảnh thông thường, mỗi tác vụ được cấp phát một tập toạ độ con rời rạc trong miền tần số Fourier 2D. Sự giao thoa (interference) giữa các tác vụ trong quá trình học bị triệt tiêu hoàn toàn nhờ tính trực giao của phân hoạch tần số.
3. **Không gian Điều chế Hợp nhất (UMS) Khép kín (UMS-in-the-Loop)**: Trong khi train, đường đi tối ưu gradient được chiếu trực tiếp qua toán tử không gian con ứng viên. Kết thúc mỗi tác vụ, ma trận tần số thưa được phân rã SVD và gộp vào một hệ cơ sở trực chuẩn dùng chung $(P, Q)$ thông qua phép quay trơn Cayley trên đa tạp Stiefel. Mô hình loại bỏ hoàn toàn tham số thô của tác vụ cũ, chỉ lưu trữ một vector toạ độ thu gọn $z_t \in \mathbb{R}^R$ cùng mạng gating dự đoán support.

---

## 3. Phân hoạch Tham số trên Miền Tần số Rời rạc (FocalExpert)

### 3.1 Biến đổi Tần số và Điều chế Trọng số
Với mỗi lớp tuyến tính có đầu vào $H \in \mathbb{R}^{B \times N \times D}$ và trọng số cố định $W \in \mathbb{R}^{D \times D}$, phép điều chế cộng thêm ma trận thích ứng $\Delta W_t \in \mathbb{R}^{D \times D}$:
$$H' = H (W + \Delta W_t) = H W + H \Delta W_t$$
Đối với Output Projection, đầu ra sau điều chế là:
$$H_{out} = H_{attn} W_{proj} + H_{attn} \Delta W_{proj, t}$$
Ma trận trọng số $\Delta W_t$ được tái tạo từ phần thực của phép biến đổi Fourier ngược 2D Discrete Inverse Fourier Transform ($\text{iFFT2}$) trên ma trận tần số thưa $S_t \in \mathbb{R}^{D \times D}$:
$$\Delta W_t = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(S_t) \right)$$
trong đó $\alpha = 300.0$ là hệ số tỷ lệ cố định, và công thức biến đổi Fourier ngược 2 chiều:
$$\left[\mathcal{F}^{-1}(S_t)\right]_{j, k} = \frac{1}{D} \sum_{u=0}^{D-1} \sum_{v=0}^{D-1} S_t(u, v) \exp\left( i 2\pi \left(\frac{ju + kv}{D}\right) \right)$$

### 3.2 Phân hoạch Miền Tần số Trực giao
Tổng số hệ số tần số là $D^2 = 768 \times 768 = 589.824$. Một hoán vị giả ngẫu nhiên cố định $\pi$ được xác lập trên toàn bộ $D^2$ toạ độ.
Đối với $T = 10$ tác vụ, phổ tần số được phân hoạch thành $T$ tập con rời rạc từng đôi một $\{\Omega_1, \Omega_2, \dots, \Omega_T\}$:
$$\Omega_t \cap \Omega_{t'} = \emptyset \quad \forall t \neq t', \quad |\Omega_t| = N_{frq} = 5.898$$
Mỗi tác vụ được cấp đúng $N_{frq} = 5.898$ hệ số ($\approx 1.0\%$ kích thước ma trận), đảm bảo độ thưa $99.0\%$. Sau toàn bộ $10$ tác vụ, tổng dung lượng phổ chỉ chiếm $\approx 10\%$ ma trận.
Tác vụ $\mathcal{T}_t$ chỉ tối ưu vector hệ số riêng $\theta_t \in \mathbb{R}^{N_{frq}}$. Ma trận $S_t$ được gán:
$$S_t(u, v) = \begin{cases} \theta_{t, m} & \text{khi } (u, v) = \pi_{t}(m) \in \Omega_t \\ 0 & \text{ngược lại} \end{cases}$$
Do tính chất $\Omega_t \cap \Omega_{t'} = \emptyset$, đạo hàm của hàm mất mát tác vụ này đối với tham số của tác vụ khác luôn bằng 0:
$$\frac{\partial S_t(u, v)}{\partial \theta_{t', m}} = 0 \quad \forall t \neq t'$$

---

## 4. Quá trình Huấn luyện Khép kín: UMS-in-the-Loop

Các phương pháp CL thông thường gặp vấn đề sai lệch mục tiêu (objective mismatch): tham số được tối ưu tự do trong không gian vô hạn chiều lúc train, nhưng sau đó bị cắt xén (truncate SVD) lúc nén, gây suy giảm độ chính xác đột ngột. 
Cơ chế **UMS-in-the-Loop** giải quyết triệt để vấn đề này bằng cách ép quỹ đạo học của $\theta_t$ phải nằm trọn trong không gian con ứng viên ngay trong từng bước lan truyền tiến (forward pass).

### 4.1 Thiết lập Không gian con Ứng viên
Định kỳ mỗi $E_{refresh} = 2$ epoch (và tại đầu tác vụ), ma trận $S_t$ hiện thời được đối chiếu với cơ sở trực chuẩn đã tích lũy $P, Q \in \mathbb{R}^{D \times R}$ ($R = 768$, số chiều hiện tại $r_{cur} \le R$) và phép quay $R_o \in \mathcal{SO}(D)$:
1. Chiếu $S_t$ về hệ quy chiếu chính tắc:
   $$T_t = R_o^T S_t R_o$$
2. Triệt tiêu các thành phần đã nằm trong cơ sở cũ:
   $$U_P = (I - P_{:, :r_{cur}} P_{:, :r_{cur}}^T) T_t, \quad V_Q = T_t (I - Q_{:, :r_{cur}} Q_{:, :r_{cur}}^T)$$
3. Phân rã SVD thu gọn phần dư với rank $r = 72$:
   $$U_P \approx U_r \Sigma_r V_r^T, \quad U_r \in \mathbb{R}^{D \times r}, \quad V_r \in \mathbb{R}^{D \times r}$$
4. Trực chuẩn hóa các vector phần dư qua phép phân rã QR và ghép nối thành hệ cơ sở ứng viên $P_{cand}, Q_{cand} \in \mathbb{R}^{D \times K}$ ($K = r_{cur} + r$):
   $$P_{cand} = [P_{:, :r_{cur}}, \text{qr}(U_r)], \quad Q_{cand} = [Q_{:, :r_{cur}}, \text{qr}(V_r)]$$
5. Chuyển cơ sở ứng viên quay lại hệ toạ độ làm việc:
   $$A = R_o P_{cand} \in \mathbb{R}^{D \times K}, \quad B = R_o Q_{cand} \in \mathbb{R}^{D \times K}$$

### 4.2 Lan truyền Gradient qua Toán tử Chiếu
Với mỗi vị trí tần số $(u, v) = \pi_t(m) \in \Omega_t$, giá trị tái tạo từ biểu diễn đường chéo toạ độ $z \in \mathbb{R}^K$ là:
$$\left[ A \text{diag}(z) B^T \right]_{u, v} = \sum_{k=1}^K A_{u, k} B_{v, k} z_k$$
Toán tử chiếu trên tập support được biểu diễn bởi ma trận $G \in \mathbb{R}^{N_{frq} \times K}$ với hàng thứ $m$ là tích Hadamard:
$$G_{m, :} = A_{u(m), :} \odot B_{v(m), :}, \quad (u(m), v(m)) = \pi_t(m)$$
Trong forward pass, vector tham số $\theta_t \in \mathbb{R}^{N_{frq}}$ được chiếu qua $G$:
$$z_{proj} = G^T \theta_t \in \mathbb{R}^K$$
$$\hat{v}_t = G z_{proj} = G G^T \theta_t \in \mathbb{R}^{N_{frq}}$$
Ma trận tần số $\hat{S}_t$ được tạo lập trực tiếp từ $\hat{v}_t$:
$$\hat{S}_t(u, v) = \begin{cases} \hat{v}_{t, m} & \text{khi } (u, v) = \pi_t(m) \in \Omega_t \\ 0 & \text{ngược lại} \end{cases}$$
Ma trận trọng số điều chế trong quá trình train là:
$$\Delta W_t = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(\hat{S}_t) \right)$$
Nhờ đó, gradient từ hàm loss phân loại $\mathcal{L}_{CE}$ truyền thẳng qua toán tử chiếu $G G^T$:
$$\frac{\partial \mathcal{L}_{task}}{\partial \theta_t} = G G^T \frac{\partial \mathcal{L}_{task}}{\partial \hat{v}_t}$$
Đảm bảo tham số luôn được tối ưu trên mặt phẳng không gian con sẽ gộp ở cuối task.

---

## 5. Hợp nhất Tri thức sau mỗi Task: Không gian Điều chế Hợp nhất (UMS)

Khi hoàn thành epoch cuối của tác vụ $\mathcal{T}_t$, toàn bộ tri thức của $S_t$ được hợp nhất vĩnh viễn vào UMS.

### 5.1 SVD Phần dư và Mở rộng Cơ sở Trực chuẩn
Đưa ma trận $S_t$ về dạng chính tắc $T_t = R_o^T S_t R_o$. Tính ma trận dư sau khi trừ đi hình chiếu trên cơ sở cũ:
$$U_p = T_t - P_{:, :r_{cur}} (P_{:, :r_{cur}}^T T_t)$$
Phân rã SVD phần dư: $U_p = U_r \Sigma_r V_r^T$ với $\Sigma_r = \text{diag}(\sigma_1, \dots, \sigma_r)$. Năng lượng dư được lượng hóa:
$$\mathcal{E}(T_t) = \sum_{j=1}^r \sigma_j^2$$
So sánh với ngưỡng năng lượng $\tau = 10^{-3}$:
- **Nếu $\mathcal{E}(T_t) > \tau$ và $r_{cur} < R$**: Không gian con hiện tại chưa đủ bao quát tác vụ mới. Các hướng kỳ dị dư $U_r \Sigma_r$ được trực chuẩn hóa bằng QR và bổ sung vào $P$ và $Q$:
  $$P_{:, r_{cur}:r_{cur}+\Delta r} \leftarrow \text{qr}(U_r \Sigma_r), \quad Q_{:, r_{cur}:r_{cur}+\Delta r} \leftarrow \text{qr}(V_r')$$
  Số chiều cơ sở tăng thêm: $r_{cur} \leftarrow r_{cur} + \Delta r \le 768$.
- **Nếu $\mathcal{E}(T_t) \le \tau$**: Cơ sở hiện tại đã biểu diễn đủ tốt tác vụ $t$. Không bổ sung thêm cột nào ($\Delta r = 0$), bảo tồn ngân sách cơ sở cho các tác vụ tương lai.

Toạ độ nén $z_t \in \mathbb{R}^R$ của tác vụ được tính bằng phép chiếu đường chéo chính xác:
$$z_t = \text{diag}\left( P_{:, :r_{cur}}^T T_t Q_{:, :r_{cur}} \right) \in \mathbb{R}^{r_{cur}}$$
(và đệm các phần tử còn lại bằng 0 cho đủ kích thước $R = 768$).

### 5.2 Tối ưu hóa Phép quay Cayley trên Đa tạp Stiefel
Phép quay $R_o \in \mathbb{R}^{D \times D}$ đóng vai trò định hướng toàn cục cho không gian hợp nhất. 
Để đảm bảo $R_o$ luôn là ma trận trực chuẩn ($R_o^T R_o = I, \det(R_o) = 1$) mà không cần chiếu lại phức tạp, chúng tôi áp dụng **Biến đổi Cayley**.
Khởi tạo ma trận tự do $W_{rot} \in \mathbb{R}^{D \times D}$, tạo ma trận phản xứng $A = W_{rot} - W_{rot}^T$. Khi đó ma trận quay $R_o \in \mathcal{SO}(D)$ được xác định:
$$R_o = \left( I - \frac{1}{2} A \right)^{-1} \left( I + \frac{1}{2} A \right)$$
Ma trận $W_{rot}$ được tối ưu trong $N_{rot} = 15$ bước bằng Adam ($\text{lr} = 10^{-4}$) theo hàm mất mát đa mục tiêu:
$$\mathcal{L}_{rot} = 1.0 \cdot \mathcal{L}_{mse} + 0.1 \cdot \mathcal{L}_{bce} + 0.01 \cdot \mathcal{L}_{gauss} + 0.1 \cdot \mathcal{L}_{diag} + 2.0 \cdot \mathcal{L}_{anchor}$$

Ý nghĩa toán học của từng thành phần:
1. **$\mathcal{L}_{mse}$ (Tái tạo trên Support)**: Đo sai số giữa ma trận tần số thật $S_t$ và ma trận tái tạo $\hat{S}_{rot} = R_o (P \text{diag}(z_t) Q^T) R_o^T$ chỉ tại các vị trí khác không trong $\Omega_t$:
   $$\mathcal{L}_{mse} = \frac{1}{|\Omega_t|} \sum_{(u, v) \in \Omega_t} \left( S_t(u, v) - [\hat{S}_{rot}]_{u, v} \right)^2$$
2. **$\mathcal{L}_{bce}$ (Phân loại Nhị phân Support)**: Huấn luyện Support Head phân biệt vị trí tần số hoạt động với trọng số dương $w_{pos} \approx 99.0$.
3. **$\mathcal{L}_{gauss}$ (Tiền nghiệm Gaussian)**: Ép các giá trị tái tạo trên $\Omega_t$ tuân theo phân phối chuẩn $\mathcal{N}(0, 1)$, chống bùng nổ biên độ:
   $$\mathcal{L}_{gauss} = \mu_{pred}^2 + (\sigma_{pred} - 1.0)^2$$
4. **$\mathcal{L}_{diag}$ (Triệt tiêu Năng lượng Ngoài Đường chéo)**: Triệt tiêu các phần tử ngoài đường chéo của $Z_t = P^T (R_o^T S_t R_o) Q$, đảm bảo tính độc lập giữa các chiều toạ độ.
5. **$\mathcal{L}_{anchor}$ (Neo Giữ Không gian con Cũ)**: Ngăn chặn phép quay mới làm biến dạng cơ sở và Support Head của các tác vụ cũ:
   $$\mathcal{L}_{anchor} = \frac{1}{2 r_{old}} \left( \| (R_o - R_{prev}) P_{old} \|_F^2 + \| (R_o - R_{prev}) Q_{old} \|_F^2 \right) + \frac{1}{2 r_{old}} \left( \| W_L^{old} - W_{L, prev}^{old} \|_F^2 + \| W_R^{old} - W_{R, prev}^{old} \|_F^2 \right)$$

### 5.3 Mạng Phân loại Hỗ trợ Song tuyến tính (Bilinear Support Head)
Phép biến đổi Fourier ngược tích phân trên toàn bộ lưới $D \times D$. Nếu $99\%$ phần tử zero có nhiễu tái tạo nhỏ, chúng sẽ cộng dồn thành nhiễu không gian rất lớn.
**Bilinear Support Head** $\mathcal{H}_{supp}$ với các ma trận $W_L, W_R \in \mathbb{R}^{D \times R}$ và bias $B \in \mathbb{R}^{D \times D}$ dự đoán mặt nạ xác suất:
$$M_t = \sigma\left( W_L \text{diag}(z_t) W_R^T + B \right) \in [0, 1]^{D \times D}$$
Ma trận tần số nén được lọc cứng bằng mặt nạ:
$$\hat{S}_{gated, t}(u, v) = \begin{cases} [R_o (P \text{diag}(z_t) Q^T) R_o^T]_{u, v} & \text{khi } M_t(u, v) \ge 0.5 \\ 0 & \text{ngược lại} \end{cases}$$
Ma trận trọng số không gian phục hồi hoàn chỉnh là:
$$W_t = \alpha \cdot \text{Re}\left( \mathcal{F}^{-1}(\hat{S}_{gated, t}) \right)$$

---

## 6. Quá trình Đánh giá và Suy luận: Phân định Miền bằng Khoảng cách Mahalanobis

Khi kiểm thử, mô hình đánh giá trên tập test mà không có nhãn tác vụ đi kèm. Quy trình suy luận bao gồm 2 bước:

### 6.1 Mô hình hóa Phân phối Đặc trưng Tác vụ
Cuối mỗi tác vụ $\mathcal{T}_t$, mô hình ViT đông cứng trích xuất class token $h(x) \in \mathbb{R}^D$ của tập train và đưa qua lớp bottleneck tuyến tính $\text{fcArr}_t$:
$$z_{t, i} = \text{fcArr}_t(h(x_{t, i})) \in \mathbb{R}^{d_{fc}} \quad (d_{fc} = 500)$$
Ước lượng vector kỳ vọng $\mu_t$ và ma trận hiệp phương sai $\Sigma_t$:
$$\mu_t = \frac{1}{N_t} \sum_{i=1}^{N_t} z_{t, i}, \quad \Sigma_t = \frac{1}{N_t - 1} \sum_{i=1}^{N_t} (z_{t, i} - \mu_t)(z_{t, i} - \mu_t)^T + 10^{-4} I$$

### 6.2 Phân định Tác vụ và Dự đoán Lớp
Với batch ảnh kiểm thử $X_{test}$, đặc trưng được chiếu qua từng bottleneck tác vụ đã học $k \in \{1, \dots, t\}$ để tính khoảng cách Mahalanobis:
$$d_M(Z_b^{(k)}, \mathcal{N}(\mu_k, \Sigma_k)) = \sqrt{ (Z_b^{(k)} - \mu_k)^T \Sigma_k^{-1} (Z_b^{(k)} - \mu_k) }$$
Tác vụ tương ứng $k^*$ được suy ra từ khoảng cách trung bình nhỏ nhất:
$$k^* = \arg\min_{k} \frac{1}{B} \sum_{b=1}^B d_M(Z_b^{(k)}, \mathcal{N}(\mu_k, \Sigma_k))$$
Sau khi xác định được $k^*$:
1. Khôi phục trọng số điều chế $W_{k^*}$ thông qua toạ độ $z_{k^*}$ từ không gian UMS và nạp vào 6 khối ViT trên.
2. Lan truyền ảnh $X_{test}$ qua mạng ViT đã điều chế để thu được biểu diễn $f_{k^*}(X_{test})$.
3. Đầu phân loại $\text{classifierArr}_{k^*}$ đưa ra nhãn dự đoán cuối cùng:
   $$\hat{c} = \arg\max_{c \in \mathcal{Y}_{k^*}} \left[ \text{classifierArr}_{k^*}\left( \text{fcArr}_{k^*}(f_{k^*}(X_{test})) \right) \right]_c$$

---

## 7. Phân tích Lý thuyết và Độ phức tạp Tính toán

### 7.1 Độ phức tạp Lưu trữ Bộ nhớ
Với $D = 768$, $R = 768$, $d_{fc} = 500$, và $T = 10$:
- Bộ nhớ dùng chung cố định: Cơ sở $P, Q$ ($2 \times 768^2$), bộ sinh Cayley $W_{rot}$ ($768^2$), và Support Head ($3 \times 768^2$).
- **Chi phí biên cho mỗi tác vụ mới**: Chỉ cần đúng **$R = 768$ giá trị toạ độ thực** ($z_t \in \mathbb{R}^{768}$) cùng đầu phân loại tuyến tính.
- Tỷ lệ nén tham số đạt tới **$99.8\%$** so với việc lưu trữ ma trận điều chế đầy đủ ($768$ giá trị float so với $589.824$ giá trị float mỗi ma trận).

### 7.2 Tính Trực giao Triệt tiêu Quên lãng Thảm khốc
Sự quên lãng thảm khốc giữa tác vụ $t$ và $t'$ được triệt tiêu dựa trên 2 cơ chế trực giao kép:
1. **Trực giao trên Miền Tần số**:
   Do $\Omega_t \cap \Omega_{t'} = \emptyset$, tích vô hướng Frobenius giữa 2 ma trận tần số:
   $$\langle S_t, S_{t'} \rangle_F = \sum_{(u, v)} S_t(u, v) S_{t'}(u, v) = 0$$
   Theo định lý Parseval, biến đổi Fourier bảo toàn tích vô hướng:
   $$\langle \Delta W_t, \Delta W_{t'} \rangle_F = \alpha^2 \langle \mathcal{F}^{-1}(S_t), \mathcal{F}^{-1}(S_{t'}) \rangle_F = 0$$
   Hai ma trận điều chế không gian $\Delta W_t$ và $\Delta W_{t'}$ hoàn toàn trực giao với nhau.
2. **Trực chuẩn trong Cơ sở Không gian con**:
   Do các cột của $P$ và $Q$ là trực chuẩn ($P^T P = I_R, Q^T Q = I_R$) và toạ độ $z_t, z_{t'}$ chiếm giữ các dải chỉ số tách biệt, nhiễu xuyên kênh giữa các tác vụ được chặn trên bởi ngưỡng năng lượng kỳ dị $\tau = 10^{-3}$:
   $$\| P \text{diag}(z_t) Q^T \cdot (P \text{diag}(z_{t'}) Q^T)^T \|_F \le \mathcal{O}(\tau)$$
   Đảm bảo khả năng lưu giữ tri thức cũ nguyên vẹn trong suốt vòng đời học liên tục.
