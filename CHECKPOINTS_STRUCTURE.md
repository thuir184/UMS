# Cấu Trúc Thư Mục Checkpoints & Dữ Liệu Phân Tích (FOCAL-UMS)

Tài liệu này mô tả chi tiết kiến trúc lưu trữ, định dạng dữ liệu và ý nghĩa của các chỉ số trong thư mục `checkpoints/`. Hệ thống checkpointing được thiết kế nhằm phục vụ 2 mục đích cốt lõi:
1. **Lưu trữ & Phục hồi hoàn chỉnh trạng thái mô hình**: Cho phép nạp lại checkpoint bất kỳ lúc nào để đánh giá hoặc tiếp tục huấn luyện mà không phụ thuộc vào bộ nhớ tạm.
2. **Phân tích chuyên sâu & Tìm hướng cải tiến thuật toán**: Xuất toàn bộ thuộc tính toán học, chỉ số chẩn đoán của **UMS Consolidator** và **MoE / Experts** ra định dạng JSON sạch và PyTorch tensor `.pt` sau mỗi task để dễ dàng biểu diễn trực quan, vẽ đồ thị tiến trình và đo lường can thiệp không gian con (subspace interference).

---

## 1. Sơ Đồ Cây Thư Mục Tổng Thể

```text
checkpoints/
└── {run_name}/                              # Định danh thử nghiệm (VD: LEAR_seq-cifar10 hoặc theo --ckpt_name)
    ├── run_metadata.json                    # Cấu hình môi trường, phần cứng, thông tin chạy và toàn bộ args
    ├── overall_analysis.json                # Tổng hợp chuỗi thời gian (trajectories) qua toàn bộ các task
    │
    ├── task_0/                              # Dữ liệu được lưu sau khi hoàn thành Task 0
    │   ├── task_summary.json                # Báo cáo tổng hợp task 0: accuracy, CELoss, tóm tắt UMS & MoE
    │   │
    │   ├── model/                           # Phân hệ lưu trữ trọng số toàn diện của mô hình
    │   │   └── model_checkpoint.pt          # PyTorch checkpoint: state_dict, heads per-task, Gaussian dists, opt
    │   │
    │   ├── ums/                             # Phân hệ UMS Consolidator (Toàn bộ các block Attention)
    │   │   ├── ums_summary.json             # Tổng hợp chỉ số UMS: rank trung bình, recon error, coherence
    │   │   ├── Block9-Key_ums.pt            # Tensors UMS của Block 9 Key (P, Q, W_rot, R_o, z_store)
    │   │   ├── Block9-Key_analysis.json     # Chi tiết chỉ số phân tích UMS của Block 9 Key
    │   │   ├── Block9-Val_ums.pt            # Tensors UMS của Block 9 Value
    │   │   ├── Block9-Val_analysis.json     # Chi tiết chỉ số phân tích UMS của Block 9 Value
    │   │   ├── Block10-Key_ums.pt
    │   │   ├── Block10-Key_analysis.json
    │   │   ├── Block10-Val_ums.pt
    │   │   ├── Block10-Val_analysis.json
    │   │   ├── Block11-Key_ums.pt
    │   │   ├── Block11-Key_analysis.json
    │   │   ├── Block11-Val_ums.pt
    │   │   └── Block11-Val_analysis.json
    │   │
    │   └── moe/                             # Phân hệ Mixture of Experts & Routing (Toàn bộ các block Attention)
    │       ├── moe_summary.json             # Tổng hợp chỉ số MoE: quota, số expert bị khóa, mask ratio
    │       ├── Block9-Key_moe.pt            # Trọng số router, coeff, mask, indices của Block 9 Key
    │       ├── Block9-Key_analysis.json     # Tần suất routing, expert được gán, chuẩn coeff của Block 9 Key
    │       ├── Block9-Val_moe.pt
    │       ├── Block9-Val_analysis.json
    │       ├── Block10-Key_moe.pt
    │       ├── Block10-Key_analysis.json
    │       ├── Block10-Val_moe.pt
    │       ├── Block10-Val_analysis.json
    │       ├── Block11-Key_moe.pt
    │       ├── Block11-Key_analysis.json
    │       ├── Block11-Val_moe.pt
    │       └── Block11-Val_analysis.json
    │
    ├── task_1/                              # Dữ liệu tương tự cho Task 1
    │   ├── task_summary.json
    │   ├── model/
    │   │   └── model_checkpoint.pt
    │   ├── ums/
    │   │   └── ...
    │   └── moe/
    │       └── ...
    └── ...
```

---

## 2. Chi Tiết Các Tệp Tin Cấp Thử Nghiệm (`{run_name}/`)

### 2.1. `run_metadata.json`
Chứa các thông tin cấu hình ban đầu để đảm bảo tính tái lập (reproducibility):
```json
{
  "run_name": "LEAR_seq-cifar10",
  "created_at": "2026-09-30T14:48:00.123456",
  "model_class": "LEAR",
  "torch_version": "2.2.0",
  "cuda_available": true,
  "args": {
    "model": "LEAR",
    "dataset": "seq-cifar10",
    "lr": 0.03,
    "batch_size": 64,
    "n_epochs": 1,
    "ums_basis_rank": 1024,
    "ums_svd_rank": 16,
    "ums_tau": 0.001,
    "ums_rot_steps": 3,
    "ums_rot_lr": 0.05
  }
}
```

### 2.2. `overall_analysis.json`
Được cập nhật lũy tiến sau mỗi task. Tệp tin này tổng hợp các chuỗi thời gian (trajectories) xuyên suốt quá trình học liên tục, giúp vẽ ngay đồ thị xu hướng mà không cần duyệt qua từng thư mục task:
```json
{
  "run_name": "LEAR_seq-cifar10",
  "last_updated": "2026-09-30T15:10:00",
  "trajectories": {
    "tasks": [0, 1, 2],
    "avg_basis_rank_used": [16.0, 32.0, 48.0],
    "avg_capacity_used_pct": [1.56, 3.12, 4.68],
    "avg_relative_recon_error": [0.765, 0.718, 0.692],
    "avg_block_coherence": [0.0, 0.0549, 0.0612],
    "avg_masked_experts": [102.4, 204.8, 307.2],
    "eval_accuracies": [
      [0.852],
      [0.814, 0.887],
      [0.791, 0.865, 0.912]
    ]
  },
  "tasks": {
    "task_0": { ... },
    "task_1": { ... }
  }
}
```

---

## 3. Chi Tiết Từng Phân Hệ Trong `task_{t}/`

### 3.1. Phân Hệ Model (`task_{t}/model/`)

#### `model_checkpoint.pt`
Được lưu bằng `torch.save` với đầy đủ các thành phần trạng thái của kiến trúc Continual Learning:
- `task_id` (int): Chỉ số task hiện tại.
- `model_state_dict` (dict): Toàn bộ trọng số của backbone mạng (`model.state_dict()`).
- `fc_heads_state_dict` (list[dict]): Danh sách trọng số của từng lớp bottleneck `net.fcArr[0 ... t]` (vì `fcArr` là danh sách Python độc lập, việc lưu riêng biệt ngăn ngừa việc mất mát trọng số khi reload).
- `classifier_heads_state_dict` (list[dict]): Danh sách trọng số của từng lớp phân loại `net.classifierArr[0 ... t]`.
- `distributions` (list[dict]): Tham số thống kê của hàm khoảng cách Mahalanobis cho từng task:
  - `mean`: Vector trung bình $\mu \in \mathbb{R}^{\text{fc\_dim}}$ của đặc trưng task.
  - `covariance_matrix`: Ma trận hiệp phương sai $\Sigma \in \mathbb{R}^{\text{fc\_dim} \times \text{fc\_dim}}$.
- `optimizer_state_dict`: Trạng thái optimizer (momentum, learning rates per parameter).
- `scheduler_state_dict`: Trạng thái learning rate scheduler (nếu có).
- `eval_results`: Ma trận/mảng độ chính xác (accuracy) đánh giá sau task.
- `loss_history`: Mảng ghi nhận CELoss trung bình qua các epoch.
- `timestamp`: Dấu thời gian hoàn thành task.

---

### 3.2. Phân Hệ UMS Consolidator (`task_{t}/ums/`)

#### `ums_summary.json`
Tổng hợp các chỉ số UMS quan trọng nhất trên toàn bộ các block attention (cả Key và Value):
```json
{
  "task_id": 1,
  "total_ums_consolidators": 6,
  "basis_capacity_R": 1024,
  "avg_basis_rank_used": 32.0,
  "max_basis_rank_used": 32,
  "min_basis_rank_used": 32,
  "avg_capacity_used_pct": 3.125,
  "avg_relative_recon_error": 0.7165,
  "avg_block_coherence": 0.0549,
  "max_block_coherence": 0.0718,
  "avg_rotation_deviation": 1.412,
  "per_module_metrics": {
    "Block9-Key": {
      "pt_file": "Block9-Key_ums.pt",
      "analysis_file": "Block9-Key_analysis.json",
      "basis_rank_used": 32,
      "basis_capacity": 1024,
      "capacity_used_pct": 3.125,
      "block_coherence": 0.038,
      "rotation_deviation": 1.405,
      "fold_history": [ ... ]
    },
    ...
  }
}
```

#### `{layer_name}_ums.pt`
Lưu trữ các đối tượng tensor phục vụ tái tạo ma trận và phân tích phổ:
- `config`: Bộ siêu tham số (`dim`, `basis_rank` $R$, `svd_rank` $r$, `tau` $\tau$, `rot_steps`, `rot_lr`, `alpha` $\alpha$, `r_cur`, `tasks_folded`).
- `tensors`:
  - `P`: Ma trận cơ sở trực chuẩn bên trái $(\text{dim} \times R)$.
  - `Q`: Ma trận cơ sở trực chuẩn bên phải $(\text{dim} \times R)$.
  - `P_active`: Các cột cơ sở đang kích hoạt $(\text{dim} \times r_{\text{cur}})$.
  - `Q_active`: Các cột cơ sở đang kích hoạt $(\text{dim} \times r_{\text{cur}})$.
  - `W_rot`: Ma trận sinh phép quay Cayley $(\text{dim} \times \text{dim})$.
  - `R_o`: Ma trận quay trực giao hoàn chỉnh $R_o = (I - A/2)^{-1}(I + A/2)$ trong đó $A = W_{\text{rot}} - W_{\text{rot}}^T$.
  - `z_store`: Tensor ngăn xếp chứa các vector tọa độ $[z_0, z_1, \dots, z_t]$, kích thước $(t+1 \times R)$.
  - `w_readout`: Ma trận điều biến trọng số dày đặc $(\text{dim} \times \text{dim})$ được phục hồi qua iFFT cho task $t$.

#### `{layer_name}_analysis.json`
Chứa các đại lượng chẩn đoán số học chính xác (dạng float/list thuần túy):
```json
{
  "name": "Block9-Key",
  "dim": 768,
  "basis_capacity_R": 1024,
  "basis_rank_used_r_cur": 32,
  "basis_capacity_used_pct": 3.125,
  "tasks_folded": 2,
  "basis_orthonormality_error_P": 4.12e-06,
  "basis_orthonormality_error_Q": 3.89e-06,
  "rotation_orthogonality_error": 5.21e-06,
  "rotation_deviation_from_identity": 1.405,
  "overall_block_coherence": 0.038,
  "pairwise_block_coherence": {
    "task_0_vs_task_1": 0.03804
  },
  "coordinates_stats": {
    "task_0": {
      "l2_norm": 3.970,
      "l1_norm": 12.451,
      "active_dimensions": 16,
      "sparsity_pct": 98.43,
      "top_components": [
        {"index": 0, "value": 1.842},
        {"index": 1, "value": 1.215}
      ]
    },
    "task_1": {
      "l2_norm": 4.822,
      "l1_norm": 15.612,
      "active_dimensions": 16,
      "sparsity_pct": 98.43,
      "top_components": [ ... ]
    }
  },
  "coordinates_cross_similarity": {
    "task_0_vs_task_1": 0.0124
  },
  "fold_history": [
    {
      "task_id": 0,
      "tasks_folded": 1,
      "basis_rank_used": 16,
      "basis_capacity": 1024,
      "added_basis_cols": 16,
      "residual_energy": 25.31,
      "block_coherence": 0.0,
      "relative_recon_error": 0.778,
      "z_norm": 3.970,
      "s_norm": 6.09,
      "sparsity": 96.87,
      "nnz": 32,
      "residual_singular_values": [3.365, 2.843, 1.756, 1.679]
    },
    {
      "task_id": 1,
      "tasks_folded": 2,
      "basis_rank_used": 32,
      "basis_capacity": 1024,
      "added_basis_cols": 16,
      "residual_energy": 29.04,
      "block_coherence": 0.038,
      "relative_recon_error": 0.699,
      "z_norm": 4.822,
      "s_norm": 6.69,
      "sparsity": 96.87,
      "nnz": 32,
      "residual_singular_values": [3.381, 2.538, 2.459, 2.261]
    }
  ]
}
```

---

### 3.3. Phân Hệ MoE & Experts (`task_{t}/moe/`)

#### `moe_summary.json`
Theo dõi hạn ngạch và mức độ tiêu thụ các Expert trên toàn mạng:
```json
{
  "task_id": 1,
  "total_moe_modules": 6,
  "num_experts_per_moe": 1024,
  "avg_masked_experts": 204.8,
  "avg_remaining_experts": 819.2,
  "avg_expert_mask_pct": 20.0,
  "per_module_metrics": {
    "Block9-Key": {
      "pt_file": "Block9-Key_moe.pt",
      "analysis_file": "Block9-Key_analysis.json",
      "masked_experts_count": 204,
      "remaining_experts_count": 820,
      "expert_mask_pct": 19.92,
      "assigned_experts_this_task": [102, 103, 104, ...],
      "num_active_experts_this_task": 102,
      "coeff_norm_mean": 1.054,
      "router_weight_norm": 2.412
    },
    ...
  }
}
```

#### `{layer_name}_moe.pt`
Lưu trữ trọng số mạng của Router và các Expert:
- `tensors`:
  - `router_weight`: Ma trận trọng số Linear router $(N_{\text{experts}} \times \text{dim})$.
  - `router_bias`: Vector bias của router $(N_{\text{experts}},)$.
  - `coeff`: Ma trận tham số tần số của các expert $(N_{\text{experts}} \times N_{\text{frq}})$.
  - `expert_mask`: Tensor mặt nạ đánh dấu khóa expert $(-\infty$ cho expert đã cấp phát, $0$ cho expert tự do).
  - `list_indices`: Bảng ánh xạ vị trí tần số 2D $(N_{\text{experts}} \times N_{\text{frq}})$.

#### `{layer_name}_analysis.json`
Thống kê hành vi định tuyến và phân bổ tài nguyên:
- `masked_experts_count`: Số lượng expert đã bị khóa sau task $t$.
- `remaining_experts_count`: Số lượng expert khả dụng cho các task tương lai.
- `expert_mask_pct`: Tỷ lệ phần trăm tài nguyên expert đã tiêu thụ.
- `expert_coeff_norms_summary`: Thống kê độ lớn chuẩn tham số của các expert (mean, std, min, max).
- `task_history`: Lưu lại lịch sử chi tiết cho từng task:
  - `allocated_experts`: Danh sách index các expert được gán cho task.
  - `expert_counts`: Tần suất chọn của từng expert từ router trong suốt quá trình huấn luyện task.
  - `num_active_experts`: Số lượng expert thực tế có phát sinh gradient/được chọn.
  - `allocated_limit`: Hạn ngạch (quota) tối đa cho phép cho mỗi task.
  - `S_norm`, `S_sparsity`, `S_nnz`: Đặc trưng cấu trúc của ma trận tọa độ thưa $S_t$.

---

### 3.4. Báo Cáo Tổng Hợp Task (`task_{t}/task_summary.json`)

Tệp tin này tích hợp toàn bộ thông tin quan trọng nhất của task thành một bản tóm tắt duy nhất:
```json
{
  "task_id": 1,
  "timestamp": "2026-09-30T15:10:00",
  "model_checkpoint_file": "model/model_checkpoint.pt",
  "eval_results": [0.814, 0.887],
  "loss_history": [0.312, 0.245],
  "ums_summary": {
    "total_ums_consolidators": 6,
    "basis_capacity_R": 1024,
    "avg_basis_rank_used": 32.0,
    "avg_capacity_used_pct": 3.125,
    "avg_relative_recon_error": 0.7165,
    "avg_block_coherence": 0.0549
  },
  "moe_summary": {
    "total_moe_modules": 6,
    "num_experts_per_moe": 1024,
    "avg_masked_experts": 204.8,
    "avg_expert_mask_pct": 20.0
  },
  "extra_metadata": {}
}
```

---

## 4. Ứng Dụng Dữ Liệu Cho Phân Tích & Cải Thiện Thuật Toán

| Vấn đề nghiên cứu | Trường dữ liệu cần phân tích | Hướng cải thiện thuật toán gợi ý |
| :--- | :--- | :--- |
| **Bão hòa không gian con (Subspace Saturation)** | `basis_capacity_used_pct`, `added_basis_cols` trong `fold_history` | Nếu $r_{\text{cur}} \to R$ quá nhanh, tăng $\tau$ (`--ums_tau`) hoặc áp dụng kỹ thuật nén pruning các chiều cơ sở ít năng lượng. |
| **Nhiễu chéo giữa các Task (Cross-task Interference)** | `overall_block_coherence`, `coordinates_cross_similarity` ($z_i \cdot z_j$) | Nếu coherence hoặc similarity giữa task cũ và mới tăng cao, bổ sung loss phạt trực giao (orthogonal penalty) giữa các khối canonical frame hoặc vector tọa độ $z$. |
| **Lệch phép quay Cayley (Rotation Drift)** | `rotation_deviation_from_identity`, `relative_recon_error` | Nếu việc tối ưu phép quay làm tăng lỗi tái tạo của các task cũ, giảm `--ums_rot_steps` hoặc áp dụng momentum regularizer giữ $R_o$ gần với trạng thái trước đó. |
| **Cạn kiệt tài nguyên Expert (Expert Depletion)** | `expert_mask_pct`, `num_active_experts` vs `allocated_limit` | Nếu số lượng active expert luôn nhỏ hơn quota, có thể giảm số lượng expert cho mỗi task để tiết kiệm hoặc điều chỉnh entropy loss của router để khai thác đều hơn. |
| **Năng lượng tham số Expert (Coeff Explosion)** | `expert_coeff_norms_summary` | Theo dõi xem chuẩn tham số của các expert về sau có bị bùng nổ hay không để áp dụng Weight Decay hoặc Gradient Clipping phù hợp. |

---

## 5. Hướng Dẫn Sử Dụng Trong Mã Nguồn & Dòng Lệnh

### 5.1. Các Tham Số CLI
Khi chạy các script huấn luyện (ví dụ `python -u main_domain.py ...`), có thể truyền các tham số:
- `--checkpoint_dir <path>`: Thư mục gốc lưu trữ checkpoints (mặc định: `checkpoints`).
- `--ckpt_name <name>`: Đặt tên cho lần chạy (tạo thư mục `checkpoints/<name>/`).
- `--save_recon_matrices`: Cờ tùy chọn để lưu ma trận điều biến dày đặc $W_t$ kích thước $(768 \times 768)$ ra disk.
- `--disable_task_checkpoints`: Tắt tính năng lưu checkpoint sau mỗi task nếu muốn tiết kiệm dung lượng ổ đĩa.

### 5.2. Nạp Lại Checkpoint Bằng Python API
```python
from utils.task_checkpointing import load_task_checkpoint

# Nạp lại checkpoint của Task 1 vào mô hình đã khởi tạo
checkpoint_dir = "checkpoints/LEAR_seq-cifar10/task_1"
checkpoint_data = load_task_checkpoint(
    checkpoint_path_or_task_dir=checkpoint_dir,
    model=model,
    optimizer=optimizer  # (Tùy chọn) nạp lại trạng thái optimizer
)

print(f"Loaded Task ID: {checkpoint_data['task_id']}")
print(f"Eval Accuracies: {checkpoint_data['eval_results']}")
```

### 5.3. Trích Xuất Dữ Liệu Trajectory Để Vẽ Biểu Đồ
```python
from utils.task_checkpointing import load_task_analysis_summary
import matplotlib.pyplot as plt

# Đọc toàn bộ chuỗi số liệu của thử nghiệm
summary = load_task_analysis_summary("checkpoints/LEAR_seq-cifar10")
traj = summary["trajectories"]

tasks = traj["tasks"]
coherence = traj["avg_block_coherence"]
rank_used = traj["avg_basis_rank_used"]

# Vẽ đồ thị mức độ can thiệp (Coherence) qua các task
plt.figure(figsize=(8, 4))
plt.plot(tasks, coherence, marker='o', label='Subspace Block Coherence')
plt.xlabel('Task Index')
plt.ylabel('Coherence')
plt.title('Cross-task Subspace Interference Over Continual Tasks')
plt.grid(True)
plt.savefig('coherence_trend.png')
```
