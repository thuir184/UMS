# FOCAL-UMS: Hợp Nhất Cơ Sở Trực Chuẩn và Không Gian Điều Chế Thống Nhất trên Miền Tần Số cho Học Liên Tục Không Lưu Mẫu

---

## MỤC LỤC
- [1. Phát biểu Bài toán và Kiến trúc Cơ sở](#1-phat-bieu-bai-toan-va-kien-truc-co-so)
  - [1.1 Bài toán Học Liên tục Không Lưu trữ Mẫu (Exemplar-Free Continual Learning)](#11-bai-toan-hoc-lien-tuc-khong-luu-tru-mau-exemplar-free-continual-learning)
  - [1.2 Kiến trúc Vision Transformer (ViT)](#12-kien-truc-vision-transformer-vit)
- [2. Tổng quan về Kiến trúc Đề xuất](#2-tong-quan-ve-kien-truc-de-xuat)
- [3. Phân hoạch Tham số trên Miền Tần số Rời rạc (FocalExpert)](#3-phan-hoach-tham-so-tren-mien-tan-so-roi-rac-focalexpert)
  - [3.1 Biến đổi Tần số và Điều chế Trọng số](#31-bien-doi-tan-so-va-dieu-che-trong-so)
  - [3.2 Phân hoạch Miền Tần số Trực giao](#32-phan-hoach-mien-tan-so-truc-giao)
- [4. Quá trình Huấn luyện Khép kín: UMS-in-the-Loop](#4-qua-trinh-huan-luyen-khep-kin-ums-in-the-loop)
  - [4.1 Thiết lập Không gian con Ứng viên](#41-thiet-lap-khong-gian-con-ung-vien)
  - [4.2 Lan truyền Gradient qua Toán tử Chiếu](#42-lan-truyen-gradient-qua-toan-tu-chieu)
- [5. Hợp nhất Tri thức sau mỗi Task: Không gian Điều chế Hợp nhất (UMS)](#5-hop-nhat-tri-thuc-sau-moi-task-khong-gian-dieu-che-hop-nhat-ums)
  - [5.1 SVD Phần dư và Mở rộng Cơ sở Trực chuẩn](#51-svd-phan-du-va-mo-rong-co-so-truc-chuan)
  - [5.2 Tối ưu hóa Phép quay Cayley trên Đa tạp Stiefel](#52-toi-uu-hoa-phep-quay-cayley-tren-da-tap-stiefel)
  - [5.3 Mạng Phân loại Hỗ trợ Song tuyến tính (Bilinear Support Head)](#53-mang-phan-loai-ho-tro-song-tuyen-tinh-bilinear-support-head)
- [6. Quá trình Đánh giá và Suy luận: Phân định Miền bằng Khoảng cách Mahalanobis](#6-qua-trinh-danh-gia-va-suy-luan-phan-dinh-mien-bang-khoang-cach-mahalanobis)
  - [6.1 Mô hình hóa Phân phối Đặc trưng Tác vụ](#61-mo-hinh-hoa-phan-phoi-dac-trung-tac-vu)
  - [6.2 Phân định Tác vụ và Dự đoán Lớp](#62-phan-dinh-tac-vu-va-du-doan-lop)
  - [6.3 Khôi phục Trọng số Điều chế và Thực hiện Dự đoán](#63-khoi-phuc-trong-so-dieu-che-va-thuc-hien-du-doan)
- [7. Phân tích Lý thuyết và Độ phức tạp Tính toán](#7-phan-tich-ly-thuyet-va-do-phuc-tap-tinh-toan)
  - [7.1 Độ phức tạp Lưu trữ Bộ nhớ](#71-do-phuc-tap-luu-tru-bo-nho)
  - [7.2 Tính Trực giao Triệt tiêu Quên lãng Thảm khốc](#72-tinh-truc-giao-triet-tieu-quen-lang-tham-khoc)

---

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

```
Ảnh đầu vào x ───► ViT Đông cứng (Trích xuất đặc trưng) ───► Class Token h
                                                                    │
                          ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                          ▼                                                                                   ▼
           Bottleneck tuyến tính fcArr[0]                                                      Bottleneck tuyến tính fcArr[k]
                          │                                                                                   │
                 Vector đặc trưng z^(0)                                                              Vector đặc trưng z^(k)
                          │                                                                                   │
        Khoảng cách d_M(z^(0), N(mu_0, Sigma_0))                                            Khoảng cách d_M(z^(k), N(mu_k, Sigma_k))
                          └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                                                    ▼
                                                       Argmin k* (Tác vụ Dự đoán)
                                                                    │
                                                                    ▼
                                                  Khôi phục Trọng số UMS: W_{k*}
                                                  Nạp vào 6 Block ViT [6, 7, 8, 9, 10, 11]
                                                                    │
                                                                    ▼
                                                   Lan truyền tiến qua ViT đã điều chế
                                                                    │
                                                                    ▼
                                                        classifierArr[k*](...) ───► Lớp Dự đoán c*
```

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

### 6.3 Khôi phục Trọng số Điều chế và Thực hiện Dự đoán
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
