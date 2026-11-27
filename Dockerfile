FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1

# ========================================================
# 1. Cài đặt các thư viện phụ thuộc (Dependencies)
# ========================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ========================================================
# 2. COPY DATA RIÊNG (Dung lượng lớn ~18GB, ít thay đổi)
# Layer này được đặt trước layer code để tận dụng Docker layer cache.
# Khi dữ liệu không đổi, Docker sẽ tái sử dụng cache thay vì copy lại 18GB.
# ========================================================
COPY data/ ./data/

# ========================================================
# 3. COPY CODE & CONFIGS RIÊNG (Thay đổi thường xuyên)
# Tách riêng các thư mục và file source code. Khi chỉnh sửa code,
# Docker chỉ invalidate cache từ layer này, giúp build lại trong vài giây.
# ========================================================
COPY backbone/ ./backbone/
COPY configs/ ./configs/
COPY datasets/ ./datasets/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY training_scripts/ ./training_scripts/
COPY utils/ ./utils/
COPY tests/ ./tests/
COPY *.py ./

# Tạo sẵn các thư mục phục vụ lưu trữ kết quả và checkpoint khi chạy container
RUN mkdir -p checkpoints experiment_checkpoints training_results

# Default command
CMD ["/bin/bash"]