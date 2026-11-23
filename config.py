"""
Central Configuration for Fake News Detection System.
Cấu hình tập trung cho toàn bộ hệ thống: đường dẫn, tham số mô hình, bảo mật và siêu tham số.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")
DB_PATH = os.path.join(DATA_DIR, "fake_news_system.db")

# Đường dẫn mô hình
MODELS_DIR = os.path.join(BASE_DIR, "models")
BILSTM_DIR = os.path.join(MODELS_DIR, "bilstm")

# Đường dẫn đầu ra thực nghiệm (Checkpoints, Biểu đồ, Metrics)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PHOBERT_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "phobert")
PHOBERT_BEST_DIR = os.path.join(PHOBERT_OUTPUT_DIR, "best")
EVALUATION_DIR = os.path.join(OUTPUTS_DIR, "evaluation")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Tạo tự động các thư mục cần thiết
for d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODELS_DIR, BILSTM_DIR,
    OUTPUTS_DIR, PHOBERT_OUTPUT_DIR, PHOBERT_BEST_DIR,
    EVALUATION_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR
]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# CẤU HÌNH MÔ HÌNH VÀ SIÊU THAM SỐ THỐNG NHẤT
# ==========================================
DEFAULT_MODEL_NAME = "vinai/phobert-base-v2"
NUM_CLASSES = 2
MAX_LENGTH = 256  # Thống nhất cho Train, Eval và Inference

# Siêu tham số huấn luyện (Fine-tuning)
BATCH_SIZE = 8       # Phù hợp GPU RTX 3050 Laptop (4GB VRAM)
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# Phân loại 2 nhãn nhị phân chuẩn
LABEL_NAMES = {
    0: "Tin Thật (Real)",
    1: "Tin Giả (Fake)"
}
LABEL_MAP = {
    0: "Tin Thật",
    1: "Tin Giả"
}

# Tỷ lệ mất cân bằng lớp (ước tính trên tập Train: ~269/121 = 2.22)
# Sử dụng trong Weighted CrossEntropy Loss
CLASS_WEIGHTS = [1.0, 2.22]

# ==========================================
# CẤU HÌNH BẢO MẬT VÀ QUẢN TRỊ VIÊN
# ==========================================
DEFAULT_ADMIN_USER = os.getenv("ADMIN_USER", "admin")
DEFAULT_ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "AdminPass@2026")
SECRET_KEY = os.getenv("SECRET_KEY", "fakenews_secure_session_key_2026")

ROLE_ADMIN = "admin"
ROLE_RESEARCHER = "researcher"
ROLE_USER = "user"
