"""
Central Configuration for Fake News Detection System.
Cấu hình tập trung cho toàn bộ hệ thống: đường dẫn, tham số mô hình, bảo mật, và phân quyền.
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
PHOBERT_DIR = os.path.join(MODELS_DIR, "phobert")
BILSTM_DIR = os.path.join(MODELS_DIR, "bilstm")
LSTM_DIR = os.path.join(MODELS_DIR, "lstm")
XLM_ROBERTA_DIR = os.path.join(MODELS_DIR, "xlm_roberta")

# Đường dẫn đầu ra
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Tạo tự động các thư mục cần thiết
for d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    PHOBERT_DIR, BILSTM_DIR, LSTM_DIR, XLM_ROBERTA_DIR,
    FIGURES_DIR, REPORTS_DIR, LOGS_DIR
]:
    os.makedirs(d, exist_ok=True)

# Siêu tham số mặc định cho Huấn luyện
DEFAULT_MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
SEED = 42

# Phân loại 3 nhãn chuẩn quốc tế Fact-checking (ViFactCheck / FEVER)
LABEL_NAMES = {
    0: "Supported (Tin Thật / Được Xác Thực)",
    1: "Refuted (Tin Giả / Bị Bác Bỏ)",
    2: "Not Enough Info (Chưa Đủ Thông Tin)"
}

# Phân quyền người dùng (Roles)
ROLE_ADMIN = "admin"
ROLE_RESEARCHER = "researcher"
ROLE_USER = "user"

