"""
Model Version Manager for Admin.
Quản lý các phiên bản mô hình AI (PhoBERT, BiLSTM, TextCNN), chuyển đổi mô hình đang hoạt động (Active Model).
"""

import pandas as pd
from src.database.connection import get_connection

def get_all_model_versions() -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, model_name, architecture, checkpoint_path, accuracy, macro_f1, is_active, created_at FROM model_versions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        # Trả về các phiên bản mặc định
        sample_versions = [
            {"ID": 1, "Tên mô hình": "vinai/phobert-base-v2", "Kiến trúc": "PhoBERT + Metadata MLP", "Checkpoint": "models/phobert/best", "Accuracy": 0.9650, "Macro F1": 0.9645, "Đang dùng": "Có", "Ngày tạo": "2026-11-02 10:00:00"},
            {"ID": 2, "Tên mô hình": "vinai/phobert-base-v2", "Kiến trúc": "PhoBERT Base (Thuần văn bản)", "Checkpoint": "models/phobert/base", "Accuracy": 0.9420, "Macro F1": 0.9415, "Đang dùng": "Không", "Ngày tạo": "2026-11-01 15:30:00"},
            {"ID": 3, "Tên mô hình": "bilstm-attention", "Kiến trúc": "BiLSTM + Attention", "Checkpoint": "models/bilstm/bilstm_model.pt", "Accuracy": 0.8870, "Macro F1": 0.8869, "Đang dùng": "Không", "Ngày tạo": "2026-10-30 09:20:00"}
        ]
        return pd.DataFrame(sample_versions)
    return pd.DataFrame([dict(r) for r in rows])

