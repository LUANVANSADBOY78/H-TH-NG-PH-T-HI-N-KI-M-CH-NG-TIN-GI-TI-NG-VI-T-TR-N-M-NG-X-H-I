"""
Dataset Manager for Admin.
Quản lý các tập dữ liệu huấn luyện, thông tin chất lượng và phân chia dữ liệu.
"""

import pandas as pd
from src.database.connection import get_connection

def get_registered_datasets() -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, file_path, row_count, split_ratio, version, created_at FROM datasets ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["ID", "Tên tập dữ liệu", "Đường dẫn", "Số dòng", "Tỉ lệ chia", "Phiên bản", "Ngày tạo"])
    return pd.DataFrame([dict(r) for r in rows])

