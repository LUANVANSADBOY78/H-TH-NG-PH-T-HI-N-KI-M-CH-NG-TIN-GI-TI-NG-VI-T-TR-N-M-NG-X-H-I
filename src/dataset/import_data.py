"""
Module Import Dữ liệu & Đăng ký Dataset vào Hệ thống.
"""

import os
import shutil
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RAW_DATA_DIR
from src.database.connection import get_connection

def import_and_register_dataset(file_obj, dataset_name: str, split_ratio: str = "70/15/15", version: str = "v1.0") -> tuple:
    """Import file CSV/Excel vào thư mục raw và đăng ký vào CSDL."""
    if file_obj is None:
        return False, "Vui lòng chọn một file dữ liệu hợp lệ!"

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        return False, "Chỉ hỗ trợ file dạng .csv hoặc .xlsx!"

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        return False, f"Lỗi đọc file: {str(e)}"

    # Kiểm tra cột tối thiểu
    if "text" not in df.columns or "label" not in df.columns:
        return False, "File dữ liệu bắt buộc phải có 2 cột 'text' và 'label'!"

    target_name = f"{dataset_name.strip().replace(' ', '_')}_{version}.csv"
    dest_path = os.path.join(RAW_DATA_DIR, target_name)
    df.to_csv(dest_path, index=False, encoding="utf-8")

    # Lưu vào CSDL
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO datasets (name, file_path, row_count, split_ratio, version, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (dataset_name.strip(), dest_path, len(df), split_ratio, version, now))
    conn.commit()
    conn.close()

    return True, f"Import thành công tập dữ liệu '{dataset_name}' ({len(df)} dòng) tại {dest_path}!"

