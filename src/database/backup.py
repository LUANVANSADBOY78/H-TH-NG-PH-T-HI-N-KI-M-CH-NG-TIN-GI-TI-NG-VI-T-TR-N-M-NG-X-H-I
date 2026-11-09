# src\database\backup.py
"""
Database backup and restore utilities.
"""

import shutil
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DB_PATH, OUTPUTS_DIR

def backup_database(backup_dir: str = os.path.join(OUTPUTS_DIR, "backups")) -> str:
    """Tạo bản sao lưu dữ liệu SQLite có gán nhãn thời gian."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"backup_{timestamp}.db")
    shutil.copy2(DB_PATH, backup_file)
    return backup_file

def restore_database(backup_file: str) -> bool:
    """Khôi phục cơ sở dữ liệu từ file sao lưu."""
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, DB_PATH)
        return True
    return False

if __name__ == "__main__":
    b_file = backup_database()
    print(f"Backup created at: {b_file}")

