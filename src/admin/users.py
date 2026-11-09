"""
User management operations for Admin.
Thêm, sửa, khóa, mở khóa, xóa và đổi quyền người dùng.
"""

import pandas as pd
from src.database.connection import get_connection

def get_all_users() -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, full_name, email, role, status, created_at FROM users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["ID", "Username", "Họ tên", "Email", "Vai trò", "Trạng thái", "Ngày tạo"])
    return pd.DataFrame([dict(r) for r in rows])

def toggle_user_status(user_id: int) -> tuple:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Không tìm thấy người dùng!"
    
    new_status = "locked" if row["status"] == "active" else "active"
    cursor.execute("UPDATE users SET status = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return True, f"Đã chuyển trạng thái người dùng #{user_id} sang '{new_status}'!"

def change_user_role(user_id: int, new_role: str) -> tuple:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()
    conn.close()
    return True, f"Đã cập nhật quyền người dùng #{user_id} thành '{new_role}'!"

