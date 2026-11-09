"""
Module Xác thực & Đăng nhập Người dùng (Authentication & Registration).
Hỗ trợ:
- Mã hóa mật khẩu bảo mật bằng SHA-256
- Đăng ký tài khoản mới với vai trò mặc định
- Đăng nhập xác thực và kiểm tra trạng thái khóa tài khoản
- Đổi mật khẩu
"""

import hashlib
from datetime import datetime
from src.database.connection import get_connection

def hash_password(password: str) -> str:
    """Băm mật khẩu bằng thuật toán SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username: str, password: str, full_name: str, email: str = "", role: str = "user") -> tuple:
    """Đăng ký tài khoản người dùng mới."""
    if not username or not password or not full_name:
        return False, "Vui lòng nhập đầy đủ tên đăng nhập, mật khẩu và họ tên!"
    if len(password) < 6:
        return False, "Mật khẩu phải có tối thiểu 6 ký tự!"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username.strip().lower(),))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "Tên đăng nhập này đã được sử dụng!"

    pwd_hash = hash_password(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO users (username, password_hash, full_name, email, role, status, created_at)
    VALUES (?, ?, ?, ?, ?, 'active', ?)
    """, (username.strip().lower(), pwd_hash, full_name.strip(), email.strip(), role, now))
    conn.commit()
    conn.close()
    return True, "Đăng ký tài khoản thành công! Bạn có thể đăng nhập ngay."

def authenticate_user(username: str, password: str) -> tuple:
    """Xác thực đăng nhập người dùng."""
    if not username or not password:
        return False, None, "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu!"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, username, password_hash, full_name, email, role, status
    FROM users WHERE username = ?
    """, (username.strip().lower(),))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        return False, None, "Tên đăng nhập không tồn tại!"

    if user["status"] == "locked":
        return False, None, "Tài khoản của bạn đã bị khóa bởi Quản trị viên!"

    pwd_hash = hash_password(password)
    if pwd_hash != user["password_hash"]:
        return False, None, "Mật khẩu không chính xác!"

    user_info = {
        "id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"]
    }
    return True, user_info, f"Đăng nhập thành công! Chào mừng {user['full_name']} ({user['role']})."

