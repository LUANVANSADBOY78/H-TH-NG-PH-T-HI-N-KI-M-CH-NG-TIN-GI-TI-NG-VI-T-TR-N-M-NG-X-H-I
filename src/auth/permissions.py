"""
Module Phân quyền Người dùng theo Vai trò (Role-Based Access Control - RBAC).
Các vai trò:
- admin: Toàn quyền quản trị tài khoản, mô hình, dataset, logs.
- researcher: Quyền huấn luyện mô hình, kiểm tra dataset, xem metrics.
- user: Quyền kiểm tra tin tức, tra cứu lịch sử cá nhân, tương tác chatbot.
"""

def can_access_admin_panel(role: str) -> bool:
    return role in ["admin", "researcher"]

def can_manage_users(role: str) -> bool:
    return role == "admin"

def can_manage_models(role: str) -> bool:
    return role in ["admin", "researcher"]

def can_manage_datasets(role: str) -> bool:
    return role in ["admin", "researcher"]

