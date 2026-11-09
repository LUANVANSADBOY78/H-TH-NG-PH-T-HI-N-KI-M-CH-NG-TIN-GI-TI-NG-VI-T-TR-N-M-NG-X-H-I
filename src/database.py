"""
Module Quản lý Cơ sở dữ liệu SQLite cho hệ thống Phát hiện & Kiểm chứng Tin giả.
Bao gồm các bảng:
1. history: Lịch sử các bài viết đã kiểm tra (User & Public)
2. evidence_sources: Kho dữ liệu bằng chứng và nguồn tin đã xác minh (Fact-check Knowledge Base)
3. feedbacks: Phản hồi báo cáo kết quả dự đoán sai từ người dùng
4. audit_logs: Nhật ký hoạt động quản trị viên
"""

import os
import sys
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_PATH = "data/fake_news_system.db"

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path: str = DB_PATH):
    """Khởi tạo cấu trúc các bảng dữ liệu nếu chưa tồn tại."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Bảng lịch sử kiểm tra
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT DEFAULT 'Khách',
        content TEXT NOT NULL,
        source_url TEXT,
        domain TEXT,
        predicted_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        prob_real REAL NOT NULL,
        prob_fake REAL NOT NULL,
        prob_nei REAL DEFAULT 0.0,
        evidence_matched TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # 2. Bảng kho bằng chứng đã xác thực (Fact-checked Evidence)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evidence_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        verdict TEXT NOT NULL, -- 'Supported' (Đúng sự thật) hoặc 'Refuted' (Tin đồn sai sự thật)
        publisher TEXT NOT NULL, -- 'Bộ Y Tế', 'VTV', 'TTXVN', 'Cục An Toàn Thông Tin', etc.
        source_url TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # 3. Bảng phản hồi người dùng (User feedback / Report)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedbacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        history_id INTEGER,
        user_note TEXT,
        reported_label TEXT,
        status TEXT DEFAULT 'Chờ xử lý', -- 'Chờ xử lý', 'Đã duyệt', 'Đã hủy'
        created_at TEXT NOT NULL,
        FOREIGN KEY (history_id) REFERENCES history(id)
    )
    """)

    # 4. Bảng nhật ký hoạt động (Audit log)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        details TEXT,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()

    # Nạp sẵn một số bằng chứng uy tín từ cơ quan chính thống để kiểm chứng
    cursor.execute("SELECT COUNT(*) FROM evidence_sources")
    if cursor.fetchone()[0] == 0:
        sample_evidences = [
            (
                "Đính chính tin đồn uống nước lá ổi và chanh sả chữa ung thư",
                "Bộ Y tế khẳng định không có bất kỳ cơ sở khoa học nào chứng minh uống nước lá ổi, chanh sả có thể tiêu diệt tế bào ung thư. Người bệnh cần tuân thủ phác đồ điều trị chuyên khoa của bệnh viện.",
                "Refuted",
                "Bộ Y Tế",
                "https://moh.gov.vn/tin-tuc-chinh-thong",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Cảnh báo tin nhắn mạo danh tặng tiền 4G từ nhà mạng",
                "Cục An toàn thông tin (Bộ TT&TT) cảnh báo chiêu trò lừa đảo nhắn tin tổng đài 1900 để nhận tiền cước 4G. Đây là thủ đoạn trừ tiền tài khoản viễn thông của người dùng.",
                "Refuted",
                "Cục An Toàn Thông Tin",
                "https://ais.gov.vn/canh-bao-lua-dao",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Công bố lịch thi tốt nghiệp THPT Quốc Gia chính thức",
                "Bộ Giáo dục và Đào tạo chính thức ban hành kế hoạch tổ chức kỳ thi tốt nghiệp THPT trên toàn quốc vào cuối tháng 6 theo đúng quy chế.",
                "Supported",
                "Bộ GD&ĐT",
                "https://moet.gov.vn/lich-thi-thpt",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Bác bỏ tin đồn thất thiệt về tai nạn máy bay tại Vũng Tàu",
                "Cục Hàng không Việt Nam khẳng định toàn bộ các chuyến bay hoạt động bình thường, thông tin rơi máy bay tại Vũng Tàu là hoàn toàn bịa đặt do các trang mạng câu view.",
                "Refuted",
                "Cục Hàng Không Việt Nam",
                "https://caa.gov.vn/thong-cao-chinh-thuc",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Thông tin dịch bệnh mùa hè và khuyến cáo phòng chống",
                "Cục Y tế Dự phòng khuyến cáo người dân giữ vệ sinh an toàn thực phẩm, tiêm chủng đầy đủ để phòng các bệnh truyền nhiễm mùa hè.",
                "Supported",
                "VTV News",
                "https://vtv.vn/suc-khoe/khuyen-cao-mua-he",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        ]
        cursor.executemany("""
        INSERT INTO evidence_sources (title, content, verdict, publisher, source_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, sample_evidences)
        conn.commit()

    conn.close()

# --- Các hàm thao tác với Bảng Lịch Sử ---

def save_history(
    content: str,
    predicted_label: str,
    confidence: float,
    prob_real: float,
    prob_fake: float,
    prob_nei: float = 0.0,
    source_url: str = "",
    domain: str = "",
    evidence_matched: str = "",
    user_name: str = "Người dùng"
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO history (user_name, content, source_url, domain, predicted_label, confidence, prob_real, prob_fake, prob_nei, evidence_matched, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_name, content, source_url, domain, predicted_label, confidence, prob_real, prob_fake, prob_nei, evidence_matched, created_at))
    conn.commit()
    history_id = cursor.lastrowid
    conn.close()
    return history_id

def get_history(limit: int = 50, keyword: str = "") -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    if keyword:
        cursor.execute("""
        SELECT * FROM history 
        WHERE content LIKE ? OR predicted_label LIKE ? OR domain LIKE ?
        ORDER BY id DESC LIMIT ?
        """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
    else:
        cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_history_item(history_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history WHERE id = ?", (history_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()

# --- Các hàm thao tác với Kho Bằng Chứng (Evidence) ---

def get_all_evidences() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evidence_sources ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_evidence(title: str, content: str, verdict: str, publisher: str, source_url: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO evidence_sources (title, content, verdict, publisher, source_url, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (title, content, verdict, publisher, source_url, created_at))
    conn.commit()
    conn.close()

# --- Các hàm thao tác với Báo Cáo Phản Hồi (Feedback) ---

def submit_feedback(history_id: int, user_note: str, reported_label: str):
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO feedbacks (history_id, user_note, reported_label, created_at)
    VALUES (?, ?, ?, ?)
    """, (history_id, user_note, reported_label, created_at))
    conn.commit()
    conn.close()

def get_all_feedbacks() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT f.id, f.history_id, h.content, h.predicted_label, f.reported_label, f.user_note, f.status, f.created_at
    FROM feedbacks f
    LEFT JOIN history h ON f.history_id = h.id
    ORDER BY f.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_feedback_status(feedback_id: int, new_status: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE feedbacks SET status = ? WHERE id = ?", (new_status, feedback_id))
    conn.commit()
    conn.close()

# --- Các hàm Thống kê Hệ thống (Dashboard Analytics) ---

def get_dashboard_stats() -> Dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM history")
    total_checks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE predicted_label LIKE '%Thật%' OR predicted_label LIKE '%Supported%'")
    total_real = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE predicted_label LIKE '%Giả%' OR predicted_label LIKE '%Refuted%'")
    total_fake = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE predicted_label LIKE '%Chưa đủ%' OR predicted_label LIKE '%NEI%'")
    total_nei = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE status = 'Chờ xử lý'")
    pending_feedbacks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM evidence_sources")
    total_evidences = cursor.fetchone()[0]

    conn.close()
    return {
        "total_checks": total_checks,
        "total_real": total_real,
        "total_fake": total_fake,
        "total_nei": total_nei,
        "pending_feedbacks": pending_feedbacks,
        "total_evidences": total_evidences
    }

# Khởi tạo DB ngay khi module được nạp
init_database()

