# src\database\models.py
"""
Database schemas and table initializers.
Hỗ trợ đầy đủ: Người dùng, Phân quyền, Lịch sử kiểm tra, Kho bằng chứng,
Quản lý Dataset, Phiên bản Mô hình AI và Audit Logs.
"""

from datetime import datetime
from src.database.connection import get_connection

def init_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Bảng người dùng và phân quyền (Auth & Roles)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT,
        role TEXT NOT NULL DEFAULT 'user', -- 'admin', 'researcher', 'user'
        status TEXT NOT NULL DEFAULT 'active', -- 'active', 'locked'
        created_at TEXT NOT NULL
    )
    """)

    # 2. Bảng lịch sử kiểm tra bài viết
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT DEFAULT 'Khách',
        content TEXT NOT NULL,
        headline TEXT,
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

    # 3. Bảng kho bằng chứng đã xác minh (Evidence Knowledge Base)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evidence_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        verdict TEXT NOT NULL, -- 'Supported' hoặc 'Refuted'
        publisher TEXT NOT NULL,
        source_url TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # 4. Bảng phản hồi khiếu nại của người dùng
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedbacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        history_id INTEGER,
        user_note TEXT,
        reported_label TEXT,
        status TEXT DEFAULT 'Chờ xử lý',
        created_at TEXT NOT NULL,
        FOREIGN KEY (history_id) REFERENCES history(id)
    )
    """)

    # 5. Bảng quản lý tập dữ liệu (Dataset Management)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        split_ratio TEXT NOT NULL,
        version TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # 6. Bảng quản lý phiên bản mô hình AI (Model Versioning)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        architecture TEXT NOT NULL,
        checkpoint_path TEXT NOT NULL,
        accuracy REAL,
        macro_f1 REAL,
        is_active INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """)

    # 7. Bảng nhật ký hoạt động (Audit Logs)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()

    # Nạp tài khoản Admin và Researcher mặc định nếu chưa có
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        import hashlib
        # Password mặc định cho tài khoản admin là 'admin123'
        pwd_admin = hashlib.sha256("admin123".encode()).hexdigest()
        pwd_user = hashlib.sha256("user123".encode()).hexdigest()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO users (username, password_hash, full_name, email, role, status, created_at)
        VALUES 
        ('admin', ?, 'Quản Trị Viên Hệ Thống', 'admin@fakenews.vn', 'admin', 'active', ?),
        ('nghiencuu', ?, 'Nghiên Cứu Viên AI', 'researcher@fakenews.vn', 'researcher', 'active', ?),
        ('user', ?, 'Người Dùng Kiểm Thử', 'user@fakenews.vn', 'user', 'active', ?)
        """, (pwd_admin, now, pwd_admin, now, pwd_user, now))
        conn.commit()

    # Nạp bằng chứng mẫu
    cursor.execute("SELECT COUNT(*) FROM evidence_sources")
    if cursor.fetchone()[0] == 0:
        sample_evidences = [
            (
                "Đính chính tin đồn uống nước lá ổi và chanh sả chữa ung thư",
                "Bộ Y tế khẳng định không có bất kỳ cơ sở khoa học nào chứng minh uống nước lá ổi, chanh sả có thể tiêu diệt tế bào ung thư.",
                "Refuted",
                "Bộ Y Tế",
                "https://moh.gov.vn",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Cảnh báo tin nhắn mạo danh tặng tiền 4G từ nhà mạng",
                "Cục An toàn thông tin (Bộ TT&TT) cảnh báo chiêu trò lừa đảo nhắn tin tổng đài 1900 để nhận tiền cước 4G.",
                "Refuted",
                "Cục An Toàn Thông Tin",
                "https://ais.gov.vn",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            (
                "Công bố lịch thi tốt nghiệp THPT Quốc Gia chính thức",
                "Bộ Giáo dục và Đào tạo chính thức ban hành kế hoạch tổ chức kỳ thi tốt nghiệp THPT trên toàn quốc vào cuối tháng 6.",
                "Supported",
                "Bộ GD&ĐT",
                "https://moet.gov.vn",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        ]
        cursor.executemany("""
        INSERT INTO evidence_sources (title, content, verdict, publisher, source_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, sample_evidences)
        conn.commit()

    conn.close()

if __name__ == "__main__":
    init_tables()
    print("Database tables initialized successfully.")

