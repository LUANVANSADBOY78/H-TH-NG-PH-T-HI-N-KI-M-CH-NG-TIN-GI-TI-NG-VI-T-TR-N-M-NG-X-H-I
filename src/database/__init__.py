"""
Database package: exposes core database operations.
"""

import os
import sys

# Đảm bảo đường dẫn gốc của project luôn có trong sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.database.connection import get_connection
    from src.database.models import init_tables
    from src.database.backup import backup_database, restore_database
except ImportError:
    from database.connection import get_connection
    from database.models import init_tables
    from database.backup import backup_database, restore_database

init_database = init_tables
DB_PATH = "data/fake_news_system.db"

# Expose history, evidence, feedback methods from database.py
def get_all_evidences():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evidence_sources ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_history(content, predicted_label, confidence, prob_real, prob_fake, prob_nei=0.0, source_url="", domain="", evidence_matched="", user_name="Khách"):
    from datetime import datetime
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO history (user_name, content, source_url, domain, predicted_label, confidence, prob_real, prob_fake, prob_nei, evidence_matched, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_name, content, source_url, domain, predicted_label, confidence, prob_real, prob_fake, prob_nei, evidence_matched, created_at))
    conn.commit()
    hist_id = cursor.lastrowid
    conn.close()
    return hist_id

def get_history(limit=50, keyword=""):
    conn = get_connection()
    cursor = conn.cursor()
    if keyword:
        cursor.execute("SELECT * FROM history WHERE content LIKE ? OR predicted_label LIKE ? ORDER BY id DESC LIMIT ?", (f"%{keyword}%", f"%{keyword}%", limit))
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

def add_evidence(title, content, verdict, publisher, source_url=""):
    from datetime import datetime
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO evidence_sources (title, content, verdict, publisher, source_url, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (title, content, verdict, publisher, source_url, created_at))
    conn.commit()
    conn.close()

def submit_feedback(history_id, user_note, reported_label):
    from datetime import datetime
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO feedbacks (history_id, user_note, reported_label, created_at)
    VALUES (?, ?, ?, ?)
    """, (history_id, user_note, reported_label, created_at))
    conn.commit()
    conn.close()

def get_all_feedbacks():
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

def update_feedback_status(feedback_id, new_status):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE feedbacks SET status = ? WHERE id = ?", (new_status, feedback_id))
    conn.commit()
    conn.close()

def get_dashboard_stats():
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
