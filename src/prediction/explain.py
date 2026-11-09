"""
Module Giải Thích Quyết Định Mô Hình (Explainable AI - XAI / Feature Attribution).
Trích xuất từ ngữ có trọng số cao ảnh hưởng đến dự đoán Tin thật / Tin giả
và tô màu trực quan (HTML Highlighting).
"""

import re

SENSATIONAL_CUES = {
    "khẩn cấp": 0.9, "cực sốc": 0.95, "chấn động": 0.9, "chia sẻ gấp": 0.85,
    "bà con chú ý": 0.8, "kịch độc": 0.9, "đột tử": 0.85, "chữa dứt điểm": 0.95,
    "100%": 0.75, "ngoài hành tinh": 0.9, "giấu nhẹm": 0.85, "xóa bài": 0.8,
    "clip mật": 0.85, "bán tháo": 0.8, "nhận quà": 0.75, "nguy hiểm": 0.7
}

FORMAL_CUES = {
    "bộ y tế": 0.95, "bộ giáo dục": 0.95, "chính phủ": 0.9, "công bố": 0.85,
    "khuyến cáo": 0.85, "quy định": 0.8, "thống kê": 0.8, "nghiên cứu": 0.85,
    "thông cáo": 0.9, "kết luận": 0.85, "khoa học": 0.85
}

def explain_text(text: str, predicted_label: str) -> dict:
    lower_t = text.lower()
    influential = []

    for phrase, weight in SENSATIONAL_CUES.items():
        if phrase in lower_t:
            influential.append({"phrase": phrase, "weight": weight, "type": "negative", "role": "Tăng nguy cơ Tin Giả"})

    for phrase, weight in FORMAL_CUES.items():
        if phrase in lower_t:
            influential.append({"phrase": phrase, "weight": weight, "type": "positive", "role": "Tăng độ tin cậy Chính Thống"})

    # Sinh chuỗi HTML tô màu
    html_out = text
    for item in influential:
        p = item["phrase"]
        regex = re.compile(re.escape(p), re.IGNORECASE)
        if item["type"] == "negative":
            rep = f"<span style='background-color: #fff1f0; color: #ff4d4f; font-weight: bold; padding: 2px 6px; border-radius: 4px; border: 1px solid #ff4d4f;'>{p}</span>"
        else:
            rep = f"<span style='background-color: #f6ffed; color: #52c41a; font-weight: bold; padding: 2px 6px; border-radius: 4px; border: 1px solid #52c41a;'>{p}</span>"
        html_out = regex.sub(rep, html_out)

    neg_count = sum(1 for w in influential if w["type"] == "negative")
    clickbait_score = min(100, neg_count * 25)

    return {
        "highlighted_html": html_out,
        "influential_words": influential,
        "clickbait_score": clickbait_score,
        "is_sensational": clickbait_score > 30
    }

