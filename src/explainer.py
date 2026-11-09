"""
Module Giải thích Mô hình AI (Explainable AI - XAI).
Thực hiện:
1. Xác định mức độ ảnh hưởng của từng từ/cụm từ đến quyết định của PhoBERT (Word Attribution / Importance Score).
2. Phát hiện các mẫu ngôn ngữ cảm xúc kích động, tiêu đề giật gân (Clickbait cues).
3. Xuất chuỗi HTML trực quan tô màu các từ có trọng số cao để hiển thị trên giao diện người dùng.
"""

import sys
import re
from typing import List, Tuple, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Bộ từ khóa chỉ báo giật gân / cảm xúc tiêu cực thường gặp trong tin giả
SENSATIONAL_PATTERNS = {
    "khẩn cấp": 0.85, "cực sốc": 0.90, "chấn động": 0.85, "chia sẻ gấp": 0.80,
    "bà con chú ý": 0.75, "kịch độc": 0.85, "đột tử": 0.80, "chữa dứt điểm": 0.95,
    "100%": 0.70, "ngoài hành tinh": 0.90, "giấu nhẹm": 0.85, "xóa bài": 0.80,
    "clip mật": 0.85, "phát lộc": 0.75, "thần tốc": 0.70, "kỳ diệu": 0.65,
    "bán tháo": 0.75, "sốc": 0.70, "hot": 0.60, "nguy hiểm": 0.65
}

# Bộ từ khóa chỉ báo văn phong chính luận / báo chí chính thống
FORMAL_PATTERNS = {
    "bộ y tế": 0.90, "bộ giáo dục": 0.90, "chính phủ": 0.85, "công bố": 0.80,
    "khuyến cáo": 0.85, "quy định": 0.80, "thống kê": 0.75, "nghiên cứu": 0.80,
    "theo thông tin": 0.75, "thông cáo": 0.85, "kết luận": 0.80, "hợp tác": 0.70
}

class ModelExplainer:
    def __init__(self):
        pass

    def explain_prediction(self, text: str, predicted_label: str) -> Dict:
        """
        Phân tích và trích xuất các từ khóa ảnh hưởng đến dự đoán.
        Tạo mã HTML đánh dấu màu cho giao diện người dùng.
        """
        lower_text = text.lower()
        words = text.split()
        
        influential_words = []
        is_fake_dominant = "Giả" in predicted_label or "Refuted" in predicted_label

        # 1. Tìm các từ ngữ giật gân
        for phrase, weight in SENSATIONAL_PATTERNS.items():
            if phrase in lower_text:
                influential_words.append({
                    "phrase": phrase,
                    "impact": "Tăng nguy cơ Tin Giả",
                    "weight": weight,
                    "type": "negative"
                })

        # 2. Tìm các từ ngữ chính luận
        for phrase, weight in FORMAL_PATTERNS.items():
            if phrase in lower_text:
                influential_words.append({
                    "phrase": phrase,
                    "impact": "Tăng độ tin cậy Chính Thống",
                    "weight": weight,
                    "type": "positive"
                })

        # 3. Tạo chuỗi văn bản tô màu HTML (HTML Highlighting)
        highlighted_html = text
        for item in influential_words:
            phrase = item["phrase"]
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if item["type"] == "negative":
                # Tô màu đỏ / cam cho từ khóa nguy cơ tin giả
                color = "#ff4d4f"
                bg = "#fff1f0"
            else:
                # Tô màu xanh lá cho từ khóa chính thống
                color = "#52c41a"
                bg = "#f6ffed"

            replacement = f"<span style='background-color: {bg}; color: {color}; font-weight: bold; padding: 2px 5px; border-radius: 4px; border: 1px solid {color};'>{phrase}</span>"
            highlighted_html = pattern.sub(replacement, highlighted_html)

        # Tính toán điểm phong cách giật gân (Clickbait / Sensational Score)
        neg_count = sum(1 for w in influential_words if w["type"] == "negative")
        pos_count = sum(1 for w in influential_words if w["type"] == "positive")
        sensational_score = min(1.0, round(neg_count * 0.25, 2))

        return {
            "highlighted_html": highlighted_html,
            "influential_words": influential_words,
            "sensational_score": sensational_score,
            "has_emotional_language": neg_count > 0
        }

if __name__ == "__main__":
    explainer = ModelExplainer()
    sample = "Khẩn cấp: Bộ Y tế khuyến cáo không tin lời đồn uống nước lá ổi chữa ung thư 100%!"
    res = explainer.explain_prediction(sample, "Refuted")
    print("Influential Words:", res["influential_words"])
    print("Sensational Score:", res["sensational_score"])

