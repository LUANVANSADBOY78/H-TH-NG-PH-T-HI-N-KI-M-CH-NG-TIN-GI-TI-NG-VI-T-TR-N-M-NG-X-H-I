"""
Module Xử lý Ngôn ngữ Tự nhiên & Học sâu Nâng cao (Advanced Deep Learning & NLP):
1. Phân tích nội dung bài viết dài (Long-Text Chunking & Aggregation)
2. So sánh tính nhất quán giữa Tiêu đề và Thân bài (Headline vs Body Clickbait Inconsistency)
3. So sánh đối chiếu 2 bài viết (Article Comparison & Contradiction Analysis)
4. Phân tích mức độ kích động & cảm xúc tiêu cực (Emotional Tone & Sensationalism)
5. Phát hiện dữ liệu ngoài phân phối (Out-of-Distribution / OOD Detection)
"""

import sys
import re
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class AdvancedNLPToolkit:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    # 1. Phân tích bài viết dài (Chunking)
    def chunk_long_text(self, text: str, max_words: int = 150, overlap: int = 30) -> list:
        """
        Chia văn bản dài thành các đoạn (chunks) có độ gối đầu (overlap)
        để giải quyết giới hạn max_length=256 của Transformer.
        """
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += (max_words - overlap)
        return chunks

    def aggregate_chunk_predictions(self, chunk_probs: list) -> dict:
        """
        Tổng hợp dự đoán từ các chunk (Max Pooling + Average Pooling).
        Nếu có ít nhất 1 đoạn chứa phát ngôn sai lệch nghiêm trọng, toàn bộ bài sẽ bị cảnh báo.
        """
        if not chunk_probs:
            return {"final_fake_prob": 0.5, "max_chunk_fake": 0.5, "num_chunks": 0}

        fake_probs = [p[1] for p in chunk_probs]
        max_fake = max(fake_probs)
        avg_fake = sum(fake_probs) / len(fake_probs)

        # Trọng số ưu tiên đoạn có nguy cơ cao nhất (Risk-sensitive aggregation)
        final_fake_prob = 0.6 * max_fake + 0.4 * avg_fake

        return {
            "final_fake_prob": round(final_fake_prob, 4),
            "max_chunk_fake": round(max_fake, 4),
            "avg_chunk_fake": round(avg_fake, 4),
            "num_chunks": len(chunk_probs)
        }

    # 2. So sánh Tiêu đề vs Thân bài
    def check_headline_body_consistency(self, headline: str, body: str) -> dict:
        """
        Kiểm tra độ tương đồng giữa Tiêu đề và Thân bài.
        Hỗ trợ chuẩn hóa font chữ mạng xã hội (Math bold / nghiêng) và phân biệt giữa
        tên Fanpage/tiêu đề ngắn với chiêu trò giật tít câu view (Clickbait).
        """
        if not headline or not body:
            return {
                "similarity": 1.0,
                "is_inconsistent": False,
                "warning": "Không đủ thông tin tiêu đề để đối chiếu."
            }

        # Chuẩn hóa font chữ Unicode nghệ thuật (VD: 𝐓𝐈𝐄 𝐌𝐄𝐍 -> TIE MEN)
        import unicodedata
        h_norm = unicodedata.normalize("NFC", unicodedata.normalize("NFKD", str(headline).strip()))
        b_norm = unicodedata.normalize("NFC", unicodedata.normalize("NFKD", str(body).strip()))

        # Kiểm tra nếu các từ trong tiêu đề xuất hiện trực tiếp trong thân bài
        h_words = [w.lower() for w in re.findall(r"\w+", h_norm) if len(w) >= 2]
        b_lower = b_norm.lower()
        matched_words = [w for w in h_words if w in b_lower]
        word_overlap_ratio = len(matched_words) / max(1, len(h_words))

        corpus = [h_norm, b_norm]
        try:
            matrix = self.vectorizer.fit_transform(corpus)
            sim = float(cosine_similarity(matrix[0], matrix[1])[0][0])
        except Exception:
            sim = word_overlap_ratio

        # Nếu có từ khóa tiêu đề nằm trực tiếp trong thân bài -> tăng độ tương đồng thực tế
        effective_sim = max(sim, word_overlap_ratio)

        sensational_triggers = ["khẩn cấp", "chấn động", "cực sốc", "lộ clip", "kịch độc", "đột tử", "ung thư", "kinh hoàng", "bị bắt"]
        has_sensational = any(k in h_norm.lower() for k in sensational_triggers)

        if effective_sim >= 0.12 or word_overlap_ratio >= 0.3:
            warning = "✅ Tiêu đề và nội dung thân bài có sự liên kết nhất quán."
            is_inconsistent = False
        elif has_sensational and effective_sim < 0.10:
            warning = "⚠️ CẢNH BÁO CLICKBAIT: Tiêu đề mang tính giật gân, câu view nhưng nội dung bài không tương xứng."
            is_inconsistent = True
        else:
            warning = "ℹ️ Tiêu đề ngắn gọn / tên trang thương hiệu mạng xã hội (Độ liên kết nội dung hài hòa)."
            is_inconsistent = False

        return {
            "similarity": round(effective_sim, 3),
            "is_inconsistent": is_inconsistent,
            "warning": warning
        }

    # 3. So sánh đối chiếu 2 bài viết
    def compare_two_articles(self, text_a: str, text_b: str) -> dict:
        """
        So sánh mức độ tương đồng và phát hiện điểm mâu thuẫn giữa 2 bài viết.
        """
        if not text_a or not text_b:
            return {"similarity": 0.0, "status": "Vui lòng nhập đủ cả 2 bài viết."}

        corpus = [text_a, text_b]
        matrix = self.vectorizer.fit_transform(corpus)
        sim = float(cosine_similarity(matrix[0], matrix[1])[0][0])

        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))

        common_words = list(words_a.intersection(words_b))
        diff_a = list(words_a.difference(words_b))
        diff_b = list(words_b.difference(words_a))

        # Đánh giá trạng thái
        if sim > 0.70:
            status = "🔴 Hai bài viết có nội dung trùng lặp rất cao (>70%). Khả năng cùng copy một nguồn."
        elif sim > 0.35:
            status = "🟡 Hai bài viết cùng đề cập chung một chủ đề nhưng có góc nhìn hoặc chi tiết khác nhau."
        else:
            status = "🟢 Hai bài viết đề cập hai sự việc hoàn toàn khác nhau."

        return {
            "similarity_score": round(sim * 100, 1),
            "status": status,
            "shared_keywords_count": len(common_words),
            "shared_keywords_sample": common_words[:12],
            "unique_words_a_sample": diff_a[:8],
            "unique_words_b_sample": diff_b[:8]
        }

    # 4. Phân tích cảm xúc & Ngôn ngữ kích động
    def analyze_emotional_tone(self, text: str) -> dict:
        """Đo lường mức độ kích động, giận dữ hoặc hoảng loạn trong văn bản."""
        lower_text = text.lower()
        
        alarm_cues = ["nguy_hiểm", "chết", "kinh_hoàng", "thảm_họa", "sốc", "khẩn_cấp", "hoang_mang", "kịch_độc"]
        positive_cues = ["ổn_định", "phát_triển", "thành_công", "hồi_phục", "khuyến_cáo", "an_toàn", "bình_thường"]

        num_alarm = sum(1 for w in alarm_cues if w in lower_text)
        num_pos = sum(1 for w in positive_cues if w in lower_text)

        # Tính điểm giật gân (Toxicity / Sensationalism) từ 0 đến 100
        score = min(100, num_alarm * 20)
        
        if score > 60:
            tone = "🔴 Giọng điệu kích động, hoảng loạn cao (Đặc trưng của Fake News)"
        elif score > 25:
            tone = "🟡 Giọng điệu cảnh báo vừa phải"
        else:
            tone = "🟢 Giọng điệu trung tính, chuẩn mực báo chí"

        return {
            "sensationalism_score": score,
            "tone_description": tone,
            "alarm_keywords_found": [w for w in alarm_cues if w in lower_text]
        }

    # 5. Phát hiện dữ liệu ngoài phân phối (Out-of-Distribution - OOD)
    def detect_out_of_distribution(self, prob_real: float, prob_fake: float) -> dict:
        """
        Dựa trên Entropy để phát hiện bài viết có chủ đề hoàn toàn xa lạ
        với kho dữ liệu huấn luyện của PhoBERT.
        """
        p1 = max(1e-6, min(1 - 1e-6, prob_real))
        p2 = max(1e-6, min(1 - 1e-6, prob_fake))
        entropy = - (p1 * math.log2(p1) + p2 * math.log2(p2))

        # Entropy cực đại là 1.0 (khi p1 = p2 = 0.5)
        is_ood = entropy > 0.95

        return {
            "entropy": round(entropy, 4),
            "is_ood": is_ood,
            "message": "⚠️ Dữ liệu ngoài phân phối (OOD): Mô hình chưa từng gặp dạng văn bản này trong tập huấn luyện!" if is_ood else "✅ Dữ liệu nằm trong miền phân phối tin cậy của mô hình."
        }

if __name__ == "__main__":
    nlp = AdvancedNLPToolkit()
    t1 = "Uống nước lá ổi và chanh sả giúp tiêu diệt 100% tế bào ung thư sau 7 ngày"
    t2 = "Bộ Y Tế khuyến cáo không nên tin vào bài thuốc lá ổi chữa ung thư"
    comp = nlp.compare_two_articles(t1, t2)
    print("Comparison:", comp)

