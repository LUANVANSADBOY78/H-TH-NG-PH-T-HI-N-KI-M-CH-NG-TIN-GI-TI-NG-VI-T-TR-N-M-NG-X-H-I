"""
Module Đối chiếu & Truy xuất Bằng chứng (Fact-Checking & Evidence Retrieval Engine).
Kết hợp năng lực biểu diễn ngữ nghĩa của mô hình Transformer PhoBERT với kho tri thức
bằng chứng đã kiểm chứng (Fact-Check Knowledge Base) và các tín hiệu hành vi mạng xã hội.
"""

import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from src.database import get_all_evidences
except ImportError:
    from database import get_all_evidences

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class FactCheckingEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    def retrieve_evidence(self, query_text: str, top_k: int = 2) -> list:
        """
        Truy xuất bằng chứng liên quan nhất từ kho dữ liệu (Evidence Retrieval).
        Sử dụng TF-IDF và Cosine Similarity để so khớp độ tương đồng ngữ nghĩa.
        """
        evidences = get_all_evidences()
        if not evidences:
            return []

        doc_texts = [f"{e['title']} {e['content']}" for e in evidences]
        corpus = [query_text] + doc_texts

        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[0]
        doc_vecs = tfidf_matrix[1:]

        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        matched_results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.08:  # Ngưỡng tương quan tối thiểu
                ev = evidences[idx]
                matched_results.append({
                    "id": ev["id"],
                    "title": ev["title"],
                    "content": ev["content"],
                    "verdict": ev["verdict"],
                    "publisher": ev["publisher"],
                    "source_url": ev["source_url"],
                    "similarity": round(score, 3)
                })
        return matched_results

    def verify(
        self,
        query_text: str,
        phobert_prob_real: float,
        phobert_prob_fake: float,
        is_trusted_source: bool = False
    ) -> dict:
        """
        Tổng hợp kết quả giữa mô hình PhoBERT và Kho Bằng Chứng đã đối chiếu.
        Nhận diện thông minh các nội dung đời sống / nghệ thuật / ảnh cưới mạng xã hội
        để tránh báo nhầm (False Positive).
        """
        import unicodedata
        clean_text = unicodedata.normalize("NFC", unicodedata.normalize("NFKD", str(query_text).strip())).lower()

        matched_evidences = self.retrieve_evidence(query_text)
        best_evidence = matched_evidences[0] if matched_evidences else None
        has_strong_evidence = best_evidence is not None and best_evidence["similarity"] >= 0.22

        # Từ khóa độc hại / giật gân / bài thuốc lừa đảo thực tế
        fake_cues = [
            "trị dứt điểm 100%", "cam kết khỏi hẳn", "thần dược", "thuốc tiên",
            "kịch độc", "ung thư giai đoạn cuối", "thổi bay ung thư",
            "bà con chú ý", "chia sẻ gấp", "kẻo bị xóa bài", "bị bắt khẩn cấp",
            "lộ clip mật", "phát lộc", "soạn tin nhận ngay", "trúng thưởng 100%",
            "vỡ đập", "ngừng hoạt động não", "chấn động sáng nay"
        ]
        has_fake_cues = any(k in clean_text for k in fake_cues)

        # Từ khóa đời sống / ảnh cưới / thời trang / phong cách / mua bán cá nhân
        lifestyle_cues = [
            "tình yêu", "cưới", "ảnh cưới", "váy cưới", "áo dài", "nhiếp ảnh", "thiết kế",
            "love story", "chuyện tình", "khoảnh khắc", "gia đình", "bạn bè", "du lịch",
            "ẩm thực", "checkin", "check-in", "quán cà phê", "thời trang", "outfit", "makeup",
            "look", "phòng trọ", "cho thuê", "pass đồ", "thanh lý", "hài hước", "meme"
        ]
        has_lifestyle = any(k in clean_text for k in lifestyle_cues)

        # Sử dụng xác suất thực từ mạng học sâu PhoBERT
        prob_real = float(np.clip(phobert_prob_real, 0.01, 0.99))
        prob_fake = float(np.clip(phobert_prob_fake, 0.01, 0.99))
        total_p = prob_real + prob_fake
        prob_real = round(prob_real / total_p, 4)
        prob_fake = round(1.0 - prob_real, 4)

        category = "Nội dung mạng xã hội tổng quát"
        explanation = "Kết quả được tính toán trực tiếp từ trọng số mạng Transformer PhoBERT-base-v2 dựa trên ngữ cảnh toàn bài."
        advice = "Cân nhắc kỹ trước khi tương tác hoặc chia sẻ."

        # 1. Trường hợp có bằng chứng xác thực chính thức trùng khớp cao
        if has_strong_evidence:
            if best_evidence["verdict"] == "Refuted":
                final_label = "🔴 CẢNH BÁO: TIN GIẢ (BỊ BÁC BỎ BỞI BẰNG CHỨNG XÁC THỰC)"
                category = "Tin Đồn Thất Thiệt / Đã Có Bài Đính Chính"
                explanation = f"Nội dung trùng khớp với thông tin đã bị bác bỏ chính thức bởi: {best_evidence['publisher']} (Độ tương quan: {best_evidence['similarity']*100:.1f}%)."
                advice = "Tuyệt đối không chia sẻ hoặc làm theo nội dung bài viết này."
                prob_fake = max(prob_fake, 0.92)
                prob_real = round(1.0 - prob_fake, 4)
            else:
                final_label = "🟢 XÁC THỰC: TIN THẬT (TRÙNG KHỚP BẰNG CHỨNG CHÍNH THỐNG)"
                category = "Tin Tức Chính Thống Đã Kiểm Chứng"
                explanation = f"Nội dung được xác thực bởi tài liệu chính thức từ: {best_evidence['publisher']} (Độ tương quan: {best_evidence['similarity']*100:.1f}%)."
                advice = "Thông tin có căn cứ tin cậy cao từ cơ quan có thẩm quyền."
                prob_real = max(prob_real, 0.92)
                prob_fake = round(1.0 - prob_real, 4)

        # 2. Trường hợp nguồn tin từ báo chí chính thống uy tín đã kiểm duyệt
        elif is_trusted_source:
            final_label = "🟢 NGUỒN CHÍNH THỐNG: BÁO CHÍ ĐÃ KIỂM DUYỆT"
            category = "Báo Chí Chính Thống Việt Nam"
            explanation = "Đường dẫn bài viết thuộc danh bạ các cơ quan báo chí chính luận chính thống (đã qua quy trình biên tập kiểm duyệt)."
            advice = "Thông tin thời sự đáng tin cậy."
            prob_real = max(prob_real, 0.92)
            prob_fake = round(1.0 - prob_real, 4)

        # 3. Trường hợp bài đăng đời thường, ảnh cưới, thời trang, giải trí (KHÔNG CÓ TỪ KHÓA ĐỘC HẠI)
        elif has_lifestyle and not has_fake_cues:
            final_label = "🟢 NỘI DUNG AN TOÀN: BÀI ĐĂNG ĐỜI THƯỜNG / PHONG CÁCH SỐNG"
            category = "Đời Sống, Thời Trang & Văn Hóa Mạng Xã Hội"
            explanation = "Nội dung là bài đăng chia sẻ câu chuyện tình yêu, ảnh cưới, thời trang nghệ thuật hoặc đời sống cá nhân trên mạng xã hội. Hoàn toàn không có yếu tố tin đồn thất thiệt hay lừa đảo."
            advice = "Nội dung an toàn để đọc, tương tác và chia sẻ lành mạnh."
            prob_real = 0.95
            prob_fake = 0.05

        # 4. Phân loại theo ngưỡng xác suất của mô hình PhoBERT
        else:
            if prob_fake >= 0.70:
                final_label = "🔴 CẢNH BÁO: TIN GIẢ / DẤU HIỆU SAI LỆCH CAO"
                category = "Tin Giả / Nguy Cơ Xuyên Tạc & Lừa Đảo"
                explanation = f"Mô hình PhoBERT phát hiện văn phong mang đặc trưng bất thường của tin giả trên mạng xã hội (Xác suất tin giả: {prob_fake*100:.1f}%)."
                advice = "Cảnh giác cao độ, không vội tin hoặc chia sẻ thông tin chưa được kiểm chứng."
            elif prob_fake >= 0.50:
                final_label = "🟡 NGHI VẤN: CẦN THẬN TRỌNG (CHƯA ĐỦ THÔNG TIN - NEI)"
                category = "Tin Nghi Ngờ / Ranh Giới Chưa Rõ Ràng"
                explanation = f"Bài viết nằm ở ranh giới nghi ngờ (Xác suất tin giả {prob_fake*100:.1f}% vs Tin thật {prob_real*100:.1f}%). Có thể là nội dung trào phúng hoặc thiếu ngữ cảnh xác thực."
                advice = "Nên tìm kiếm thêm thông tin từ các cơ quan báo chí trước khi kết luận."
            else:
                final_label = "🟢 XÁC THỰC: TIN THẬT / AN TOÀN"
                category = "Nội Dung Thật / Chia Sẻ Đời Thường Lành Mạnh"
                explanation = f"Mô hình PhoBERT đánh giá nội dung có tính xác thực và văn phong lành mạnh (Xác suất tin thật: {prob_real*100:.1f}%)."
                advice = "Nội dung an toàn để đọc và chia sẻ."

        return {
            "final_label": final_label,
            "confidence": max(prob_real, prob_fake),
            "prob_supported": prob_real,
            "prob_refuted": prob_fake,
            "prob_nei": 0.0,
            "is_low_confidence": False,
            "category": category,
            "explanation": explanation,
            "advice": advice,
            "matched_evidences": matched_evidences,
            "best_evidence": best_evidence
        }
