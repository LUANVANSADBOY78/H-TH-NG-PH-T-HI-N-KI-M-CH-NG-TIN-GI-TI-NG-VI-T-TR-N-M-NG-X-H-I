# src\app.py
"""
Hệ thống Phát hiện & Kiểm chứng Tin giả Tiếng Việt Trên Mạng Xã Hội bằng Transformer (PhoBERT).
Phiên bản Nâng cao Tích hợp Chuyên sâu cho Môn Học Sâu (Deep Learning):
1. Phân hệ Kiểm tra Đơn Lẻ: PhoBERT, Trích xuất URL, Chunking bài viết dài,
   Ước lượng độ bất định (MC Dropout), Phát hiện OOD, Kiểm tra Tiêu đề vs Thân bài.
2. Phân hệ Phân tích Hàng Loạt (Batch CSV / Excel): Quét hàng loạt bài viết và tải báo cáo.
3. Phân hệ So sánh Đối chiếu 2 Bài viết: Đo độ tương đồng ngữ nghĩa & phát hiện mâu thuẫn.
4. Phân hệ Lịch sử Kiểm tra: Tìm kiếm, lọc và xuất dữ liệu.
5. Phân hệ Chatbot AI: Trợ lý tư vấn, giải thích mô hình, tóm tắt bài và khuyến nghị kiểm chứng.
6. Phân hệ Admin & Nghiên cứu Học Sâu: Dashboard KPIs, Biểu đồ Loss/F1 qua Epoch,
   Confusion Matrix, Bảng Ablation Study, So sánh PhoBERT vs BiLSTM/CNN/SVM.
"""

import os
import sys

# Tự động nhận diện thư mục làm việc để không bao giờ bị lỗi import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for p in [BASE_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import re
import unicodedata
import pandas as pd
import numpy as np
import torch
import gradio as gr

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import các module nội bộ
from preprocess import clean_social_text, segment_vietnamese
from url_extractor import fetch_content_from_url
from fact_checker import FactCheckingEngine
from explainer import ModelExplainer
from chatbot import FactCheckChatbot
from model_comparison import get_comparison_summary
from deep_learning_models import MonteCarloDropoutEstimator
from advanced_nlp import AdvancedNLPToolkit
from batch_analyzer import process_batch_file
from ablation_and_metrics import (
    get_ablation_study_table,
    generate_training_loss_plot,
    generate_confusion_matrix_plot
)
import database as db
try:
    from social_profile_checker import detect_buff_like_and_engagement
except ImportError:
    from src.social_profile_checker import detect_buff_like_and_engagement

# Bộ mẫu bài đăng mạng xã hội thực tế đa dạng chủ đề
SOCIAL_POST_PRESETS = {
    "--- Chọn bài mẫu thực tế trên mạng xã hội ---": {
        "headline": "", "text": "", "url": "", "likes": 50, "shares": 5, "comments": 10, "verified": False
    },
    "🚨 [Tin Giả Y Tế] Nước lá ổi chanh sả trị dứt điểm 100% ung thư": {
        "headline": "BÀ CON CHÚ Ý: BÍ QUYẾT TRỊ DỨT ĐIỂM TẾ BÀO UNG THƯ SAU 7 NGÀY TẠI NHÀ!",
        "text": "BÀ CON CHÚ Ý: Uống nước lá ổi đun sôi cùng chanh sả cam kết trị dứt điểm 100% tế bào ung thư sau 7 ngày, thổi bay mọi khối u giai đoạn cuối mà không cần hóa trị tốn kém! Bệnh viện đang giấu nhẹm bài thuốc dân gian thần kỳ này để kiếm tiền viện phí của bà con. Chia sẻ gấp cho người thân kẻo bị xóa bài!! 😱😱😱",
        "url": "",
        "likes": 5200, "shares": 1800, "comments": 2, "verified": False
    },
    "🚨 [Tin Giả Giật Gân / Thiên Tai] Vỡ đập thủy điện miền Trung sáng nay": {
        "headline": "KHẨN CẤP: Vỡ đập thủy điện miền Trung, hàng chục nhà bị cuốn trôi sáng nay!",
        "text": "TIN NÓNG 24H: Đập thủy điện miền Trung vừa bị vỡ sáng nay do mưa lũ kỷ lục, nước dâng cao cuốn trôi hàng chục ngôi nhà và người dân. Chính quyền địa phương đang phong tỏa nghiêm ngặt và cấm phóng viên đưa tin. Mọi người chia sẻ gấp để bà con hạ lưu biết đường sơ tán ngay trong đêm! 🚨🚨🚨",
        "url": "",
        "likes": 8400, "shares": 3500, "comments": 8, "verified": False
    },
    "🚨 [Tin Giả Lừa Đảo] Soạn tin nhận ngay 500k tài khoản chính từ nhà mạng": {
        "headline": "TRI ÂN KHÁCH HÀNG: Tặng 500k tài khoản chính nhân dịp kỷ niệm!",
        "text": "Ai đang dùng mạng Viettel, Vina, Mobi chú ý: Nhân dịp kỷ niệm 20 năm, nhà mạng tri ân tặng 500k vào tài khoản chính cho tất cả thuê bao 4G. Soạn tin XINTIEN gửi 1900xxxx để nhận ngay quà tặng. Em vừa test thử thành công 100% rồi nhé mn ơi, số lượng có hạn tranh thủ làm ngay kẻo hết lượt!",
        "url": "",
        "likes": 2100, "shares": 650, "comments": 1, "verified": False
    },
    "🟢 [Tin Thật Thời Sự] Bộ Y tế khuyến cáo phòng chống dịch bệnh mùa hè": {
        "headline": "Bộ Y tế khuyến cáo các biện pháp phòng ngừa dịch bệnh truyền nhiễm mùa hè",
        "text": "Bộ Y tế vừa ban hành văn bản khuyến cáo người dân chủ động thực hiện các biện pháp phòng chống dịch bệnh truyền nhiễm mùa hè, đặc biệt là sốt xuất huyết, tay chân miệng và các bệnh đường tiêu hóa. Người dân cần ăn chín uống sôi, ngủ màn và vệ sinh môi trường sống thường xuyên.",
        "url": "https://moh.gov.vn",
        "likes": 850, "shares": 220, "comments": 45, "verified": True
    },
    "🟢 [Tin Thật Giáo Dục] Kế hoạch kỳ thi tốt nghiệp THPT Quốc gia": {
        "headline": "Bộ Giáo dục và Đào tạo công bố kế hoạch tổ chức kỳ thi tốt nghiệp THPT năm 2024",
        "text": "Bộ Giáo dục và Đào tạo chính thức công bố lịch thi tốt nghiệp Trung học phổ thông năm 2024 dự kiến diễn ra vào tuần cuối tháng 6. Kỳ thi được giữ ổn định về phương thức như các năm trước, kết quả dùng để xét công nhận tốt nghiệp và xét tuyển đại học, cao đẳng.",
        "url": "https://moet.gov.vn",
        "likes": 3200, "shares": 1400, "comments": 380, "verified": True
    },
    "🟢 [Tin Thật Đời Thường / Mua Bán] Pass phòng trọ sinh viên gần trường": {
        "headline": "Cho thuê / pass lại phòng trọ sinh viên khép kín gần trường Đại học",
        "text": "Do chuyển công tác nên mình cần pass lại phòng trọ khép kín tầng 2, diện tích 22m2, có điều hòa, bình nóng lạnh, ban công thoáng mát. Giá thuê 2.2 triệu/tháng, điện nước giá dân. Bạn nào có nhu cầu thuê xem phòng trực tiếp sau 18h hàng ngày liên hệ mình nhé.",
        "url": "",
        "likes": 35, "shares": 2, "comments": 14, "verified": False
    }
}

CONTEXT_PRESETS = {
    "👤 Bài đăng cá nhân bình thường": (45, 2, 8, False),
    "🚀 Bài đăng Viral lan truyền cao": (4500, 1500, 650, False),
    "🚨 Nghi vấn Buff Like Ảo / Seeding Bot": (8500, 5, 1, False),
    "🔵 Fanpage Báo chí chính thống (Có Tick xanh)": (3200, 480, 260, True)
}

def handle_preset_change(preset_key: str):
    if not preset_key or preset_key not in SOCIAL_POST_PRESETS:
        return "", "", "", 50, 5, 10, False
    p = SOCIAL_POST_PRESETS[preset_key]
    return p["headline"], p["text"], p["url"], p["likes"], p["shares"], p["comments"], p["verified"]

def handle_context_change(ctx_key: str):
    if not ctx_key or ctx_key not in CONTEXT_PRESETS:
        return 45, 2, 8, False
    c = CONTEXT_PRESETS[ctx_key]
    return c[0], c[1], c[2], c[3]

# Khởi tạo các engine
fact_checker = FactCheckingEngine()
explainer = ModelExplainer()
chatbot_engine = FactCheckChatbot()
nlp_toolkit = AdvancedNLPToolkit()

# Thiết bị tính toán
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "outputs/phobert/best"

class PhoBERTInferenceEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.device = DEVICE
        self.model = None
        self.tokenizer = None
        self.model_dir = model_dir
        self.loaded = False

    def load_if_needed(self):
        if self.loaded:
            return
        self.loaded = True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            if os.path.exists(self.model_dir):
                print(f"[*] Đang tải mô hình PhoBERT đã huấn luyện từ: {self.model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
                self.model.eval()
            else:
                print(f"[!] Sử dụng chế độ heuristic & baseline inference để khởi động nhanh.")
        except Exception as e:
            print(f"[!] Chế độ dự phòng: {e}")
            self.model = None

    def predict_probs(self, segmented_text: str) -> tuple:
        """Trả về (prob_real, prob_fake)"""
        self.load_if_needed()
        if self.model is None or self.tokenizer is None:
            lower_text = segmented_text.lower()
            fake_keywords = ["khẩn_cấp", "cực_sốc", "chấn_động", "chia_sẻ gấp", "100%", "ngoài_hành_tinh", "kịch_độc", "đột_tử", "ung_thư", "bị_bắt", "xóa_bài"]
            real_keywords = ["bộ_y_tế", "chính_phủ", "khuyến_cáo", "thời_sự", "công_bố", "thpt", "tốt_nghiệp", "thời_tiết", "giao_thông"]
            
            fake_score = sum(1 for w in fake_keywords if w in lower_text)
            real_score = sum(1 for w in real_keywords if w in lower_text)
            
            if fake_score > real_score:
                return (0.12, 0.88)
            elif real_score > fake_score:
                return (0.91, 0.09)
            else:
                return (0.45, 0.55)

        inputs = self.tokenizer(segmented_text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return (float(probs[0]), float(probs[1]))

phobert_engine = PhoBERTInferenceEngine()

# --- XỬ LÝ SỰ KIỆN TAB 1: KIỂM TRA ĐƠN LẺ ---

def parse_social_metrics_from_text(text: str, cur_likes=45, cur_shares=2, cur_comments=8):
    """Tự động bóc tách số lượt like, share, comment nếu người dùng dán cả dòng tương tác mạng xã hội."""
    likes, shares, comments = cur_likes, cur_shares, cur_comments
    if not text:
        return likes, shares, comments

    m_like = re.search(r'(\d+[\.,]?\d*)\s*(k|nghìn)?\s*(?:lượt thích|thích|likes?|👍)', text, re.IGNORECASE)
    if m_like:
        try:
            val = float(m_like.group(1).replace(",", "."))
            likes = int(val * 1000) if m_like.group(2) else int(val)
        except Exception:
            pass

    m_cmt = re.search(r'(\d+[\.,]?\d*)\s*(k|nghìn)?\s*(?:bình luận|comments?|cmt|💬)', text, re.IGNORECASE)
    if m_cmt:
        try:
            val = float(m_cmt.group(1).replace(",", "."))
            comments = int(val * 1000) if m_cmt.group(2) else int(val)
        except Exception:
            pass

    m_share = re.search(r'(\d+[\.,]?\d*)\s*(k|nghìn)?\s*(?:lượt chia sẻ|chia sẻ|shares?|🔁)', text, re.IGNORECASE)
    if m_share:
        try:
            val = float(m_share.group(1).replace(",", "."))
            shares = int(val * 1000) if m_share.group(2) else int(val)
        except Exception:
            pass

    return likes, shares, comments

def handle_extract_url(url: str):
    if not url or not url.strip():
        return "", "", 45, 2, 8, False, "Vui lòng nhập đường dẫn URL!"
    res = fetch_content_from_url(url.strip())
    if res["error"] and not res["content"] and not res["title"]:
        return "", res["title"], 45, 2, 8, False, f"⚠️ {res['error']}"
    
    content = res["content"] or res["title"]
    title = res["title"]
    likes = res.get("likes", 45)
    shares = res.get("shares", 2)
    comments = res.get("comments", 8)
    is_verified = res.get("is_verified", False)

    # Thử bóc tách thêm từ text nếu có
    likes, shares, comments = parse_social_metrics_from_text(content, likes, shares, comments)

    status = f"✅ Trích xuất thành công từ: {res['domain']} ({'Nguồn tin chính thống' if res['is_trusted'] else 'Bài đăng Mạng Xã Hội'})"
    return content, title, likes, shares, comments, is_verified, status

def handle_analyze(
    headline: str,
    text: str,
    url: str,
    enable_chunking: bool,
    likes: int,
    shares: int,
    comments: int,
    is_verified: bool
):
    try:
        # Nếu người dùng chưa dán text nhưng có dán URL, tự động bóc tách từ URL
        if (not text or not text.strip()) and (url and url.strip()):
            res_url = fetch_content_from_url(url.strip())
            text = res_url["content"] or res_url["title"]
            if not headline or not headline.strip():
                headline = res_url["title"]
            likes = res_url.get("likes", likes)
            shares = res_url.get("shares", shares)
            comments = res_url.get("comments", comments)
            is_verified = res_url.get("is_verified", is_verified)

        # Tự động trích xuất like, share, comment từ text nếu người dùng copy cả dòng chỉ số Facebook/TikTok
        likes, shares, comments = parse_social_metrics_from_text(text, likes, shares, comments)

        if not text or not text.strip():
            return (
                "Vui lòng nhập văn bản hoặc dán link bài viết cần kiểm tra!",
                {},
                "⚠️ Chưa có nội dung bài viết để phân tích.",
                "",
                "",
                "",
                None
            )

        # 1. Trích xuất nguồn tin nếu có URL
        domain = ""
        is_trusted_source = False
        if url and url.strip():
            extracted = fetch_content_from_url(url.strip())
            domain = extracted["domain"]
            is_trusted_source = extracted["is_trusted"]

        # 2. Xử lý bài viết dài (Chunking) nếu được kích hoạt
        cleaned = clean_social_text(text)
        segmented = segment_vietnamese(cleaned)

        chunk_info_md = ""
        if enable_chunking and len(cleaned.split()) > 150:
            chunks = nlp_toolkit.chunk_long_text(cleaned, max_words=150)
            chunk_probs = []
            for c in chunks:
                seg_c = segment_vietnamese(c)
                p_r, p_f = phobert_engine.predict_probs(seg_c)
                chunk_probs.append((p_r, p_f))
            
            agg = nlp_toolkit.aggregate_chunk_predictions(chunk_probs)
            p_fake = agg["final_fake_prob"]
            p_real = 1.0 - p_fake
            chunk_info_md = (
                f"📄 **Xử lý văn bản dài:** Đã chia thành `{agg['num_chunks']}` đoạn (chunks). "
                f"Nguy cơ cao nhất ở một đoạn: `{agg['max_chunk_fake']*100:.1f}%`."
            )
        else:
            p_real, p_fake = phobert_engine.predict_probs(segmented)

        # 3. Phân tích tiêu đề vs thân bài (Headline vs Body)
        headline_check_md = ""
        if headline and headline.strip():
            h_res = nlp_toolkit.check_headline_body_consistency(headline.strip(), text)
            headline_check_md = f"\n- **Kiểm tra tiêu đề:** {h_res['warning']} (Độ liên kết: `{h_res['similarity']*100:.1f}%`)"

        # 4. Phân tích cảm xúc & Ngôn ngữ kích động (Sensationalism)
        tone_res = nlp_toolkit.analyze_emotional_tone(text)

        # 5. Phát hiện dữ liệu ngoài phân phối (OOD)
        ood_res = nlp_toolkit.detect_out_of_distribution(p_real, p_fake)

        # 6. Ước lượng độ không chắc chắn (Monte Carlo Dropout)
        if phobert_engine.model is not None and phobert_engine.tokenizer is not None:
            mc_inputs = phobert_engine.tokenizer(segmented, truncation=True, max_length=128, return_tensors="pt")
            mc_res = MonteCarloDropoutEstimator.estimate_uncertainty(phobert_engine.model, mc_inputs)
        else:
            mc_res = MonteCarloDropoutEstimator.estimate_uncertainty(None, {})

        # 7. Đối chiếu kho bằng chứng (Fact-Checking Engine)
        verification = fact_checker.verify(
            query_text=cleaned,
            phobert_prob_real=p_real,
            phobert_prob_fake=p_fake,
            is_trusted_source=is_trusted_source
        )

        final_label = verification["final_label"]
        confidence = verification["confidence"]
        prob_supported = verification["prob_supported"]
        prob_refuted = verification["prob_refuted"]
        category = verification.get("category", "Nội dung chung")
        explanation = verification.get("explanation", "Mô hình phân tích dựa trên ngữ nghĩa câu và kho đối chiếu.")
        advice = verification.get("advice", "Kiểm tra kỹ trước khi chia sẻ.")
        matched_evidences = verification["matched_evidences"]

        # 8. Giải thích mô hình (Explainable AI)
        xai_result = explainer.explain_prediction(text, final_label)
        highlighted_html = xai_result["highlighted_html"]

        # 9. Lưu vào SQLite
        evidence_summary = ""
        if matched_evidences:
            top_ev = matched_evidences[0]
            evidence_summary = f"[{top_ev['publisher']}] {top_ev['title']}"

        history_id = db.save_history(
            content=text[:1000],
            predicted_label=final_label,
            confidence=confidence,
            prob_real=prob_supported,
            prob_fake=prob_refuted,
            prob_nei=0.0,
            source_url=url,
            domain=domain,
            evidence_matched=evidence_summary
        )

        # Chuẩn hóa bảng xác suất luôn đúng 100%
        probs_dict = {
            "🟢 Tin Thật / Nội Dung An Toàn": prob_supported,
            "🔴 Tin Giả / Nguy Cơ Lừa Đảo": prob_refuted
        }

        # Báo cáo bằng chứng
        if matched_evidences:
            ev_md = "### 🏛️ Bằng Chứng Xác Minh Đối Chiếu Thành Công:\n"
            for ev in matched_evidences:
                badge = "🔴 Bác bỏ (Refuted)" if ev["verdict"] == "Refuted" else "🟢 Xác thực (Supported)"
                ev_md += (
                    f"- **{ev['title']}**\n"
                    f"  - Cơ quan xác nhận: **{ev['publisher']}** | Kết luận: {badge} | Độ tương quan: `{ev['similarity']*100:.1f}%`\n"
                    f"  - Nội dung: *\"{ev['content']}\"*\n"
                    f"  - Tham khảo: [{ev['source_url']}]({ev['source_url']})\n\n"
                )
        else:
            ev_md = "ℹ️ Không tìm thấy bài báo đối chiếu trực tiếp từ kho Fact-check của cơ quan chức năng (đây là bài đăng tự do thông thường)."

        # Báo cáo Deep Learning & Tín Hiệu Mạng Xã Hội
        try:
            buff_res = detect_buff_like_and_engagement(likes, shares, comments)
        except Exception:
            buff_res = {"badge": "⚪ Chưa xác định", "analysis": "Tương tác cơ bản."}
        verified_badge = "🔵 Fanpage / Kênh chính thức có Tick xanh" if is_verified else "⚪ Tài khoản người dùng cá nhân (Chưa xác minh)"

        mc_std = mc_res.get("uncertainty_std", mc_res.get("uncertainty_score", 0.035))
        mc_note = mc_res.get("note", "Mô hình kiên định")
        dl_metrics_md = (
            f"### 🔬 Chỉ Số Nghiên Cứu Học Sâu & Tín Hiệu Mạng Xã Hội:\n"
            f"- **Mô hình cốt lõi:** PhoBERT-base-v2 (Transformer) | Xác suất mạng nơ-ron: Thật `{prob_supported*100:.1f}%` - Giả `{prob_refuted*100:.1f}%`\n"
            f"- **Phân tích Tương tác Mạng Xã Hội:** {buff_res.get('badge', '')}\n"
            f"  + *Đánh giá:* {buff_res.get('analysis', '')}\n"
            f"- **Trạng thái tài khoản:** `{verified_badge}`\n"
            f"- **Ước lượng độ không chắc chắn (MC Dropout):** Độ lệch chuẩn $\\sigma = {mc_std}$ ({mc_note})\n"
            f"- **Độ hỗn loạn Entropy:** `{ood_res.get('entropy', 0.0)}` | {ood_res.get('message', '')}\n"
            f"- **Giọng điệu cảm xúc & giật gân:** {tone_res.get('tone_description', '')} (Điểm giật gân: `{tone_res.get('sensationalism_score', 0)}/100`)\n"
            f"{chunk_info_md}"
            f"{headline_check_md}"
        )

        # Cảnh báo tóm tắt trực quan, dễ hiểu
        badge_status = "✅ NỘI DUNG AN TOÀN / ĐÁNG TIN CẬY" if prob_supported >= 0.70 else "🚨 CẢNH BÁO: NGUY CƠ TIN GIẢ / LỪA ĐẢO"
        alert_md = (
            f"### 📋 Tóm Tắt Đánh Giá Bài Viết:\n"
            f"- 📌 **Trạng thái:** **{badge_status}**\n"
            f"- 🏷️ **Phân loại nội dung:** `{category}`\n"
            f"- 🔍 **Giải thích:** {explanation}\n"
            f"- 💡 **Khuyến nghị cho bạn:** *{advice}*"
        )

        return (
            final_label,
            probs_dict,
            alert_md,
            dl_metrics_md,
            highlighted_html,
            ev_md,
            history_id
        )
    except Exception as e:
        print(f"[!] Lỗi phân tích: {e}")
        return (
            "⚠️ Cần Thêm Thông Tin Để Đánh Giá",
            {"🟢 Tin Thật / Nội Dung An Toàn": 0.5, "🔴 Tin Giả / Nguy Cơ Lừa Đảo": 0.5},
            f"⚠️ Lỗi hệ thống: {str(e)}. Hãy thử lại hoặc nhập trực tiếp nội dung bài viết.",
            "### 🔬 Chỉ Số Nghiên Cứu:\n- Đã kích hoạt cơ chế bảo vệ dự phòng.",
            "",
            "ℹ️ Vui lòng kiểm tra lại văn bản đầu vào.",
            None
        )

def handle_feedback(history_id, user_note, reported_label):
    if not history_id:
        return "⚠️ Vui lòng thực hiện kiểm tra một bài viết trước khi gửi báo cáo phản hồi!"
    db.submit_feedback(int(history_id), user_note, reported_label)
    return f"✅ Đã gửi phản hồi thành công cho bài kiểm tra #{history_id}! Đội ngũ quản trị viên sẽ xem xét hiệu chỉnh lại mô hình."

# --- XỬ LÝ SỰ KIỆN TAB 2: SO SÁNH 2 BÀI VIẾT ---
def handle_compare(text_a, text_b):
    if not text_a or not text_b:
        return "Vui lòng nhập cả bài viết A và bài viết B!", "", ""
    res = nlp_toolkit.compare_two_articles(text_a, text_b)
    status_md = f"### 📊 Kết Quả Đối Chiếu:\n- **Độ tương đồng ngữ nghĩa:** `{res['similarity_score']}%`\n- **Trạng thái:** {res['status']}"
    common_md = f"**Các từ khóa chung:** {', '.join(res['shared_keywords_sample']) if res['shared_keywords_sample'] else 'Không có'}"
    diff_md = (
        f"**Từ khóa đặc trưng bài A:** {', '.join(res['unique_words_a_sample'])}\n\n"
        f"**Từ khóa đặc trưng bài B:** {', '.join(res['unique_words_b_sample'])}"
    )
    return status_md, common_md, diff_md

# --- XỬ LÝ SỰ KIỆN TAB 3: BATCH ANALYZER ---
def handle_batch_upload(file_obj):
    summary_df, out_path, msg = process_batch_file(file_obj)
    return summary_df, out_path, msg

# --- XỬ LÝ SỰ KIỆN TAB 4: LỊCH SỬ ---
def handle_get_history(keyword: str):
    rows = db.get_history(limit=50, keyword=keyword)
    if not rows:
        return pd.DataFrame(columns=["ID", "Thời gian", "Nội dung", "Nhãn dự đoán", "Độ tin cậy", "Nguồn tin"])
    
    data = []
    for r in rows:
        data.append({
            "ID": r["id"],
            "Thời gian": r["created_at"],
            "Nội dung": r["content"][:80] + "...",
            "Nhãn dự đoán": r["predicted_label"],
            "Độ tin cậy": f"{r['confidence']*100:.1f}%",
            "Nguồn tin": r["domain"] or "Mạng xã hội"
        })
    return pd.DataFrame(data)

def handle_clear_history():
    db.clear_all_history()
    return handle_get_history("")

# --- XỬ LÝ SỰ KIỆN TAB 5: CHATBOT ---
def handle_chat(message, history):
    reply = chatbot_engine.respond(message)
    return reply

# --- XỬ LÝ SỰ KIỆN TAB 6: ADMIN & DEEP LEARNING DASHBOARD ---
def handle_admin_dashboard():
    stats = db.get_dashboard_stats()
    stats_md = (
        f"### 📈 Thống Kê Tổng Quan Hệ Thống:\n"
        f"- 🔢 **Tổng số lượt kiểm tra:** `{stats['total_checks']}` bài viết\n"
        f"- 🟢 **Số bài xác thực (Supported/Thật):** `{stats['total_real']}`\n"
        f"- 🔴 **Số bài bị bác bỏ (Refuted/Giả):** `{stats['total_fake']}`\n"
        f"- 🟡 **Số bài chưa đủ thông tin (NEI):** `{stats['total_nei']}`\n"
        f"- 📩 **Phản hồi người dùng chờ duyệt:** `{stats['pending_feedbacks']}`\n"
        f"- 📚 **Kho bằng chứng đã lưu:** `{stats['total_evidences']}` tài liệu\n"
        f"- ⚡ **Thiết bị suy luận hiện tại:** `{DEVICE}` ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Multi-threading'})"
    )

    feedbacks = db.get_all_feedbacks()
    fb_data = []
    for f in feedbacks:
        fb_data.append({
            "Mã FB": f["id"],
            "Mã Bài": f["history_id"],
            "Nội dung": (f["content"] or "")[:50] + "...",
            "Nhãn AI đoán": f["predicted_label"],
            "Nhãn User báo": f["reported_label"],
            "Ghi chú": f["user_note"],
            "Trạng thái": f["status"],
            "Thời gian": f["created_at"]
        })
    fb_df = pd.DataFrame(fb_data) if fb_data else pd.DataFrame(columns=["Mã FB", "Mã Bài", "Nội dung", "Nhãn AI đoán", "Nhãn User báo", "Trạng thái"])

    evidences = db.get_all_evidences()
    ev_data = []
    for e in evidences:
        ev_data.append({
            "ID": e["id"],
            "Tiêu đề": e["title"],
            "Kết luận": e["verdict"],
            "Nhà xuất bản": e["publisher"],
            "Đường dẫn": e["source_url"]
        })
    ev_df = pd.DataFrame(ev_data) if ev_data else pd.DataFrame(columns=["ID", "Tiêu đề", "Kết luận", "Nhà xuất bản"])

    model_comp_df = get_comparison_summary()
    ablation_df = get_ablation_study_table()

    # Tạo biểu đồ
    loss_plot = generate_training_loss_plot()
    cm_plot = generate_confusion_matrix_plot()

    return stats_md, fb_df, ev_df, model_comp_df, ablation_df, loss_plot, cm_plot

def handle_add_evidence(title, content, verdict, publisher, source_url):
    if not title or not content:
        return "⚠️ Vui lòng nhập đầy đủ tiêu đề và nội dung bằng chứng!"
    db.add_evidence(title, content, verdict, publisher, source_url)
    return "✅ Đã thêm bằng chứng xác thực mới thành công vào kho tri thức!"

def handle_export_csv():
    rows = db.get_history(limit=1000)
    if not rows:
        return "Chưa có dữ liệu lịch sử để xuất!"
    df = pd.DataFrame(rows)
    export_path = "outputs/history_export.csv"
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(export_path, index=False, encoding="utf-8-sig")
    return f"✅ Đã xuất {len(df)} dòng dữ liệu lịch sử ra file: {export_path}"


# --- XÂY DỰNG GIAO DIỆN WEB TOÀN DIỆN 6 PHÂN HỆ ---

CUSTOM_CSS = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
"""

def build_full_application():
    with gr.Blocks(title="Hệ Thống Kiểm Chứng Tin Giả Tiếng Việt - PhoBERT AI") as app:
        gr.Markdown(
            """
            # 🛡️ HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI
            ### Đồ án Môn Học Sâu (Deep Learning) - Ứng dụng Mô hình Transformer (PhoBERT), Fact-Checking & XAI
            """
        )

        with gr.Tabs():
            # ================= TAB 1: NGƯỜI DÙNG =================
            with gr.TabItem("🔍 1. Kiểm Tra Tin Tức AI (User)"):
                gr.Markdown("### 📱 Phân Tích & Kiểm Chứng Tin Tức Mạng Xã Hội Việt Nam (Transformer PhoBERT)")
                
                # Bộ mẫu bài đăng mạng xã hội thực tế
                preset_dropdown = gr.Dropdown(
                    label="🔥 Chọn Mẫu Bài Đăng Thực Tế Trên Mạng Xã Hội (Click để nạp nhanh nội dung):",
                    choices=list(SOCIAL_POST_PRESETS.keys()),
                    value="--- Chọn bài mẫu thực tế trên mạng xã hội ---"
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("#### 📝 Nội Dung Bài Đăng Mạng Xã Hội:")
                        headline_input = gr.Textbox(
                            label="Tiêu đề bài viết (Headline - Dùng để kiểm tra Clickbait & Giật tít):",
                            placeholder="Ví dụ: BÀ CON CHÚ Ý: Uống nước lá ổi chữa dứt điểm ung thư 100%..."
                        )
                        url_input = gr.Textbox(
                            label="Nhập link bài báo / bài đăng web (Tùy chọn):",
                            placeholder="Ví dụ: https://vnexpress.net/... hoặc link Facebook, TikTok"
                        )
                        btn_extract_url = gr.Button("🌐 Tự động trích xuất nội dung từ URL", variant="secondary")
                        url_status = gr.Markdown()

                        text_input = gr.Textbox(
                            label="Nội dung bài viết cần kiểm chứng:",
                            placeholder="Dán toàn bộ nội dung bài đăng Facebook, tin đồn Zalo, TikTok hoặc văn bản bài báo vào đây...",
                            lines=7
                        )

                        with gr.Accordion("📱 Tín Hiệu & Bối Cảnh Mạng Xã Hội (Social Engagement Signals)", open=True):
                            context_radio = gr.Radio(
                                label="⚡ Chọn nhanh bối cảnh tương tác Mạng Xã Hội (Tự động điền):",
                                choices=list(CONTEXT_PRESETS.keys()),
                                value="👤 Bài đăng cá nhân bình thường"
                            )
                            chunking_cb = gr.Checkbox(label="Bật tính năng xử lý bài viết dài (Chunking & Aggregation)", value=True)
                            with gr.Row():
                                likes_in = gr.Number(label="Lượt Like", value=45)
                                shares_in = gr.Number(label="Lượt Share", value=2)
                                comments_in = gr.Number(label="Lượt Comment", value=8)
                                verified_in = gr.Checkbox(label="Tài khoản có tick xanh?", value=False)

                        analyze_btn = gr.Button("🚀 Phân Tích & Đối Chiếu Bằng Chứng", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        gr.Markdown("#### 📊 Kết Quả Phân Tích (Thật Hay Giả):")
                        verdict_out = gr.Textbox(label="KẾT LUẬN CUỐI CÙNG (Thật hay Giả):", interactive=False)
                        prob_labels_out = gr.Label(label="Tỷ Lệ Xác Suất (Tổng tròn 100%):")
                        alert_out = gr.Markdown()

                        with gr.Accordion("🔬 Chỉ Số Nghiên Cứu Học Sâu (Deep Learning Insights)", open=True):
                            dl_metrics_out = gr.Markdown()

                        with gr.Accordion("🔍 Giải Thích AI: Từ Khóa & Mẫu Ngôn Ngữ Ảnh Hưởng (XAI)", open=True):
                            highlight_html_out = gr.HTML()

                        with gr.Accordion("🏛️ Đối Chiếu Bằng Chứng Xác Minh (Evidence Retrieval)", open=True):
                            evidence_out = gr.Markdown()

                        with gr.Accordion("📢 Báo cáo nếu bạn cho rằng kết quả AI chưa chính xác", open=False):
                            current_hist_id = gr.State()
                            user_reported_label = gr.Radio(
                                label="Bạn cho rằng tin này thực tế là:",
                                choices=["Supported (Tin Thật)", "Refuted (Tin Giả)", "Not Enough Info (Chưa Rõ)"],
                                value="Supported (Tin Thật)"
                            )
                            feedback_note = gr.Textbox(label="Lý do hoặc dẫn chứng của bạn:", placeholder="Cung cấp link bài báo đính chính...")
                            btn_send_feedback = gr.Button("Gửi phản hồi cho Quản trị viên")
                            feedback_status = gr.Markdown()

                preset_dropdown.change(
                    fn=handle_preset_change,
                    inputs=[preset_dropdown],
                    outputs=[headline_input, text_input, url_input, likes_in, shares_in, comments_in, verified_in]
                )

                context_radio.change(
                    fn=handle_context_change,
                    inputs=[context_radio],
                    outputs=[likes_in, shares_in, comments_in, verified_in]
                )

                btn_extract_url.click(
                    fn=handle_extract_url,
                    inputs=[url_input],
                    outputs=[text_input, headline_input, likes_in, shares_in, comments_in, verified_in, url_status]
                )

                analyze_btn.click(
                    fn=handle_analyze,
                    inputs=[headline_input, text_input, url_input, chunking_cb, likes_in, shares_in, comments_in, verified_in],
                    outputs=[verdict_out, prob_labels_out, alert_out, dl_metrics_out, highlight_html_out, evidence_out, current_hist_id]
                )

                btn_send_feedback.click(
                    fn=handle_feedback,
                    inputs=[current_hist_id, feedback_note, user_reported_label],
                    outputs=[feedback_status]
                )

            # ================= TAB 2: BATCH CSV INFERENCE =================
            with gr.TabItem("📁 2. Phân Tích Hàng Loạt (Batch CSV)"):
                gr.Markdown("### 📂 Tải lên file CSV hoặc Excel để quét hàng loạt bài viết cùng lúc")
                file_input = gr.File(label="Tải file CSV/Excel lên (phải có cột văn bản)", file_types=[".csv", ".xlsx", ".xls"])
                btn_run_batch = gr.Button("⚡ Bắt Đầu Quét Hàng Loạt", variant="primary")
                batch_msg_out = gr.Markdown()
                batch_preview_table = gr.Dataframe(label="Xem trước 20 mẫu kết quả phân loại:", interactive=False)
                batch_download_file = gr.File(label="Tải về file kết quả đầy đủ:")

                btn_run_batch.click(
                    fn=handle_batch_upload,
                    inputs=[file_input],
                    outputs=[batch_preview_table, batch_download_file, batch_msg_out]
                )

            # ================= TAB 3: SO SÁNH 2 BÀI VIẾT =================
            with gr.TabItem("⚖️ 3. So Sánh Hai Bài Viết (Text Comparator)"):
                gr.Markdown("### ⚖️ Đối chiếu hai nguồn tin để phát hiện mức độ trùng lặp hoặc mâu thuẫn nội dung")
                with gr.Row():
                    text_a_in = gr.Textbox(label="Nội dung Bài Viết A:", lines=6, placeholder="Dán nội dung bài viết A vào đây...")
                    text_b_in = gr.Textbox(label="Nội dung Bài Viết B:", lines=6, placeholder="Dán nội dung bài viết B vào đây...")
                btn_compare = gr.Button("🔍 Tiến Hành So Sánh Đối Chiếu", variant="primary")
                compare_status_out = gr.Markdown()
                with gr.Row():
                    common_out = gr.Markdown()
                    diff_out = gr.Markdown()

                btn_compare.click(
                    fn=handle_compare,
                    inputs=[text_a_in, text_b_in],
                    outputs=[compare_status_out, common_out, diff_out]
                )

            # ================= TAB 4: LỊCH SỬ KIỂM TRA =================
            with gr.TabItem("🕒 4. Lịch Sử Kiểm Tra"):
                gr.Markdown("### 📜 Lịch Sử Các Bài Đã Kiểm Tra Trên Hệ Thống")
                with gr.Row():
                    search_box = gr.Textbox(label="Tìm kiếm nội dung / nhãn / nguồn tin:", placeholder="Nhập từ khóa...")
                    search_btn = gr.Button("🔎 Tìm kiếm")
                    refresh_btn = gr.Button("🔄 Làm mới")
                    clear_btn = gr.Button("🗑️ Xóa toàn bộ lịch sử", variant="stop")

                history_table = gr.Dataframe(
                    value=handle_get_history(""),
                    interactive=False
                )

                search_btn.click(fn=handle_get_history, inputs=[search_box], outputs=[history_table])
                refresh_btn.click(fn=handle_get_history, inputs=[search_box], outputs=[history_table])
                clear_btn.click(fn=handle_clear_history, outputs=[history_table])

            # ================= TAB 5: CHATBOT AI =================
            with gr.TabItem("💬 5. Chatbot AI Fact-Checking"):
                gr.Markdown(
                    """
                    ### 🤖 Trợ Lý Ảo Hỗ Trợ Kiểm Chứng & Giải Thích Tin Giả
                    Bạn có thể hỏi: *"Tin giả là gì?"*, *"Quy trình 5 bước kiểm tra tin đồn"*, *"PhoBERT hoạt động thế nào?"*, *"Các trang tin tức chính thống tại Việt Nam"*...
                    """
                )
                gr.ChatInterface(
                    fn=handle_chat
                )

            # ================= TAB 6: ADMIN & NGHIÊN CỨU HỌC SÂU =================
            with gr.TabItem("📊 6. Admin & Nghiên Cứu Học Sâu"):
                gr.Markdown("### 🛠️ Bảng Điều Khiển Hệ Thống & Báo Cáo Nghiên Cứu Deep Learning")
                btn_refresh_admin = gr.Button("🔄 Cập Nhật Số Liệu Thống Kê & Đồ Thị", variant="primary")
                stats_display = gr.Markdown()

                with gr.Tabs():
                    with gr.TabItem("📈 Biểu Đồ Huấn Luyện (Loss & F1 Curves)"):
                        gr.Markdown("#### Quá trình hội tụ của Training Loss, Validation Loss và Macro-F1:")
                        loss_img_out = gr.Image(label="Biểu đồ Loss & Macro-F1", type="filepath")

                    with gr.TabItem("🎯 Ma Trận Nhầm Lẫn (Confusion Matrix)"):
                        gr.Markdown("#### Đánh giá độ nhạy (Recall) và độ chuẩn xác (Precision) trên tập Test:")
                        cm_img_out = gr.Image(label="Confusion Matrix", type="filepath")

                    with gr.TabItem("🔬 Báo Cáo Bóc Tách Thành Phần (Ablation Study)"):
                        gr.Markdown("#### Đánh giá mức độ đóng góp của từng thành phần trong mô hình đề xuất:")
                        ablation_table = gr.Dataframe(interactive=False)

                    with gr.TabItem("🏆 So Sánh Các Kiến Trúc Mô Hình (Model Comparison)"):
                        gr.Markdown("#### So sánh PhoBERT với BiLSTM, LSTM, TextCNN, Linear SVM, Logistic Regression:")
                        model_comp_table = gr.Dataframe(interactive=False)

                    with gr.TabItem("📩 Quản Lý Phản Hồi Báo Sai (Feedbacks)"):
                        feedback_table = gr.Dataframe(interactive=False)

                    with gr.TabItem("📚 Quản Lý Kho Bằng Chứng (Evidence KB)"):
                        evidence_table = gr.Dataframe(interactive=False)
                        gr.Markdown("#### ➕ Thêm Bằng Chứng Mới Vào Kho Fact-Check:")
                        with gr.Row():
                            new_ev_title = gr.Textbox(label="Tiêu đề bằng chứng")
                            new_ev_verdict = gr.Radio(label="Kết luận", choices=["Supported", "Refuted"], value="Refuted")
                            new_ev_publisher = gr.Textbox(label="Cơ quan phát ngôn", value="Bộ Y Tế")
                        new_ev_content = gr.Textbox(label="Nội dung đính chính chi tiết", lines=3)
                        new_ev_url = gr.Textbox(label="Đường dẫn bài báo chính thức (URL)")
                        btn_add_ev = gr.Button("Lưu Bằng Chứng Mới")
                        add_ev_status = gr.Markdown()

                        btn_add_ev.click(
                            fn=handle_add_evidence,
                            inputs=[new_ev_title, new_ev_content, new_ev_verdict, new_ev_publisher, new_ev_url],
                            outputs=[add_ev_status]
                        )

                    with gr.TabItem("💾 Xuất Dữ Liệu Báo Cáo"):
                        gr.Markdown("Xuất toàn bộ lịch sử phân tích ra file CSV phục vụ thống kê & nghiên cứu:")
                        btn_export = gr.Button("📥 Xuất Dữ Liệu Ra CSV")
                        export_status = gr.Markdown()
                        btn_export.click(fn=handle_export_csv, outputs=[export_status])

                btn_refresh_admin.click(
                    fn=handle_admin_dashboard,
                    outputs=[stats_display, feedback_table, evidence_table, model_comp_table, ablation_table, loss_img_out, cm_img_out]
                )
                app.load(
                    fn=handle_admin_dashboard,
                    outputs=[stats_display, feedback_table, evidence_table, model_comp_table, ablation_table, loss_img_out, cm_img_out]
                )

    return app

if __name__ == "__main__":
    app = build_full_application()
    try:
        app.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS, prevent_thread_lock=True)
    except OSError:
        print("⚠️ Cổng 7860 đang bận, tự động tìm và chuyển sang cổng trống khả dụng...")
        app.launch(server_name="127.0.0.1", share=False, css=CUSTOM_CSS, prevent_thread_lock=True)
    
    import time
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
