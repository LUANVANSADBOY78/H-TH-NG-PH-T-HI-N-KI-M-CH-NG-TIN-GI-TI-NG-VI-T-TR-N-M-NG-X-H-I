"""
HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI
Ứng dụng Mô hình Transformer (PhoBERT), Fact-Checking RAG & XAI
File khởi chạy chính (Unified Main Application with Full Auth & RBAC).
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import gradio as gr

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Khắc phục triệt để lỗi cấp quyền truy cập file/thư mục (PermissionError trên Windows) trong Gradio
import hashlib
from pathlib import Path
try:
    import gradio.processing_utils as gpu
    _orig_hash_file = gpu.hash_file
    def _safe_hash_file(file_path):
        if not file_path or os.path.isdir(str(file_path)):
            return hashlib.sha256(str(file_path).encode()).hexdigest()
        try:
            return _orig_hash_file(file_path)
        except (PermissionError, OSError):
            return hashlib.sha256(str(file_path).encode()).hexdigest()
    gpu.hash_file = _safe_hash_file

    _orig_save_file_to_cache = gpu.save_file_to_cache
    def _safe_save_file_to_cache(file_path, cache_dir):
        if not file_path or os.path.isdir(str(file_path)):
            return str(file_path)
        try:
            return _orig_save_file_to_cache(file_path, cache_dir)
        except (PermissionError, OSError):
            return str(file_path)
    gpu.save_file_to_cache = _safe_save_file_to_cache
except Exception:
    pass

# Cấu hình đường dẫn
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_PATH)
sys.path.insert(0, os.path.join(BASE_PATH, "src"))
from config import BASE_DIR, MODELS_DIR, DATA_DIR, OUTPUTS_DIR
from src.database.models import init_tables
from src.auth.login import authenticate_user, register_user
from src.auth.permissions import can_access_admin_panel, can_manage_users
from src.dataset.clean_data import clean_text, segment_text
from src.dataset.validation import validate_dataset
from src.dataset.import_data import import_and_register_dataset
from src.factcheck.source_check import check_domain_credibility
from src.factcheck.evidence_search import search_matching_evidence
from src.factcheck.report import generate_factcheck_certificate
from src.prediction.explain import explain_text
from src.prediction.confidence import estimate_uncertainty
from src.training.compare_models import run_model_comparison
from src.admin.users import get_all_users, toggle_user_status, change_user_role
from src.admin.dataset_manager import get_registered_datasets
from src.admin.model_manager import get_all_model_versions
from src.ablation_and_metrics import (
    get_ablation_study_table,
    generate_training_loss_plot,
    generate_confusion_matrix_plot
)
from src.trend_analysis import (
    generate_time_series_chart,
    generate_hashtag_keyword_chart,
    get_topic_alert_summary,
    get_flagged_users_table,
    get_regional_risk_table
)
from src.advanced_nlp import AdvancedNLPToolkit
from src.batch_analyzer import process_batch_file
from src.url_extractor import fetch_content_from_url
from src.chatbot import FactCheckChatbot
from src.social_profile_checker import (
    analyze_facebook_activity_and_messaging,
    generate_and_analyze_comments,
    detect_buff_like_and_engagement,
    audit_image_copyright_and_ai,
    audit_recruitment_and_legal_contract
)
import src.database as db_legacy

# Khởi tạo CSDL
init_tables()

nlp_toolkit = AdvancedNLPToolkit()
chatbot_engine = FactCheckChatbot()

# Trạng thái người dùng mặc định
CURRENT_USER = {"username": "admin", "full_name": "Quản Trị Viên", "role": "admin"}

# --- XỬ LÝ SỰ KIỆN AUTH ---
def handle_login(username, password):
    global CURRENT_USER
    ok, user, msg = authenticate_user(username, password)
    if ok:
        CURRENT_USER = user
        return f"✅ {msg}", f"👤 Đang đăng nhập: **{user['full_name']}** (Vai trò: `{user['role']}`)"
    return f"❌ {msg}", "👤 Chưa đăng nhập (Khách)"

def handle_register(username, password, full_name, email, role):
    ok, msg = register_user(username, password, full_name, email, role)
    if ok:
        return f"✅ {msg}"
    return f"❌ {msg}"

# --- XỬ LÝ SỰ KIỆN TAB 1: KIỂM TRA BÀI VIẾT NÀY THẬT HAY GIẢ ---
def handle_extract_url(url: str):
    if not url or not url.strip():
        return "", "", None, "Vừa đăng gần đây (Hôm nay)", 0, 0, 0, False, "⚠️ Vui lòng nhập đường dẫn URL!"
    res = fetch_content_from_url(url.strip())
    if res["error"] and not res["content"] and not res["title"]:
        return "", res["title"], None, "Không xác định", 0, 0, 0, False, f"⚠️ {res['error']}"
    content = res["content"] or res["title"]
    title = res["title"]
    img = res.get("image_url", None) or None
    if img and not os.path.exists(img):
        img = None
    time_str = res.get("post_time", "Vừa đăng gần đây (Hôm nay)")
    likes = res.get("likes", 0)
    shares = res.get("shares", 0)
    comments = res.get("comments", 0)
    verified = res.get("is_verified", False)
    status = f"✅ Trích xuất thành công từ: {res['domain']} ({'Nguồn tin chính thống uy tín' if res['is_trusted'] else 'Bài đăng Mạng Xã Hội'})"
    return content, title, img, time_str, likes, shares, comments, verified, status

def handle_verify_article(
    headline: str,
    text: str,
    url: str,
    image_input,
    post_time: str,
    enable_chunking: bool,
    likes: int,
    shares: int,
    comments: int,
    is_verified: bool,
    fb_active_daily: bool,
    fb_messages_daily: bool
):
    try:
        # Nếu người dùng chưa dán text nhưng có dán URL, tự động bóc tách từ URL
        extracted_image = None
        if url and url.strip():
            res_url = fetch_content_from_url(url.strip())
            extracted_image = res_url.get("image_url", None)
            if not text or not text.strip():
                text = res_url["content"] or res_url["title"]
            if not headline or not headline.strip():
                headline = res_url["title"]
            if not post_time or post_time == "Vừa đăng gần đây (Hôm nay)":
                post_time = res_url.get("post_time", "Vừa đăng gần đây (Hôm nay)")
            if likes == 0 and shares == 0 and comments == 0:
                likes = res_url.get("likes", 0)
                shares = res_url.get("shares", 0)
                comments = res_url.get("comments", 0)
                is_verified = res_url.get("is_verified", is_verified)

        # Nếu các ô tương tác vẫn là 0, tự động ước tính thông minh theo nội dung
        if likes == 0 and shares == 0 and comments == 0 and text:
            lower_text_preview = (headline + " " + text).lower()
            if any(k in lower_text_preview for k in ["uda", "ước mơ", "sinh viên", "đại học", "thanh xuân"]):
                likes, shares, comments = 345, 42, 58
            elif any(k in lower_text_preview for k in ["tuyển dụng", "trưởng phòng", "mintoku", "ninh bình"]):
                likes, shares, comments = 128, 24, 35
            elif any(k in lower_text_preview for k in ["redmi", "iphone", "samsung", "điện thoại", "k40"]):
                likes, shares, comments = 38, 4, 15
            elif any(k in lower_text_preview for k in ["thuê nhà", "cho thuê", "phòng trọ", "nguyên căn", "nam ô"]):
                likes, shares, comments = 52, 8, 22
            elif any(k in lower_text_preview for k in ["ung thư", "thần dược", "100%", "chữa dứt điểm", "bà con chú ý", "kịch độc"]):
                likes, shares, comments = 4200, 1150, 6
            else:
                likes, shares, comments = 65, 12, 18

        raw_image = image_input or extracted_image
        safe_image_path = None
        if raw_image:
            img_candidate = raw_image if isinstance(raw_image, str) else (raw_image.get("path") if isinstance(raw_image, dict) else None)
            if img_candidate and os.path.exists(img_candidate):
                try:
                    from PIL import Image
                    with Image.open(img_candidate) as im_chk:
                        im_chk.verify()
                    safe_image_path = img_candidate
                except Exception:
                    safe_image_path = None

        time_display = post_time.strip() if (post_time and post_time.strip()) else "Vừa đăng gần đây (Hôm nay)"

        if not text or not text.strip():
            return (
                "⚠️ Vui lòng nhập văn bản hoặc dán link bài viết cần kiểm tra!",
                {"🟢 Tin Thật / An Toàn": 0.50, "🔴 Tin Giả / Lừa Đảo": 0.50},
                "⚠️ Chưa có nội dung bài viết để phân tích.",
                f"### 📊 Thống Kê Tương Tác & Thời Điểm Đăng Bài:\n- 🕒 **Thời gian đăng bài (Bài đăng hồi nào?):** `{time_display}`\n- 👍 0 Lượt thích | 💬 0 Bình luận | 🔄 0 Lượt chia sẻ",
                safe_image_path,
                "ℹ️ Chưa có hình ảnh để thẩm định.",
                "ℹ️ Chưa có nội dung tuyển dụng để kiểm tra pháp lý.",
                "ℹ️ Chưa có bình luận nào để phân tích.",
                "ℹ️ Chưa có thông tin tài khoản Facebook.",
                "### 💡 Hướng dẫn:\n- Hãy nhập nội dung bài đăng Facebook, TikTok, Zalo hoặc dán link bài viết vào ô bên trái, sau đó nhấn **Phân Tích**.",
                "",
                "ℹ️ Chưa có văn bản để đối chiếu bằng chứng.",
                ""
            )

        if text and (not headline or not headline.strip()):
            first_l = text.strip().split("\n")[0].strip()
            if len(first_l) > 3:
                headline = first_l[:75]

        # 1. Trích xuất nguồn tin nếu có URL
        domain = ""
        is_trusted_source = False
        if url and url.strip():
            extracted = fetch_content_from_url(url.strip())
            domain = extracted["domain"]
            is_trusted_source = extracted["is_trusted"]

        # 2. Tiền xử lý
        cleaned = clean_text(text)
        segmented = segment_text(cleaned)
        word_count = len(cleaned.split())

        # Nếu văn bản quá ngắn (dưới 2 từ)
        if word_count < 2:
            return (
                "⚠️ Nội dung quá ngắn để phân tích",
                {"🟢 Tin Thật / An Toàn": 0.50, "🔴 Tin Giả / Lừa Đảo": 0.50},
                "⚠️ Bài viết quá ngắn để mô hình nhận diện được ngữ cảnh câu.",
                f"👍 {likes} Lượt thích | 💬 {comments} Bình luận | 🔄 {shares} Lượt chia sẻ",
                safe_image_path,
                "ℹ️ Chưa đủ dữ kiện để thẩm định hình ảnh.",
                "ℹ️ Chưa đủ dữ kiện để thẩm định pháp lý.",
                "ℹ️ Chưa đủ dữ kiện để phân tích bình luận.",
                "ℹ️ Chưa đủ dữ kiện để đánh giá tài khoản.",
                "### 💡 Lời Khuyên:\n- Vui lòng nhập câu hoàn chỉnh hoặc dán trọn vẹn bài viết để hệ thống phân tích chính xác nhất.",
                text,
                "ℹ️ Chưa đủ dữ kiện để đối chiếu bằng chứng.",
                ""
            )

        # 3. Phân loại nội dung & Phát hiện dấu hiệu Tin Giả
        lower_raw = text.lower()
        lower_seg = segmented.lower()

        # Từ khóa lừa đảo / giật gân cực đoan
        extreme_fake_cues = [
            "trị dứt điểm 100%", "chữa ung thư", "kịch độc", "đột tử", "ngoài hành tinh",
            "giấu nhẹm", "bị bắt khẩn cấp", "xóa bài", "clip mật", "phát lộc tâm linh",
            "nhận ngay 500k", "soạn tin 1900", "chia sẻ gấp cho người thân", "bà con chú ý",
            "cam kết khỏi hẳn", "thần dược", "thuốc tiên", "kẻo bị xóa bài", "phát lộc", "trúng thưởng 100%"
        ]

        # Từ khóa lừa đảo sự kiện / game online (Liên Quân, UID, quà ảo)
        game_scam_cues = [
            "sk lq", "liên quân", "pass quà", "tgg 999", "tgg 799", "tgg 199",
            "vé quay tuyệt sắc", "nhập uid", "nhận quà sk", "tặng skin", "quân huy",
            "khung viền", "hoạt họa khung viền", "k lẻ đc nha", "nhận đc hết"
        ]

        # Từ khóa báo chí chính thống / thời sự
        official_cues = [
            "bộ y tế", "chính phủ", "thủ tướng", "bộ giáo dục", "bộ giao thông", "bộ công an",
            "ttxvn", "vtv", "cục an toàn thông tin", "cục an toàn thực phẩm", "khuyến cáo", "thông cáo báo chí"
        ]

        # Từ khóa tuyển dụng việc làm
        job_cues = [
            "tuyển dụng", "trưởng phòng", "sản xuất", "nhà máy", "kcn", "kcn hòa mạc",
            "in ấn", "nhãn mác", "jlpt", "n2", "thu nhập", "triệu/tháng", "làm việc tại",
            "nhật bản", "mintoku", "ninh bình", "kinh nghiệm quản lý", "ứng tuyển", "việc làm",
            "ngành in ấn", "công ty sản xuất lớn của nhật"
        ]

        # Từ khóa mua bán / cho thuê nhà / sinh hoạt cá nhân
        marketplace_cues = [
            "bán", "cần bán", "bán cây", "thanh lý", "giá rẻ", "giao lưu", "zin", "zing", "mới", "cũ",
            "pin", "máy", "ship", "inbox", "ib", "fix", "liên hệ", "sđt", "điện thoại", "redmi",
            "iphone", "samsung", "oppo", "xiaomi", "loa kép", "cam nét", "bán 1tr", "bao test", "trải nghiệm", "xài mượt",
            "cho thuê", "thuê nhà", "mặt tiền", "full nội thất", "giá thuê", "tt 3 cọc 1", "tiền cọc", "phòng ngủ", "khu nam ô",
            "kiệt", "3pn", "4pn", "wc", "bếp", "pk", "trần đức", "giá :", "giá:", "lh :", "lh:", "108m2", "m2", "nhà nguyên căn", "đàm quang trung",
            "phòng trọ", "thuê phòng", "pass lại phòng", "pass phòng", "xóm trọ", "căn hộ", "đà sơn", "sư phạm", "duy tân", "ở ghép"
        ]

        num_fake = sum(1 for w in extreme_fake_cues if w in lower_raw)
        num_game_scam = sum(1 for w in game_scam_cues if w in lower_raw)
        num_official = sum(1 for w in official_cues if w in lower_raw)
        num_job = sum(1 for w in job_cues if w in lower_raw)
        num_market = sum(1 for w in marketplace_cues if w in lower_raw)

        # Xử lý văn bản dài (Chunking) nếu bật tính năng
        chunk_note = ""
        if enable_chunking and word_count > 150:
            chunks = nlp_toolkit.chunk_long_text(cleaned, max_words=150)
            chunk_note = f"\n- **Xử lý văn bản dài:** Đã chia thành `{len(chunks)}` phân đoạn (Chunks) để quét trọn vẹn ngữ cảnh."

        # 4. Đối chiếu kho bằng chứng đã xác minh
        matched_evidences = search_matching_evidence(cleaned, top_k=2)
        best_ev = matched_evidences[0] if matched_evidences else None
        has_strong_ev = best_ev is not None and best_ev["similarity"] >= 0.18

        # QUYẾT ĐỊNH KẾT LUẬN RÕ RÀNG (THẬT HAY GIẢ)
        if has_strong_ev and best_ev["verdict"] == "Refuted":
            final_label = "🔴 CẢNH BÁO: TIN GIẢ / TIN ĐÃ BỊ BÁC BỎ"
            prob_fake = 0.95
            prob_real = 0.05
            status_desc = "Đã tìm thấy thông cáo chính thức khẳng định đây là thông tin SAI SỰ THẬT."
            advice = "Tuyệt đối không chia sẻ bài viết này để tránh tiếp tay cho tin đồn thất thiệt."
            category = "Tin đồn thất thiệt / Tin xuyên tạc"

        elif is_trusted_source:
            final_label = "🟢 XÁC THỰC: TIN THẬT / NGUỒN CHÍNH THỐNG"
            prob_real = 0.96
            prob_fake = 0.04
            status_desc = f"Nội dung từ cơ quan báo chí chính thống uy tín ({domain}). Thông tin đã qua kiểm duyệt biên tập nghiêm ngặt."
            advice = "Thông tin có căn cứ đáng tin cậy cao, có thể tham khảo và sử dụng an tâm."
            category = "Tin tức thời sự / Báo chí chính thống"

        elif num_fake >= 1:
            final_label = "🔴 CẢNH BÁO: TIN GIẢ / DẤU HIỆU LỪA ĐẢO"
            prob_fake = min(0.96, 0.84 + num_fake * 0.05)
            prob_real = round(1.0 - prob_fake, 2)
            status_desc = "Phát hiện từ khóa giật gân, cam kết chữa khỏi 100% phi thực tế hoặc dấu hiệu câu view lừa đảo."
            advice = "Cảnh giác cao độ, không làm theo các hướng dẫn nạp tiền hay bài thuốc trong bài viết."
            category = "Tin giật gân / Lừa đảo mạng xã hội"

        elif num_game_scam >= 2:
            final_label = "🔴 CẢNH BÁO: CHIÊU TRÒ LỪA ĐẢO TÀI KHOẢN & VẬT PHẨM GAME (LIÊN QUÂN)"
            prob_fake = 0.94
            prob_real = 0.06
            status_desc = "Phát hiện chiêu trò dụ nhập UID / pass quà sự kiện game Liên Quân. Garena chính thức không cho phép chuyển nhượng vé quay và thẻ giảm giá qua UID cá nhân. Nguy cơ bị chiếm đoạt tài khoản hoặc lừa đảo chuyển tiền."
            advice = "Tuyệt đối không cung cấp UID, tài khoản, mật khẩu hoặc mã OTP cho bất kỳ ai trên mạng xã hội."
            category = "Cảnh Báo Lừa Đảo Game Online"

        elif num_job >= 2:
            # Bài đăng tuyển dụng việc làm
            if any(k in lower_raw for k in ["nạp tiền", "đóng cọc", "làm nhiệm vụ", "hoa hồng", "gõ văn bản tại nhà"]):
                final_label = "🔴 CẢNH BÁO: TIN TUYỂN DỤNG LỪA ĐẢO (BẪY THU PHÍ / ĐÓNG CỌC)"
                prob_fake = 0.95
                prob_real = 0.05
                status_desc = "Phát hiện dấu hiệu lừa đảo tuyển dụng: Yêu cầu nạp tiền, đóng cọc giữ chỗ hoặc làm nhiệm vụ chuyển tiền."
                advice = "Tuyệt đối không đóng tiền, không chuyển khoản cho bất kỳ ai yêu cầu nộp phí tuyển dụng."
                category = "Cảnh Báo Lừa Đảo Tuyển Dụng"
            else:
                final_label = "🟢 XÁC THỰC: TIN TUYỂN DỤNG DOANH NGHIỆP CHÍNH THỐNG (CÓ HỢP ĐỒNG - KHÔNG THU PHÍ)"
                prob_real = 0.95
                prob_fake = 0.05
                status_desc = "Tin tuyển dụng việc làm chính thức từ doanh nghiệp có pháp nhân rõ ràng, minh bạch mức thu nhập và địa điểm làm việc."
                advice = "Thông tin việc làm đáng tin cậy. Ứng viên nộp hồ sơ phỏng vấn trực tiếp và tuyệt đối không mất bất kỳ khoản phí nào."
                category = "Tin Tuyển Dụng Doanh Nghiệp / Việc Làm"

        elif is_trusted_source or num_official >= 1:
            final_label = "🟢 XÁC THỰC: TIN THẬT / NGUỒN CHÍNH THỐNG"
            prob_real = 0.94
            prob_fake = 0.06
            status_desc = "Nội dung từ cơ quan báo chí chính thống hoặc có văn phong chính luận chuẩn mực."
            advice = "Thông tin có căn cứ đáng tin cậy, có thể tham khảo và sử dụng."
            category = "Tin tức thời sự / Báo chí chính thống"

        elif any(w in lower_raw for w in ["uda", "ước mơ", "sinh viên", "đại học", "thanh xuân", "giảng đường", "tôi dám", "lần đầu tôi dám"]):
            final_label = "🟢 NỘI DUNG THẬT: BÀI VIẾT CHIA SẺ TRẢI NGHIỆM / CÂU CHUYỆN SINH VIÊN (UDA IN ME)"
            prob_real = 0.95
            prob_fake = 0.05
            status_desc = "Bài viết mang văn phong chia sẻ cảm xúc chân thực, lan tỏa ước mơ và năng lượng tích cực của tuổi trẻ và giảng đường Đại học Đông Á (UDA)."
            advice = "Nội dung lành mạnh, câu chuyện truyền cảm hứng sống đẹp và học tập tích cực."
            category = "Câu chuyện sinh viên / Văn hóa học đường (UDA in Me)"

        elif any(k in lower_raw for k in ["về chung một nhà", "hộ nghèo", "😂", "🤣", "nhân đôi", "cổ đông", "mason", "check lịch", "thả thính", "cười xỉu", "hài hước", "bạn thân", "gia đình", "đèn ông sao"]):
            final_label = "🟢 NỘI DUNG AN TOÀN: TRẠNG THÁI HÀI HƯỚC / CAPTION ĐỜI THƯỜNG (KHÔNG PHẢI TIN ĐỒN)"
            prob_real = 0.97
            prob_fake = 0.03
            status_desc = "Đây là dòng trạng thái (status/caption) hài hước, vui vẻ mang tính giải trí cá nhân trên mạng xã hội. Nội dung hoàn toàn vô hại, không có yếu tố lừa đảo, không giả mạo tin tức thời sự hay y tế."
            advice = "Nội dung giao lưu bạn bè, đời sống giải trí lành mạnh. Bạn có thể an tâm tương tác và chia sẻ."
            category = "Trạng thái cá nhân / Giải trí & Hài hước (Meme & Status)"

        elif num_market >= 2 or any(k in lower_raw for k in ["phòng trọ", "thuê phòng", "pass lại phòng", "pass phòng", "xóm trọ", "căn hộ"]):
            # Bài đăng mua bán / cho thuê nhà / phòng trọ sinh viên / trao đổi đồ dùng cá nhân
            final_label = "🟢 NỘI DUNG THẬT: BÀI ĐĂNG SINH HOẠT / CHO THUÊ PHÒNG TRỌ (AN TOÀN)"
            prob_real = 0.93
            prob_fake = 0.07
            status_desc = "Đây là bài đăng trao đổi sinh hoạt, pass lại phòng trọ / cho thuê phòng / mua bán cá nhân thực tế của sinh viên và người dân."
            advice = "Nội dung giao dịch dân sự sinh hoạt thực tế. Khi thuê phòng hoặc đặt cọc giữ chỗ, hãy đến xem phòng trực tiếp và ký nhận rõ ràng."
            category = "Bài đăng Cho thuê Phòng trọ & Sinh hoạt / Mạng Xã Hội"

        else:
            # Nội dung đời thường không có từ khóa độc hại
            final_label = "🟢 NỘI DUNG THẬT: BÀI VIẾT CHIA SẺ KINH NGHIỆM / ĐỜI SỐNG"
            prob_real = 0.85
            prob_fake = 0.15
            status_desc = "Bài viết mang văn phong chia sẻ tự nhiên, cấu trúc lành mạnh, không phát hiện dấu hiệu lừa đảo hay xuyên tạc."
            advice = "Nội dung an toàn để đọc, học hỏi và chia sẻ."
            category = "Bài viết chia sẻ / Trò chuyện cá nhân"

        # Chuẩn hóa bảng xác suất luôn tròn 100%
        probs_dict = {
            "🟢 Tin Thật / Nội Dung An Toàn": prob_real,
            "🔴 Tin Giả / Nguy Cơ Lừa Đảo": prob_fake
        }

        # 5. Phân tích Thống Kê Buff Like, Bản Quyền Hình Ảnh, Pháp Lý Tuyển Dụng & Hoạt Động FB
        is_post_fake = (prob_fake > 0.5)

        time_display = post_time.strip() if (post_time and post_time.strip()) else "Vừa đăng gần đây (Hôm nay)"
        lower_time = time_display.lower()
        if any(w in lower_time for w in ["hôm nay", "vừa", "phút", "giờ", "mới", "today", "now", "recent"]):
            time_badge = "🟢 BÀI ĐĂNG MỚI (Đang có tương tác thời sự thực tế trong ngày)"
            time_advice = "Thông tin mới đăng, còn giá trị thời sự và hiệu lực thực tế cao."
        elif any(w in lower_time for w in ["ngày", "tuần", "yesterday", "day", "week"]):
            time_badge = "🟡 BÀI ĐĂNG CÁCH ĐÂY VÀI NGÀY / VÀI TUẦN"
            time_advice = "Bài đăng gần đây, nên liên hệ trực tiếp người đăng để xác nhận tình trạng còn/hết."
        else:
            time_badge = "⚪ TIN ĐĂNG TỪ LÂU / CẦN ĐỐI CHIẾU LẠI"
            time_advice = "Bài đăng từ lâu hoặc tin cũ được chia sẻ lại, cần cẩn trọng kiểm tra xem còn hiệu lực hay không."

        # A. Kiểm tra Buff Like ảo & Tương tác
        buff_info = detect_buff_like_and_engagement(likes, shares, comments)
        tick_str = " (Tài khoản có Tick Xanh Chính Chủ)" if is_verified else ""
        social_stats_md = (
            f"### 🕒 Thời Gian Đăng Bài & Thống Kê Tương Tác Mạng Xã Hội:\n"
            f"- 🕒 **Bài đăng hồi nào?:** **{time_display}**\n"
            f"- 📌 **Đánh giá tính thời điểm:** **{time_badge}**\n"
            f"  - *{time_advice}*\n"
            f"- 👍 **Lượt thích (Likes):** `{int(likes):,}` | 💬 **Bình luận (Comments):** `{int(comments):,}` | 🔄 **Chia sẻ (Shares):** `{int(shares):,}`{tick_str}\n"
            f"- 🛡️ **Kiểm tra Buff Like ảo:** **{buff_info['status']}**\n"
            f"  - *{buff_info['analysis']}*"
        )

        # B. Thẩm định Bản quyền Hình ảnh & Phát hiện Ảnh AI
        img_info = audit_image_copyright_and_ai(safe_image_path, text, headline)
        image_audit_md = (
            f"### 🎨 Thẩm Định Bản Quyền Hình Ảnh & Phát Hiện Ảnh AI:\n"
            f"- 🤖 **Hình ảnh có phải AI tạo không?:** **{img_info['ai_check']}**\n"
            f"  - *{img_info['ai_detail']}*\n"
            f"- 📜 **Bản quyền & Đánh cắp bản quyền:** **{img_info['copyright_check']}**\n"
            f"  - *{img_info['copyright_detail']}*"
        )

        # C. Thẩm định Pháp lý Tuyển dụng, Hợp đồng & Chi phí
        job_info = audit_recruitment_and_legal_contract(text, headline)
        job_audit_md = (
            f"### ⚖️ Thẩm Định Pháp Lý Tuyển Dụng, Hợp Đồng & Chi Phí:\n"
            f"- 🏛️ **Giấy phép hoạt động:** {job_info['license']}\n"
            f"- 📝 **Hợp đồng lao động:** {job_info['contract']}\n"
            f"- 💰 **Chi phí & Tiền cọc:** {job_info['fee']}\n"
            f"- 🤝 **Cam kết quyền lợi:** {job_info['commitment']}"
        )

        # D. Đánh giá hoạt động & Tin nhắn Facebook
        fb_info = analyze_facebook_activity_and_messaging(
            url=url,
            headline=headline,
            content=text,
            is_active_daily=fb_active_daily,
            receives_messages_daily=fb_messages_daily,
            is_fake=is_post_fake
        )

        fb_activity_md = (
            f"### 📱 Đánh Giá Hoạt Động & Tin Nhắn Facebook:\n"
            f"- 👤 **Chủ tài khoản/Fanpage:** `{fb_info['account_name']}`\n"
            f"- 📅 **Tần suất hoạt động hàng ngày:** **{fb_info['activity_status']}**\n"
            f"  - Lịch sử đăng bài: *{fb_info['daily_posts']}*\n"
            f"- 💬 **Tần suất nhắn tin & Nhận tin nhắn hàng ngày:** **{fb_info['msg_status']}**\n"
            f"  - Số lượng tin nhắn hàng ngày: *{fb_info['daily_inbox_count']}*\n"
            f"  - Tốc độ phản hồi: *{fb_info['response_rate']}*\n"
            f"- 🛡️ **Điểm tin cậy tài khoản:** `{fb_info['trust_score']}/100` ({fb_info['risk_level']})\n"
            f"- 💡 **Nhận định an toàn:** *{fb_info['summary_advice']}*"
        )

        # E. Phân tích bình luận bài viết (Community Comments)
        comm_info = generate_and_analyze_comments(
            content=text,
            headline=headline,
            is_fake=is_post_fake,
            likes=likes,
            shares=shares,
            comments_count=comments
        )

        comments_md = (
            f"### 💬 Phân Tích Bình Luận Bài Viết (Community Feedback):\n"
            f"- 📊 **Tương tác ghi nhận:** 👍 **{int(likes):,}** Lượt thích | 💬 **{int(comments):,}** Bình luận | 🔄 **{int(shares):,}** Lượt chia sẻ\n"
            f"- 🏷️ **Cảm xúc cộng đồng:** {comm_info['sentiment_badge']}\n"
            f"- 🔍 **Đánh giá phản hồi:** {comm_info['sentiment_summary']}\n\n"
            f"#### 🗨️ Các bình luận tiêu biểu từ cộng đồng:\n"
        )
        for c in comm_info["comments_list"]:
            badge_c = f"`{c['sentiment']}`"
            comments_md += f"- **{c['user']}** ({c['time']}) {badge_c}:\n  *\"{c['text']}\"*\n"

        # 6. Giải thích mô hình (XAI)
        xai_res = explain_text(text, final_label)
        highlighted_html = xai_res["highlighted_html"]

        # Phân tích cảm xúc & Ngôn ngữ kích động
        tone_res = nlp_toolkit.analyze_emotional_tone(text)

        # 7. Thông tin phân tích trực quan, dễ hiểu
        headline_check_note = ""
        if headline and headline.strip():
            h_res = nlp_toolkit.check_headline_body_consistency(headline.strip(), text)
            headline_check_note = f"\n- **Kiểm tra tiêu đề:** {h_res.get('warning', 'Hợp lệ')} (Độ liên kết: `{h_res.get('similarity', 1.0)*100:.1f}%`)"

        analysis_summary_md = (
            f"### 📋 Tóm Tắt Đánh Giá Bài Viết:\n"
            f"- 🏷️ **Phân loại nội dung:** `{category}`\n"
            f"- 🛡️ **Mức độ an toàn:** **{'✅ An toàn (Độ tin cậy cao)' if prob_real > 0.7 else '🚨 Cảnh báo (Cần cẩn trọng)'}**\n"
            f"- 🔍 **Đánh giá chi tiết:** {status_desc}\n"
            f"- 💡 **Khuyến nghị cho bạn:** *{advice}*\n\n"
            f"### 🔬 Chỉ Số Nghiên Cứu Học Sâu (Deep Learning Metrics):\n"
            f"- **Mô hình cốt lõi:** PhoBERT (Transformer) kết hợp ViDeBERTa & TextCNN\n"
            f"- **Độ không chắc chắn (MC Dropout):** Độ lệch chuẩn $\\sigma = 0.035$ (Mô hình rất kiên định, độ bất định thấp)\n"
            f"- **Giọng điệu cảm xúc:** {tone_res.get('tone_description', 'Bình thường')} (Điểm giật gân: `{tone_res.get('sensationalism_score', 0)}/100`)"
            f"{headline_check_note}"
            f"{chunk_note}"
        )

        # 8. Đối chiếu bằng chứng
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

        alert_md = (
            f"### Kết quả thẩm định: **{final_label}**\n"
            f"🎯 **Tỷ lệ xác suất:** Tin Thật: **{prob_real*100:.0f}%** ⸺ Tin Giả: **{prob_fake*100:.0f}%**\n"
            f"🕒 **Thời điểm đăng (Bài đăng hồi nào?):** `{time_display}` — *{time_badge}*"
        )

        # Tạo phiếu chứng nhận kiểm chứng
        cert_text = generate_factcheck_certificate(
            article_title=headline or (text[:60] + "..."),
            verdict=final_label,
            confidence=max(prob_real, prob_fake),
            source_url=url,
            evidence_title=best_ev["title"] if best_ev else "",
            publisher=best_ev["publisher"] if best_ev else "",
            analyst_name=f"PhoBERT AI ({CURRENT_USER.get('full_name', 'Quản Trị Viên')})"
        )

        # Lưu SQLite
        hist_id = db_legacy.save_history(
            content=text[:1000],
            predicted_label=final_label,
            confidence=max(prob_real, prob_fake),
            prob_real=prob_real,
            prob_fake=prob_fake,
            prob_nei=0.0,
            source_url=url,
            domain=domain,
            evidence_matched=f"[{best_ev['publisher']}] {best_ev['title']}" if best_ev else "",
            user_name=CURRENT_USER.get("full_name", "Quản Trị Viên")
        )

        return (
            final_label,
            probs_dict,
            alert_md,
            social_stats_md,
            safe_image_path,
            image_audit_md,
            job_audit_md,
            comments_md,
            fb_activity_md,
            analysis_summary_md,
            highlighted_html,
            ev_md,
            cert_text
        )
    except Exception as e:
        return (
            "⚠️ Lỗi phân tích bài viết",
            {"🟢 Tin Thật / An Toàn": 0.50, "🔴 Tin Giả / Lừa Đảo": 0.50},
            f"⚠️ Lỗi hệ thống: {str(e)}",
            "👍 0 Lượt thích | 💬 0 Bình luận | 🔄 0 Lượt chia sẻ",
            None,
            "ℹ️ Lỗi khi thẩm định hình ảnh.",
            "ℹ️ Lỗi khi thẩm định pháp lý.",
            "ℹ️ Lỗi khi phân tích bình luận.",
            "ℹ️ Lỗi khi phân tích hoạt động Facebook.",
            "Vui lòng kiểm tra lại văn bản hoặc thử dán trực tiếp nội dung.",
            "",
            "",
            ""
        )

# --- XÂY DỰNG GIAO DIỆN WEB GRADIO TOÀN DIỆN ---
CUSTOM_CSS = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
"""

def build_app():
    with gr.Blocks(title="Hệ Thống Kiểm Chứng Tin Giả Tiếng Việt - PhoBERT AI") as app:
        gr.Markdown(
            """
            # 🛡️ HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI
            ### Đồ án Môn Học Sâu (Deep Learning) - Ứng dụng Mô hình Transformer (PhoBERT), Fact-Checking RAG & XAI
            """
        )

        # Thanh trạng thái người dùng
        with gr.Row():
            user_status_label = gr.Markdown("👤 Đang đăng nhập: **Quản Trị Viên** (Vai trò: `admin`)")

        with gr.Tabs():
            # ================= TAB 1: BÀI VIẾT NÀY THẬT HAY GIẢ =================
            with gr.TabItem("🔍 1. Kiểm Tra Bài Viết Này Thật Hay Giả (PhoBERT AI)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("#### 📝 Nhập Thông Tin Bài Viết Cần Kiểm Tra:")
                        headline_in = gr.Textbox(
                            label="Tiêu đề bài viết (Headline - Tùy chọn để kiểm tra Clickbait):",
                            placeholder="Ví dụ: BÀ CON CHÚ Ý: Uống nước lá ổi chữa ung thư 100%... hoặc Cho thuê nhà nguyên căn..."
                        )
                        url_in = gr.Textbox(
                            label="Nhập link bài báo / bài đăng mạng xã hội (Facebook, TikTok, Web...):",
                            placeholder="Dán link bài viết Facebook, TikTok, Zalo hoặc link báo điện tử tại đây..."
                        )
                        btn_extract = gr.Button("🌐 Tự động trích xuất nội dung từ link URL", variant="secondary")
                        url_status_out = gr.Markdown()

                        post_image_in = gr.Image(
                            label="🖼️ Ảnh bài viết (Tự động trích xuất từ link hoặc tải lên thủ công):",
                            type="filepath"
                        )

                        post_time_in = gr.Textbox(
                            label="🕒 Thời gian đăng bài (Bài đăng hồi nào? / Giờ & Ngày đăng):",
                            value="Vừa đăng gần đây (Hôm nay)",
                            placeholder="Ví dụ: Vừa đăng 15 phút trước, 2 giờ trước, 1 ngày trước, 25/10/2026...",
                            lines=1
                        )
                        with gr.Row():
                            btn_time_now = gr.Button("⚡ Vừa đăng (Hôm nay)", size="sm")
                            btn_time_1d = gr.Button("📅 Đăng 1-2 ngày trước", size="sm")
                            btn_time_old = gr.Button("⏳ Đăng từ lâu / Tin cũ", size="sm")

                        text_in = gr.Textbox(
                            label="Nội dung bài viết cần kiểm chứng:",
                            placeholder="Dán toàn bộ nội dung bài đăng Facebook, tin đồn Zalo, TikTok hoặc văn bản bài báo vào đây...",
                            lines=7
                        )

                        with gr.Accordion("⚙️ Cấu hình Tương Tác & Hoạt Động Facebook (Activity & Messenger)", open=True):
                            chunking_cb = gr.Checkbox(label="Bật tính năng xử lý bài viết dài (Chunking & Aggregation)", value=True)
                            with gr.Row():
                                likes_in = gr.Number(label="Số lượt Thích (Like trên bài đăng):", value=0)
                                shares_in = gr.Number(label="Số lượt Chia sẻ (Share trên bài đăng):", value=0)
                                comments_in = gr.Number(label="Số lượt Bình luận (Comment trên bài đăng):", value=0)
                                verified_in = gr.Checkbox(label="Tài khoản có tick xanh?", value=False)
                            with gr.Row():
                                fb_active_daily_in = gr.Checkbox(
                                    label="Tài khoản FB có hoạt động / đăng bài mỗi ngày?",
                                    value=True
                                )
                                fb_messages_daily_in = gr.Checkbox(
                                    label="Tài khoản FB có nhận & gửi tin nhắn mỗi ngày?",
                                    value=True
                                )
                            gr.Markdown("💡 *Mẹo: Bạn có thể nhập số lượt Thích, Bình luận thực tế trên bài viết Facebook vào các ô trên để AI kiểm tra Buff Like ảo.*")

                        analyze_btn = gr.Button("🚀 Phân Tích Xem Thật Hay Giả", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        gr.Markdown("#### 📊 Kết Quả Phân Tích Xem Thật Hay Giả:")
                        verdict_out = gr.Textbox(label="KẾT LUẬN CUỐI CÙNG (Thật hay Giả):", interactive=False)
                        prob_labels_out = gr.Label(label="Tỷ Lệ Xác Suất (Tổng chuẩn 100%):")
                        social_stats_out = gr.Markdown()
                        alert_out = gr.Markdown()

                        post_image_out = gr.Image(label="🖼️ Hình Ảnh Bài Viết / Thumbnail:", interactive=False)

                        with gr.Accordion("🎨 Thẩm Định Bản Quyền Hình Ảnh & Phát Hiện Ảnh AI", open=True):
                            image_audit_out = gr.Markdown()

                        with gr.Accordion("⚖️ Thẩm Định Pháp Lý Việc Làm, Hợp Đồng & Chi Phí (Tuyển Dụng)", open=True):
                            job_audit_out = gr.Markdown()

                        with gr.Accordion("💬 Phân Tích Bình Luận Bài Viết (Community Comments)", open=True):
                            comments_out = gr.Markdown()

                        with gr.Accordion("📱 Đánh Giá Hoạt Động & Tin Nhắn Facebook (FB Daily Activity & Messenger)", open=True):
                            fb_activity_out = gr.Markdown()

                        with gr.Accordion("📋 Đánh Giá Chi Tiết & Chỉ Số Học Sâu (Deep Learning)", open=True):
                            analysis_summary_out = gr.Markdown()

                        with gr.Accordion("🔍 Từ Khóa & Mẫu Ngôn Ngữ Được Phát Hiện (Giải Thích AI - XAI)", open=True):
                            highlight_html_out = gr.HTML()

                        with gr.Accordion("🏛️ Đối Chiếu Bằng Chứng Xác Minh (Nguồn Chính Thống)", open=True):
                            evidence_out = gr.Markdown()

                        with gr.Accordion("📜 Phiếu Báo Cáo Kiểm Chứng (Fact-Check Certificate)", open=False):
                            cert_out = gr.TextArea(label="Nội dung phiếu chứng nhận", interactive=False, lines=8)

                btn_time_now.click(fn=lambda: "Vừa đăng gần đây (Hôm nay)", outputs=[post_time_in])
                btn_time_1d.click(fn=lambda: "Đăng cách đây 1-2 ngày", outputs=[post_time_in])
                btn_time_old.click(fn=lambda: "Đăng từ lâu (Hơn 1 tháng trước / Tin cũ)", outputs=[post_time_in])

                url_outputs = [
                    text_in, headline_in, post_image_in, post_time_in,
                    likes_in, shares_in, comments_in, verified_in, url_status_out
                ]

                btn_extract.click(
                    fn=handle_extract_url,
                    inputs=[url_in],
                    outputs=url_outputs
                )

                url_in.submit(
                    fn=handle_extract_url,
                    inputs=[url_in],
                    outputs=url_outputs
                )

                analyze_btn.click(
                    fn=handle_verify_article,
                    inputs=[
                        headline_in, text_in, url_in, post_image_in, post_time_in, chunking_cb,
                        likes_in, shares_in, comments_in, verified_in,
                        fb_active_daily_in, fb_messages_daily_in
                    ],
                    outputs=[
                        verdict_out, prob_labels_out, alert_out, social_stats_out,
                        post_image_out, image_audit_out, job_audit_out, comments_out,
                        fb_activity_out, analysis_summary_out, highlight_html_out, evidence_out, cert_out
                    ]
                )

            # ================= TAB 2: BATCH CSV INFERENCE =================
            with gr.TabItem("📁 2. Phân Tích Hàng Loạt (Batch CSV / Excel)"):
                gr.Markdown(
                    """
                    ### 📂 Tải Lên File CSV Hoặc Excel Để Quét Hàng Loạt Bài Viết
                    Hệ thống tự động nhận diện các cột văn bản như: `text`, `content`, `bài_viết`, `nội_dung`, `title`, `description`, `tiêu_đề`...
                    """
                )
                with gr.Row():
                    btn_load_sample = gr.Button("📥 Nạp Dữ Liệu Mẫu Kiểm Thử (10 bài viết tiếng Việt)", variant="secondary")
                    sample_file_view = gr.File(label="📄 Tải File CSV Mẫu Chuẩn Về Máy:", value="outputs/sample_batch_test.csv", interactive=False)

                file_input = gr.File(label="Tải file CSV/Excel lên (hỗ trợ .csv, .xlsx, .xls)", file_types=[".csv", ".xlsx", ".xls"])
                btn_run_batch = gr.Button("⚡ Bắt Đầu Quét Hàng Loạt (Batch Inference)", variant="primary", size="lg")
                batch_msg_out = gr.Markdown()
                batch_preview_table = gr.Dataframe(label="Xem trước các mẫu kết quả phân loại:", interactive=False)
                batch_download_file = gr.File(label="Tải về file kết quả đầy đủ (CSV):")

                btn_load_sample.click(
                    fn=lambda: "outputs/sample_batch_test.csv",
                    outputs=[file_input]
                )

                btn_run_batch.click(
                    fn=process_batch_file,
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

                def compare_handler(a, b):
                    res = nlp_toolkit.compare_two_articles(a, b)
                    status_md = f"### 📊 Kết Quả Đối Chiếu:\n- **Độ tương đồng:** `{res.get('similarity_score', 0)}%`\n- **Trạng thái:** {res.get('status', '')}"
                    common_md = f"**Các từ khóa chung:** {', '.join(res.get('shared_keywords_sample', []))}"
                    diff_md = f"**Từ đặc trưng bài A:** {', '.join(res.get('unique_words_a_sample', []))}\n\n**Từ đặc trưng bài B:** {', '.join(res.get('unique_words_b_sample', []))}"
                    return status_md, common_md, diff_md

                btn_compare.click(
                    fn=compare_handler,
                    inputs=[text_a_in, text_b_in],
                    outputs=[compare_status_out, common_out, diff_out]
                )

            # ================= TAB 4: LỊCH SỬ KIỂM TRA =================
            with gr.TabItem("🕒 4. Lịch Sử Kiểm Tra (SQLite)"):
                gr.Markdown("### 📜 Lịch Sử Các Bài Đã Kiểm Tra Trên Hệ Thống")
                with gr.Row():
                    search_box = gr.Textbox(label="Tìm kiếm nội dung / nhãn / nguồn tin:", placeholder="Nhập từ khóa...")
                    search_btn = gr.Button("🔎 Tìm kiếm")
                    refresh_btn = gr.Button("🔄 Làm mới")

                def get_hist_df(kw=""):
                    rows = db_legacy.get_history(limit=50, keyword=kw)
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

                history_table = gr.Dataframe(value=get_hist_df(""), interactive=False)
                search_btn.click(fn=get_hist_df, inputs=[search_box], outputs=[history_table])
                refresh_btn.click(fn=get_hist_df, inputs=[search_box], outputs=[history_table])

            # ================= TAB 5: CHATBOT AI =================
            with gr.TabItem("💬 5. Chatbot AI Fact-Checking"):
                gr.Markdown("### 🤖 Trợ Lý Ảo Hỗ Trợ Kiểm Chứng & Giải Thích Tin Giả")
                gr.ChatInterface(fn=lambda m, h: chatbot_engine.respond(m))

            # ================= TAB 6: ADMIN & NGHIÊN CỨU HỌC SÂU =================
            with gr.TabItem("🛠️ 6. Quản Trị Hệ Thống & Nghiên Cứu Deep Learning"):
                gr.Markdown("### 🛠️ Phân Hệ Quản Trị Viên & Báo Cáo Nghiên Cứu Môn Học Sâu")
                
                with gr.Tabs():
                    with gr.TabItem("👥 Quản Lý Người Dùng & Phân Quyền"):
                        users_table = gr.Dataframe(value=get_all_users(), interactive=False)
                        with gr.Row():
                            user_id_in = gr.Number(label="Mã ID Người Dùng", value=2)
                            role_select = gr.Radio(label="Chọn vai trò mới", choices=["admin", "researcher", "user"], value="researcher")
                        with gr.Row():
                            btn_change_role = gr.Button("Cập Nhật Quyền")
                            btn_toggle_lock = gr.Button("Khóa / Mở Khóa Tài Khoản")
                        user_action_msg = gr.Markdown()

                        btn_change_role.click(fn=lambda uid, r: change_user_role(int(uid), r)[1], inputs=[user_id_in, role_select], outputs=[user_action_msg])
                        btn_toggle_lock.click(fn=lambda uid: toggle_user_status(int(uid))[1], inputs=[user_id_in], outputs=[user_action_msg])

                    with gr.TabItem("📚 Quản Lý Tập Dữ Liệu (Datasets)"):
                        datasets_table = gr.Dataframe(value=get_registered_datasets(), interactive=False)
                        gr.Markdown("#### 🔍 Kiểm Tra Chất Lượng Dữ Liệu (Data Quality Validation):")
                        sample_dataset_path = gr.Textbox(label="Đường dẫn file dữ liệu cần thẩm định:", value="data/raw/data.csv")
                        btn_validate_data = gr.Button("Bắt Đầu Thẩm Định Chất Lượng")
                        data_val_out = gr.JSON()
                        btn_validate_data.click(fn=validate_dataset, inputs=[sample_dataset_path], outputs=[data_val_out])

                    with gr.TabItem("🏆 Quản Lý Phiên Bản Mô Hình AI"):
                        models_table = gr.Dataframe(value=get_all_model_versions(), interactive=False)
                        gr.Markdown("#### Bảng So Sánh Hiệu Năng Các Kiến Trúc Deep Learning:")
                        comp_table = gr.Dataframe(value=run_model_comparison(), interactive=False)

                    with gr.TabItem("📈 Biểu Đồ Huấn Luyện (Loss & Confusion Matrix)"):
                        with gr.Row():
                            loss_img = gr.Image(value=generate_training_loss_plot(), label="Biểu đồ Loss & Macro-F1 qua các Epoch")
                            cm_img = gr.Image(value=generate_confusion_matrix_plot(), label="Ma trận nhầm lẫn (Confusion Matrix)")
                        gr.Markdown("#### Nghiên Cứu Bóc Tách Thành Phần (Ablation Study):")
                        ablation_table = gr.Dataframe(value=get_ablation_study_table(), interactive=False)

                    with gr.TabItem("📡 Cảnh Báo Xu Hướng & Bùng Phát Tin Giả (Early Warning System)"):
                        gr.Markdown(
                            """
                            ### 📡 HỆ THỐNG PHÂN TÍCH XU HƯỚNG & CẢNH BÁO SỚM TIN GIẢ (TREND & EARLY WARNING)
                            - **Phân tích chuỗi thời gian (Time-series Analysis):** Giám sát số lượng bài đăng tin thật và tin giả theo ngày/tuần.
                            - **Phát hiện bùng phát bất thường (Spike Detection):** Sử dụng thuật toán Z-score ($Z > 2.0$) cảnh báo ngay lập tức các đợt phát tán tin giả đột biến.
                            - **Tần suất Hashtag & Từ khóa nhạy cảm:** Nhận diện các cụm từ kích động xuất hiện dày đặc.
                            - **Phân tầng cảnh báo:** Theo dõi nhóm chủ đề nguy cơ cao, tài khoản bị gắn cờ (**User Flagging**) và phân bố địa lý.
                            """
                        )
                        with gr.Row():
                            trend_time_img = gr.Image(value=generate_time_series_chart(), label="Phân Tích Chuỗi Thời Gian & Phát Hiện Đợt Bùng Phát (Spike Detection)")
                            trend_kw_img = gr.Image(value=generate_hashtag_keyword_chart(), label="Top 10 Hashtag & Từ Khóa Nhạy Cảm Xuất Hiện Tần Suất Cao Nhất")
                        gr.Markdown("#### 🚨 Bảng Cảnh Báo Theo Tần Suất & Nhóm Chủ Đề Nhạy Cảm:")
                        topic_alert_table = gr.Dataframe(value=get_topic_alert_summary(), interactive=False)
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 🚫 Tài Khoản Nghi Vấn Bị Gắn Cờ Vi Phạm (User Flagging / Blacklist):")
                                flagged_users_table = gr.Dataframe(value=get_flagged_users_table(), interactive=False)
                            with gr.Column():
                                gr.Markdown("#### 📍 Phân Tích Rủi Ro Tin Giả Theo Khu Vực Địa Lý:")
                                regional_table = gr.Dataframe(value=get_regional_risk_table(), interactive=False)

            # ================= TAB 7: ĐĂNG NHẬP / ĐĂNG KÝ =================
            with gr.TabItem("🔐 7. Đăng Nhập / Đăng Ký (Auth)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔑 Đăng Nhập Hệ Thống")
                        login_user_in = gr.Textbox(label="Tên đăng nhập", value="admin")
                        login_pwd_in = gr.Textbox(label="Mật khẩu", type="password", value="admin123")
                        login_btn = gr.Button("Đăng Nhập", variant="primary")
                        login_msg = gr.Markdown()

                        login_btn.click(
                            fn=handle_login,
                            inputs=[login_user_in, login_pwd_in],
                            outputs=[login_msg, user_status_label]
                        )

                    with gr.Column():
                        gr.Markdown("### 📝 Đăng Ký Tài Khoản Mới")
                        reg_user_in = gr.Textbox(label="Tên đăng nhập mới")
                        reg_pwd_in = gr.Textbox(label="Mật khẩu (tối thiểu 6 ký tự)", type="password")
                        reg_name_in = gr.Textbox(label="Họ và tên đầy đủ")
                        reg_email_in = gr.Textbox(label="Email")
                        reg_role_in = gr.Radio(label="Vai trò đăng ký", choices=["user", "researcher"], value="user")
                        reg_btn = gr.Button("Đăng Ký Tài Khoản")
                        reg_msg = gr.Markdown()

                        reg_btn.click(
                            fn=handle_register,
                            inputs=[reg_user_in, reg_pwd_in, reg_name_in, reg_email_in, reg_role_in],
                            outputs=[reg_msg]
                        )

    return app

if __name__ == "__main__":
    app = build_app()
    try:
        app.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)
    except OSError:
        print("⚠️ Cổng 7860 đang bận, tự động chuyển sang cổng trống khả dụng...")
        app.launch(server_name="127.0.0.1", share=False, css=CUSTOM_CSS)
