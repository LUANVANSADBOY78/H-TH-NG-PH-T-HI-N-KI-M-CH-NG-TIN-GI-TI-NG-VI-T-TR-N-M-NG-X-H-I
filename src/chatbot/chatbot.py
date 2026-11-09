"""
Module Chatbot AI Tư vấn & Giải thích Kiểm chứng Tin giả (Fact-Checking Assistant).
"""

import sys
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KNOWLEDGE_BASE = [
    {
        "keywords": ["tin giả là gì", "khái niệm", "định nghĩa", "fake news"],
        "answer": """**Tin giả (Fake News)** là những thông tin sai lệch, xuyên tạc hoặc bịa đặt được lan truyền có chủ đích hoặc vô tình:
- **Misinformation (Thông tin sai lệch):** Chia sẻ thông tin sai nhưng người chia sẻ không có ác ý.
- **Disinformation (Thông tin xuyên tạc):** Cố ý tạo dựng và phát tán tin bịa đặt nhằm trục lợi, lừa đảo.
- **Malinformation:** Thông tin có một phần sự thật nhưng bị cắt ghép ác ý để bôi nhọ."""
    },
    {
        "keywords": ["cách kiểm tra", "hướng dẫn kiểm chứng", "làm sao biết", "quy trình"],
        "answer": """🔍 **Quy trình 5 bước tự kiểm chứng thông tin trên mạng xã hội:**
1. **Kiểm tra nguồn tin:** Cơ quan báo chí chính thống hay trang cá nhân nặc danh?
2. **Kiểm tra tác giả & chuyên gia:** Người phát ngôn có chuyên môn thật trong lĩnh vực đó không?
3. **Đối chiếu chéo (Cross-check):** Tìm kiếm trên Google News, VTV, Tuổi Trẻ, VnExpress.
4. **Kiểm tra hình ảnh/video:** Sử dụng Google Reverse Image Search.
5. **Cảnh giác với giọng điệu:** Các bài viết dùng từ "Khẩn cấp", "Chia sẻ gấp", "Trị dứt điểm 100%"."""
    },
    {
        "keywords": ["phobert", "transformer", "mô hình", "hoạt động thế nào"],
        "answer": """🧠 **Mô hình PhoBERT hoạt động ra sao trong hệ thống này?**
- **PhoBERT** là mô hình Transformer tiền huấn luyện trên 20GB dữ liệu tiếng Việt.
- **Cơ chế Self-Attention:** Cho phép mô hình hiểu ngữ cảnh giữa các từ ghép tiếng Việt (ví dụ: *Bộ_Y_tế*, *khuyến_cáo*).
- **Phân loại 3 nhãn:** Đầu ra là vector [CLS] để dự đoán: *Supported*, *Refuted*, hoặc *Not Enough Info (NEI)*."""
    },
    {
        "keywords": ["nguồn tin uy tín", "trang web chính thống", "báo nào tin được"],
        "answer": """🌐 **Các nguồn tin chính thống tin cậy hàng đầu tại Việt Nam:**
- TTXVN, VTV, VOV, Báo Nhân Dân, Quân đội Nhân dân.
- Cổng TTĐT Chính phủ (`chinhphu.vn`), Bộ Y tế (`moh.gov.vn`), Bộ GD&ĐT (`moet.gov.vn`).
- Báo chí uy tín: Tuổi Trẻ, VnExpress, Thanh Niên, Dân Trí, VietnamNet."""
    }
]

class FactCheckChatbot:
    def __init__(self):
        pass

    def respond(self, user_message: str, history: list = None) -> str:
        if not user_message or not user_message.strip():
            return "Xin chào! Tôi là Trợ lý AI Fact-Checking. Hãy đặt câu hỏi về cách kiểm tra tin giả hoặc cơ chế PhoBERT!"

        msg = user_message.lower().strip()
        for item in KNOWLEDGE_BASE:
            if any(kw in msg for kw in item["keywords"]):
                return item["answer"]

        if any(w in msg for w in ["chào", "hi", "hello", "xin chào"]):
            return "Xin chào bạn! Tôi có thể giúp bạn giải thích về tin giả, hướng dẫn kiểm chứng nguồn tin hoặc giải thích cách thức hoạt động của mô hình PhoBERT."

        return f"""Tôi đã ghi nhận câu hỏi của bạn: *"{user_message}"*.
Để kiểm tra bài viết có phải tin giả hay không, bạn hãy chuyển sang tab **'🔍 1. Bài Viết Này Thật Hay Giả?'** để dán nội dung hoặc link bài viết. Hệ thống PhoBERT và kho bằng chứng sẽ phân tích ngay lập tức!"""

