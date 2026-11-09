"""
Module Chatbot AI Tư vấn & Giải thích Kiểm chứng Tin giả (Fact-Checking Assistant).
Chức năng:
1. Giải thích các khái niệm về tin giả (Fake news, Misinformation, Disinformation).
2. Hướng dẫn quy trình 5 bước kiểm chứng thông tin trên mạng xã hội.
3. Giải thích cơ chế hoạt động của mô hình Transformer / PhoBERT và ý nghĩa xác suất.
4. Trả lời các thắc mắc về nguồn tin uy tín tại Việt Nam.
5. Đưa ra khuyến nghị khi gặp một bài đăng giật gân, câu like.
"""

import sys
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cơ sở tri thức câu hỏi thường gặp của Chatbot (KB / Rules & Guidance)
KNOWLEDGE_BASE = [
    {
        "keywords": ["tin giả là gì", "khái niệm", "định nghĩa", "fake news"],
        "answer": """**Tin giả (Fake News)** là những thông tin sai lệch, xuyên tạc hoặc bịa đặt được lan truyền có chủ đích hoặc vô tình:
- **Misinformation (Thông tin sai lệch):** Chia sẻ thông tin sai nhưng người chia sẻ không có ác ý (nghĩ là thật).
- **Disinformation (Thông tin xuyên tạc):** Cố ý tạo dựng và phát tán tin bịa đặt nhằm trục lợi, gây hoang mang dư luận hoặc lừa đảo.
- **Malinformation:** Thông tin có một phần sự thật nhưng bị cắt ghép ngữ cảnh ác ý để bôi nhọ hoặc thao túng."""
    },
    {
        "keywords": ["cách kiểm tra", "hướng dẫn kiểm chứng", "làm sao biết", "quy trình"],
        "answer": """🔍 **Quy trình 5 bước tự kiểm chứng thông tin trên mạng xã hội:**
1. **Kiểm tra nguồn tin:** Bài viết xuất phát từ cơ quan báo chí chính thống hay trang cá nhân nặc danh? Tên miền có dấu hiệu mạo danh (như .co, .cc, .xyz) không?
2. **Kiểm tra tác giả & chuyên gia:** Người phát ngôn có chuyên môn thật trong lĩnh vực đó không (bác sĩ, nhà khoa học)?
3. **Đối chiếu chéo (Cross-check):** Thử tìm kiếm tiêu đề trên Google News, VTV, VnExpress, Tuổi Trẻ xem có báo đài nào đưa tin không.
4. **Kiểm tra hình ảnh/video:** Sử dụng công cụ *Google Reverse Image Search* để xem ảnh có bị lấy từ vụ việc khác trong quá khứ hay không.
5. **Cảnh giác với giọng điệu:** Các bài viết dùng từ "Khẩn cấp", "Chia sẻ gấp", "Tuyệt mật", "Trị dứt điểm 100%" thường là bẫy câu tương tác hoặc lừa đảo."""
    },
    {
        "keywords": ["phobert", "transformer", "mô hình", "hoạt động thế nào"],
        "answer": """🧠 **Mô hình PhoBERT hoạt động ra sao trong hệ thống này?**
- **PhoBERT** là mô hình ngôn ngữ dựa trên kiến trúc Transformer (RoBERTa) được tiền huấn luyện trên 20GB dữ liệu tiếng Việt.
- **Cơ chế Self-Attention:** Cho phép mô hình nhìn toàn bộ câu cùng lúc và hiểu mối quan hệ ngữ cảnh giữa các từ ghép tiếng Việt (ví dụ: *Bộ_Y_tế*, *khuyến_cáo*).
- **Phân loại nhị phân / 3 nhãn:** Đầu ra của PhoBERT là vector biểu diễn [CLS] được đưa qua tầng phân loại để tính xác suất cho từng nhãn: *Tin thật (Supported)*, *Tin giả (Refuted)* hoặc *Chưa đủ thông tin (NEI)*."""
    },
    {
        "keywords": ["nguồn tin uy tín", "trang web chính thống", "báo nào tin được"],
        "answer": """🌐 **Các nguồn tin chính thống tin cậy hàng đầu tại Việt Nam:**
- **Thông tấn xã & Truyền hình:** Thông tấn xã Việt Nam (TTXVN), Đài Truyền hình Việt Nam (VTV), Đài Tiếng nói Việt Nam (VOV).
- **Cổng thông tin Chính phủ & Bộ ngành:** Cổng TTĐT Chính phủ (`chinhphu.vn`), Bộ Y tế (`moh.gov.vn`), Bộ GD&ĐT (`moet.gov.vn`).
- **Báo chí chính luận & uy tín:** Báo Nhân Dân, Quân đội Nhân dân, Tuổi Trẻ, VnExpress, Thanh Niên, Dân Trí, VietnamNet.
- **Cảnh báo lừa đảo mạng:** Cục An toàn thông tin (`ais.gov.vn`), Cổng Không Gian Mạng Quốc Gia (`khonggianmang.vn`)."""
    },
    {
        "keywords": ["xác suất", "độ tin cậy", "confidence", "nghĩa là gì"],
        "answer": """📊 **Ý nghĩa của Độ Tin Cậy (Confidence Score):**
- **Trên 85%**: Mô hình rất tự tin dựa trên các bằng chứng đối chiếu mạnh hoặc ngôn ngữ bài viết có cấu trúc đặc trưng rất rõ nét.
- **Từ 65% - 85%**: Độ tin cậy ở mức khá, khuyến khích đối chiếu thêm nguồn tin gốc.
- **Dưới 65%**: Mô hình cảnh báo thiếu độ tin cậy do nội dung trung tính hoặc thông tin mập mờ, hệ thống sẽ tự động chuyển sang nhãn *'Chưa Đủ Thông Tin (NEI)'* để tránh kết luận sai lệch."""
    }
]

class FactCheckChatbot:
    def __init__(self):
        pass

    def respond(self, user_message: str, history: list = None) -> str:
        """Sinh câu trả lời thông minh dựa trên câu hỏi của người dùng."""
        if not user_message or not user_message.strip():
            return "Xin chào! Tôi là Trợ lý AI Fact-Checking. Hãy đặt câu hỏi về cách kiểm tra tin giả hoặc cơ chế mô hình PhoBERT!"

        msg = user_message.lower().strip()

        # So khớp với tri thức đã cấu hình
        for item in KNOWLEDGE_BASE:
            if any(kw in msg for kw in item["keywords"]):
                return item["answer"]

        # Nếu hỏi về một bài đăng cụ thể trong cuộc hội thoại
        if "tin này" in msg or "bài viết này" in msg:
            return "Bạn hãy sao chép nội dung bài viết đó và dán vào tab **'Kiểm tra tin tức AI'** ở phía trên. Hệ thống PhoBERT sẽ tự động phân tích và đối chiếu bằng chứng ngay lập tức cho bạn!"

        # Chào hỏi thông thường
        if any(w in msg for w in ["chào", "hi", "hello", "xin chào"]):
            return "Xin chào bạn! Tôi có thể giúp bạn giải thích về tin giả, hướng dẫn các bước kiểm chứng bài đăng mạng xã hội, hoặc giải thích cách thức hoạt động của mô hình PhoBERT. Bạn cần hỗ trợ gì?"

        # Trả lời mặc định
        return f"""Tôi đã ghi nhận câu hỏi của bạn: *"{user_message}"*.
Để kiểm tra tính xác thực của một thông tin cụ thể, bạn có thể:
1. Sang tab **'Kiểm tra tin tức AI'** để dán nội dung hoặc link bài báo cần quét.
2. Tìm kiếm trên các cổng thông tin chính thống như *VTV, TTXVN, Báo Chính phủ*.
3. Đặt các câu hỏi như: *"Tin giả là gì?"*, *"Quy trình kiểm chứng thông tin"*, *"PhoBERT hoạt động thế nào?"* để tôi hỗ trợ thêm!"""

if __name__ == "__main__":
    bot = FactCheckChatbot()
    print("Chatbot test:")
    print(bot.respond("Tin giả là gì?"))

