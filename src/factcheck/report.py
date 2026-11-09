"""
Module Xuất Báo Cáo Kiểm Chứng (Fact-Checking Report Generator).
Tạo phiếu tóm tắt kết quả kiểm chứng (dạng Text / Markdown / HTML) có thể in hoặc lưu trữ.
"""

from datetime import datetime

def generate_factcheck_certificate(
    article_title: str,
    verdict: str,
    confidence: float,
    source_url: str,
    evidence_title: str,
    publisher: str,
    analyst_name: str = "Hệ Thống PhoBERT AI"
) -> str:
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    report = f"""
================================================================================
🏛️ PHIẾU KẾT QUẢ KIỂM CHỨNG THÔNG TIN MẠNG XÃ HỘI (FACT-CHECK REPORT)
Thời gian cấp: {now} | Đơn vị thẩm định: {analyst_name}
================================================================================

1. THÔNG TIN BÀI VIẾT KIỂM TRA:
- Tiêu đề / Nội dung: {article_title[:150]}...
- Nguồn phát tán: {source_url or 'Mạng xã hội tự do'}

2. KẾT QUẢ PHÂN TÍCH HỌC SÂU (DEEP LEARNING):
- Kết luận: {verdict.upper()}
- Độ tin cậy mô hình: {confidence * 100:.1f}%

3. CƠ CHẾ ĐỐI CHIẾU BẰNG CHỨNG XÁC THỰC:
- Tài liệu đối chiếu: {evidence_title or 'Chưa có thông cáo đối chiếu trực tiếp'}
- Cơ quan phát ngôn chính thức: {publisher or 'Đang xác minh'}

4. KHUYẾN NGHỊ CHO NGƯỜI DÙNG:
- Không chia sẻ các bài viết có gắn nhãn REFUTED (Tin Giả).
- Luôn kiểm chứng lại từ các kênh tin tức chính thống của Chính phủ và Bộ ban ngành.
================================================================================
    """
    return report.strip()

