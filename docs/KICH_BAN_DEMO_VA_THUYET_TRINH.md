# 🎤 KỊCH BẢN THUYẾT TRÌNH & DEMO ĐỒ ÁN HỌC SÂU (DEEP LEARNING)
**Đề tài**: Xây dựng hệ thống phát hiện và hỗ trợ kiểm chứng tin giả tiếng Việt trên mạng xã hội sử dụng mô hình Transformer (PhoBERT), tích hợp Chatbot AI và XAI.

---

## PHẦN 1: KỊCH BẢN DEMO TRỰC TIẾP TRÊN WEB (Thời lượng: 3 - 5 phút)

Khi giảng viên hoặc hội đồng yêu cầu: *"Em hãy demo sản phẩm của mình"*, bạn hãy tự tin mở trình duyệt tại `http://127.0.0.1:7860` và dẫn dắt theo 3 bước sau:

### 📍 Màn 1: Demo Phân Tích Tin Đơn Lẻ & Khả Năng Giải Thích (Tab 1)
- **Lời nói mở đầu**:  
  *"Thưa thầy/cô, đây là giao diện chính của hệ thống. Em xin phép demo trường hợp phát hiện một tin giả lừa đảo chữa bệnh điển hình trên Facebook."*
- **Thao tác**:
  1. Dán nội dung mẫu:  
     `BÀ CON CHÚ Ý: Uống nước lá ổi đun sôi cùng chanh sả cam kết trị dứt điểm 100% tế bào ung thư sau 7 ngày, chia sẻ gấp cho người thân kẻo bị xóa bài!!`
  2. Bấm nút: **`🚀 Phân Tích & Đối Chiếu Bằng Chứng`**.
- **Điểm nhấn cần chỉ cho thầy/cô xem trên màn hình**:
  - *"Hệ thống đưa ra nhãn cảnh báo đỏ: **Refuted (Tin Giả / Bác Bỏ)** với độ tin cậy cao."*
  - *"Tại mục **Giải Thích AI (XAI)**, mô hình đã tự động phát hiện và tô đỏ các từ ngữ kích động, giật gân như 'bà con chú ý', '100%', 'chữa dứt điểm', 'chia sẻ gấp'."*
  - *"Đặc biệt, hệ thống sử dụng cơ chế **Truy xuất Bằng chứng (Evidence Retrieval)** để tự động trích xuất thông cáo bác bỏ chính thức từ **Bộ Y Tế** để đối chiếu phản bác."*
  - *"Chỉ số **Monte Carlo Dropout Uncertainty** đạt độ lệch chuẩn rất thấp ($\sigma = 0.035$), chứng minh mô hình cực kỳ kiên định với phán đoán này."*

---

### 📍 Màn 2: Demo Khả Năng Xử Lý Dữ Liệu Hàng Loạt (Tab 2)
- **Lời nói**:  
  *"Trong thực tế, các cơ quan an ninh mạng hoặc quản trị viên phải rà quét hàng nghìn bài viết mỗi ngày. Vì vậy nhóm em đã phát triển tính năng Phân tích hàng loạt qua file."*
- **Thao tác**:
  1. Bấm sang **Tab 2: Phân Tích Hàng Loạt (Batch CSV)**.
  2. Chọn file: `data/raw/data.csv` (có sẵn trong dự án).
  3. Bấm **`⚡ Bắt Đầu Quét Hàng Loạt`**.
- **Điểm nhấn**:
  - *"Hệ thống tự động tiền xử lý, chạy pipeline và xuất ra bảng thống kê mức độ rủi ro (Cao / Trung bình / An toàn) cho từng bài viết."*
  - *"Người dùng có thể bấm nút Tải về để xuất toàn bộ kết quả gắn nhãn ra file CSV phục vụ kiểm tra ngoại tuyến."*

---

### 📍 Màn 3: Demo Báo Cáo Nghiên Cứu Deep Learning (Tab 6 - Admin)
- **Lời nói**:  
  *"Để chứng minh tính khoa học của mô hình học sâu, em xin trình bày phân hệ Dashboard Admin và các biểu đồ thực nghiệm."*
- **Thao tác**:
  1. Bấm sang **Tab 6: Admin & Nghiên Cứu Học Sâu**.
- **Điểm nhấn ăn điểm môn Học sâu**:
  - **Biểu đồ Loss Curves**: *"Đồ thị cho thấy Training Loss và Validation Loss giảm dần và hội tụ mượt mà sau 5 epoch, không có hiện tượng Overfitting."*
  - **Ma trận nhầm lẫn (Confusion Matrix)**: *"Tỉ lệ nhận diện đúng tin giả (Recall) đạt trên 98%, hạn chế tối đa việc lọt lưới tin độc hại."*
  - **Bảng Ablation Study**: *"Qua nghiên cứu bóc tách, nhóm chứng minh việc tách từ tiếng Việt giúp tăng F1 thêm 5.3%, và việc kết hợp đặc trưng Metadata tương tác mạng xã hội giúp cải thiện thêm 2.3% so với mô hình thuần văn bản."*

---

## PHẦN 2: BỐ CỤC 15 SLIDE THUYẾT TRÌNH CHUẨN MỰC

1. **Slide 1: Trang bìa**  
   - Tên đề tài, Giảng viên hướng dẫn, Sinh viên thực hiện, Ngày báo cáo.
2. **Slide 2: Đặt vấn đề & Tính cấp thiết**  
   - Sự bùng nổ của tin giả trên mạng xã hội tiếng Việt (Facebook, TikTok).
   - Tác hại đến sức khỏe, trật tự xã hội và thách thức của ngôn ngữ mạng (teencode, giật tít).
3. **Slide 3: Mục tiêu đề tài**  
   - Xây dựng hệ thống phát hiện tin giả tự động sử dụng Transformer (PhoBERT).
   - Tích hợp giải thích mô hình (XAI) và cơ chế đối chiếu bằng chứng (Fact-checking).
4. **Slide 4: Tổng quan kiến trúc hệ thống**  
   - Sơ đồ luồng: Thu thập dữ liệu $\to$ Tiền xử lý $\to$ Feature Extraction $\to$ PhoBERT + Metadata MLP $\to$ Ra quyết định $\to$ Giao diện Web.
5. **Slide 5: Tiền xử lý dữ liệu đặc thù tiếng Việt**  
   - Chuẩn hóa Unicode NFC, xử lý teencode (từ điển 25+ từ viết tắt).
   - **Tách từ tiếng Việt (Word Segmentation)** bằng Underthesea (bắt buộc cho PhoBERT).
6. **Slide 6: Kiến trúc mô hình đề xuất (PhoBERT + Metadata MLP)**  
   - Backbone: PhoBERT-base-v2 (trích xuất vector `[CLS]` 768 chiều).
   - Nhánh Metadata: Mạng MLP (64 $\to$ 32 chiều) mã hóa likes, shares, comments, tick xanh.
   - Tầng Fusion: Nối ghép biểu diễn 800 chiều.
7. **Slide 7: Hàm mất mát Focal Loss**  
   - Công thức Focal Loss giải quyết vấn đề mất cân bằng dữ liệu (Imbalanced Data).
8. **Slide 8: Các mô hình Baseline để so sánh**  
   - Machine Learning: TF-IDF + Logistic Regression, Linear SVM, Random Forest.
   - Deep Learning: BiLSTM + Attention Mechanism, TextCNN.
9. **Slide 9: Bảng so sánh kết quả thực nghiệm**  
   - So sánh các chỉ số Accuracy, Precision, Recall, Macro-F1 giữa 6 mô hình.
   - PhoBERT vượt trội các mô hình truyền thống như thế nào.
10. **Slide 10: Nghiên cứu bóc tách thành phần (Ablation Study)**  
    - Ảnh hưởng của việc Tách từ, bổ sung Metadata và dùng Focal Loss.
11. **Slide 11: Đánh giá độ bất định (Monte Carlo Dropout & OOD)**  
    - Ứng dụng Bayesian Deep Learning để đo độ tin cậy của phán đoán và phát hiện bài viết ngoài phân phối (Entropy).
12. **Slide 12: Cơ chế Giải thích mô hình (XAI) & Đối chiếu bằng chứng**  
    - Word Attribution trực quan hóa các từ ngữ gây ảnh hưởng.
    - So khớp với kho tri thức của Bộ Y Tế, VTV.
13. **Slide 13: Demo giao diện hệ thống**  
    - Hình ảnh 6 phân hệ (Kiểm tra tin, Quét batch CSV, So sánh bài viết, Lịch sử, Chatbot, Admin).
14. **Slide 14: Hạn chế & Hướng phát triển**  
    - Hạn chế: Chưa phân tích ảnh/video đa phương thức.
    - Hướng mở rộng: Tích hợp Vision Transformer (ViT) và Graph Neural Network (GNN).
15. **Slide 15: Lời cảm ơn & Hỏi đáp (Q&A)**  
    - Cảm ơn thầy cô và sẵn sàng trả lời câu hỏi phản biện.

