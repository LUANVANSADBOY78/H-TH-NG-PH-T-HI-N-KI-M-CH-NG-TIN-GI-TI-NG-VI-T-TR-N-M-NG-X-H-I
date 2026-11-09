# 📑 ĐỀ CƯƠNG BÁO CÁO ĐỒ ÁN MÔN HỌC SÂU (DEEP LEARNING)
**Tên đề tài**: Nghiên cứu và Xây dựng Hệ thống Phát hiện Tin giả Tiếng Việt Trên Mạng Xã Hội Sử dụng Mô hình Transformer (PhoBERT) Kết hợp Đặc trưng Tương tác.

---

## MỤC LỤC CHI TIẾT 5 CHƯƠNG CHUẨN KHOA HỌC

### LỜI MỞ ĐẦU
1. Lý do chọn đề tài (Tính cấp thiết trong bối cảnh mạng xã hội Việt Nam bùng nổ thông tin sai lệch).
2. Mục tiêu nghiên cứu.
3. Đối tượng và phạm vi nghiên cứu.
4. Bố cục của đồ án.

---

### CHƯƠNG 1: TỔNG QUAN VỀ BÀI TOÁN PHÁT HIỆN TIN GIẢ VÀ CÁC CÔNG TRÌNH LIÊN QUAN
#### 1.1. Định nghĩa và Phân loại Tin giả
- Định nghĩa *Misinformation*, *Disinformation*, *Malinformation*.
- Đặc trưng của tin giả trên mạng xã hội Việt Nam: văn phong giật gân, sử dụng teencode, câu tương tác like/share, mạo danh cơ quan chức năng.
#### 1.2. Các hướng tiếp cận bài toán Fact-Checking
- Hướng tiếp cận dựa trên nội dung văn bản (Content-based).
- Hướng tiếp cận dựa trên đặc trưng người dùng và mạng xã hội (Social context).
- Quy chuẩn đánh giá 3 nhãn Fact-checking: *Supported*, *Refuted*, *Not Enough Information (NEI)*.
#### 1.3. Khảo sát các công trình liên quan
- Các cuộc thi và tập dữ liệu tại Việt Nam: VLSP 2020 (ReINTEL), ViFactCheck, ISE-DSC01.
- Xu hướng chuyển dịch từ Machine Learning truyền thống sang Transformer SOTA.

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VỀ MÔ HÌNH HỌC SÂU VÀ TRANSFORMER
#### 2.1. Quá trình phát triển các kiến trúc Deep Learning trong NLP
- Mô hình mạng hồi quy LSTM và **Bidirectional LSTM (BiLSTM)**.
- Cơ chế **Attention Mechanism** (Bahdanau et al.).
- Mạng nơ-ron tích chập trong xử lý văn bản (**TextCNN**).
#### 2.2. Kiến trúc Transformer và Mô hình PhoBERT
- Kiến trúc Transformer Encoder: Cơ chế Multi-Head Self-Attention, Positional Encoding, Feed-Forward Network.
- Mô hình ngôn ngữ tiền huấn luyện **PhoBERT-base-v2**:
  - Huấn luyện dựa trên RoBERTa trên 20GB ngữ liệu tiếng Việt.
  - Tầm quan trọng bắt buộc của bước **Tách từ tiếng Việt (Word Segmentation)** bằng Underthesea/RDRSegmenter để tạo các từ ghép (`học_sâu`, `bộ_y_tế`).
#### 2.3. Các kỹ thuật Học sâu Nâng cao
- **Focal Loss**: Giảm trọng số các mẫu dễ, tập trung học các mẫu phân loại khó:
  $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- **Monte Carlo Dropout**: Kỹ thuật Bayesian Deep Learning ước lượng độ bất định ($\sigma$).
- **Độ hỗn loạn Entropy**: Phát hiện dữ liệu ngoài phân phối (Out-of-Distribution - OOD):
  $$H(p) = -\sum_{i} p_i \log_2(p_i)$$

---

### CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT VÀ KIẾN TRÚC HỆ THỐNG
#### 3.1. Quy trình tổng thể (End-to-End Pipeline)
- Sơ đồ khối kiến trúc hệ thống từ Thu thập $\to$ Tiền xử lý $\to$ Feature Fusion $\to$ Đánh giá $\to$ Web Interface.
#### 3.2. Tiền xử lý dữ liệu đặc thù tiếng Việt
- Chuẩn hóa Unicode NFC.
- Làm sạch URL, mã HTML, số điện thoại rác.
- Bộ từ điển chuẩn hóa Teencode và tiếng lóng mạng xã hội.
- Tách từ tiếng Việt (Word Segmentation).
#### 3.3. Kiến trúc Đa đặc trưng: PhoBERT + Metadata MLP (Feature Fusion)
- Nhánh văn bản: PhoBERT trích xuất vector biểu diễn toàn câu $[CLS]$ (768 chiều).
- Nhánh tương tác: Mạng MLP biến đổi đặc trưng số (log(likes), log(shares), log(comments), is_verified) thành vector 32 chiều.
- Tầng phân loại: Nối kết hợp ($768 + 32 = 800$ chiều) $\to$ Dropout $\to$ Linear Classifier.
#### 3.4. Cơ chế Giải thích Mô hình (XAI) và Đối chiếu Bằng chứng (Fact-Checking)
- Trích xuất từ ngữ có trọng số cao (Word Attribution) và phân loại ngữ khí giật gân (Clickbait Score).
- Engine truy xuất bằng chứng (Evidence Retrieval) từ kho tri thức của các cơ quan chính thống (Bộ Y Tế, VTV, TTXVN).

---

### CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
#### 4.1. Môi trường thực nghiệm và Siêu tham số
- Thiết bị phần cứng, phiên bản PyTorch, Transformers, Batch Size: 16, Learning Rate: 2e-5, Optimizer: AdamW, Early Stopping: 2 epochs.
#### 4.2. Bảng kết quả so sánh các mô hình
- So sánh chi tiết giữa:
  1. *Logistic Regression (TF-IDF)*
  2. *Linear SVM (TF-IDF)*
  3. *Random Forest (TF-IDF)*
  4. *BiLSTM + Attention*
  5. *TextCNN*
  6. *PhoBERT (Thuần văn bản)*
  7. *PhoBERT + Metadata MLP (Mô hình đề xuất - F1: 96.45%)*
#### 4.3. Nghiên cứu bóc tách thành phần (Ablation Study)
- Đánh giá vai trò khi bỏ tách từ (F1 giảm 5.3%), bỏ metadata (F1 giảm 2.3%), bỏ Focal Loss (F1 giảm 1.8%).
#### 4.4. Phân tích tiến trình huấn luyện và Phân tích lỗi (Error Analysis)
- Đồ thị Loss Curves và Macro-F1 curves qua các Epoch.
- Ma trận nhầm lẫn (Confusion Matrix).
- Phân tích các ca sai sót (False Positives và False Negatives).

---

### CHƯƠNG 5: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG WEB
#### 5.1. Kiến trúc phân hệ ứng dụng
- Phân hệ Người dùng: Kiểm tra bài viết, trích xuất URL, Chatbot AI, Lịch sử kiểm tra (SQLite).
- Phân hệ Phân tích Hàng loạt (Batch CSV).
- Phân hệ Đối chiếu so sánh 2 bài viết (Text Comparator).
- Phân hệ Quản trị viên (Admin Dashboard & KPI Analytics).
#### 5.2. Hình ảnh minh họa giao diện và kiểm thử thực tế

---

### KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
1. Các kết quả đã đạt được của đề tài.
2. Những hạn chế còn tồn tại.
3. Hướng nghiên cứu mở rộng trong tương lai (Mô hình đa phương thức Multimodal ảnh + chữ, Graph Neural Network).

### TÀI LIỆU THAM KHẢO
- *Attention Is All You Need* (Vaswani et al., 2017).
- *PhoBERT: Pre-trained language models for Vietnamese* (Nguyen & Nguyen, 2020).
- *Focal Loss for Dense Object Detection* (Lin et al., 2017).
- Báo cáo hội thảo VLSP 2020 Task ReINTEL.

