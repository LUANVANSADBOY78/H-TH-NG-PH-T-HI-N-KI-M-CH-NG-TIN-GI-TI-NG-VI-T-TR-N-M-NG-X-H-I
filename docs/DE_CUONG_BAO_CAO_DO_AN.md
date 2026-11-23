# 📑 ĐỀ CƯƠNG BÁO CÁO ĐỒ ÁN MÔN HỌC SÂU (DEEP LEARNING)
**Tên đề tài**: Nghiên cứu và Xây dựng Hệ thống Phát hiện Tin giả Tiếng Việt Trên Mạng Xã Hội Sử dụng Mô hình Transformer (PhoBERT-base-v2) Thuần Văn Bản và Giải thích AI (XAI).

---

## MỤC LỤC CHI TIẾT 5 CHƯƠNG CHUẨN KHOA HỌC

### LỜI MỞ ĐẦU
1. Lý do chọn đề tài (Tính cấp thiết trong bối cảnh mạng xã hội Việt Nam bùng nổ thông tin sai lệch).
2. Mục tiêu nghiên cứu và Nguyên tắc liêm chính khoa học (loại bỏ dữ liệu mô phỏng gây rò rỉ).
3. Đối tượng và phạm vi nghiên cứu (2.594 bài viết từ ViFN Dataset và mạng xã hội).
4. Bố cục của đồ án.

---

### CHƯƠNG 1: TỔNG QUAN VỀ BÀI TOÁN PHÁT HIỆN TIN GIẢ VÀ CÁC CÔNG TRÌNH LIÊN QUAN
#### 1.1. Định nghĩa và Phân loại Tin giả
- Định nghĩa *Misinformation*, *Disinformation*, *Malinformation*.
- Đặc trưng của tin giả trên mạng xã hội Việt Nam: văn phong giật gân, sử dụng teencode, mạo danh cơ quan chức năng.
#### 1.2. Chuẩn hóa bài toán phân loại nhị phân
- Nhãn 0: Tin Thật (Real)
- Nhãn 1: Tin Giả (Fake)
#### 1.3. Khảo sát các công trình liên quan
- Các cuộc thi và tập dữ liệu tại Việt Nam: VLSP 2020 (ReINTEL), ViFactCheck, ViFN Ver 3 (Viện Hàn lâm KH&CN).
- Xu hướng chuyển dịch từ Machine Learning truyền thống sang Transformer SOTA ngữ cảnh sâu.

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VỀ MÔ HÌNH HỌC SÂU VÀ TRANSFORMER
#### 2.1. Quá trình phát triển các kiến trúc Deep Learning trong NLP
- Mô hình Machine Learning truyền thống: TF-IDF + Naive Bayes / Logistic Regression / Random Forest / Linear SVM.
- Mô hình mạng hồi quy LSTM và **Bidirectional LSTM (BiLSTM)**.
#### 2.2. Kiến trúc Transformer và Mô hình PhoBERT
- Kiến trúc Transformer Encoder: Cơ chế Multi-Head Self-Attention, Positional Encoding, Feed-Forward Network.
- Mô hình ngôn ngữ tiền huấn luyện **PhoBERT-base-v2**:
  - Huấn luyện dựa trên RoBERTa trên 20GB ngữ liệu tiếng Việt.
  - Tầm quan trọng bắt buộc của bước **Tách từ tiếng Việt (Word Segmentation)** bằng Underthesea/RDRSegmenter để tạo các từ ghép (`học_sâu`, `bộ_y_tế`).
#### 2.3. Các kỹ thuật Học sâu Nâng cao
- **Weighted Cross-Entropy Loss**: Cân bằng trọng số lớp để giải quyết mất cân bằng phân bố mẫu (68.9% thật vs 31.1% giả).
- **Monte Carlo Dropout**: Kỹ thuật Bayesian Deep Learning ước lượng độ bất định ($\sigma$).
- **Explainable AI (XAI)**: Minh bạch hóa dự đoán bằng tô màu trọng số từ vựng kích động hoặc xác thực.

---

### CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT VÀ KIẾN TRÚC HỆ THỐNG
#### 3.1. Quy trình tổng thể (End-to-End Pipeline)
- Sơ đồ khối kiến trúc hệ thống từ Thu thập $\to$ Tiền xử lý $\to$ PhoBERT Encoder $\to$ Weighted Loss $\to$ Đánh giá $\to$ Web Interface.
#### 3.2. Tiền xử lý dữ liệu đặc thù tiếng Việt
- Chuẩn hóa Unicode NFC.
- Làm sạch URL, mã HTML, số điện thoại rác.
- Bộ từ điển chuẩn hóa Teencode và tiếng lóng mạng xã hội (`ko` $\to$ `không`, `đc` $\to$ `được`).
- Tách từ tiếng Việt (Word Segmentation).
#### 3.3. Kiến trúc Mô hình PhoBERT Thuần Văn Bản
- PhoBERT trích xuất vector biểu diễn toàn câu $[CLS]$ (768 chiều).
- Đầu phân loại: Dropout (0.1) $\to$ Linear (768 $\to$ 2) $\to$ Softmax.
- Loại bỏ hoàn toàn siêu dữ liệu mô phỏng để tránh Target Leakage.

---

### CHƯƠNG 4: THỰC NGHIỆM, ĐÁNH GIÁ VÀ PHÂN TÍCH LỖI
#### 4.1. Tập dữ liệu thực nghiệm và Phân bổ
- Tập dữ liệu gồm **2.594 bài viết** (Train 1.815, Val 389, Test **390** bài).
- Phân bố tập Test: **269 tin thật** và **121 tin giả**.
- Baseline đa số (Majority Accuracy): **68.97%**.
#### 4.2. Bảng kết quả so sánh thực nghiệm thực tế (Test Set N=390)
- Naive Bayes (TF-IDF): Acc 86.92%, F1 0.8291
- Logistic Regression (TF-IDF): Acc 87.69%, F1 0.8593
- Random Forest (TF-IDF): Acc 88.97%, F1 0.8657
- Linear SVM (TF-IDF): Acc 90.00%, F1 0.8801
- **PhoBERT-base-v2 (Thuần Văn Bản): Acc 94.62%, Macro F1 0.9354, ROC-AUC 0.9843**.
#### 4.3. Ma trận nhầm lẫn thực nghiệm
- True Real (TN): 264 mẫu (98.14%)
- False Fake (FP): 5 mẫu (1.86%) -> Hàng 0: $264 + 5 = 269$
- False Real (FN): 16 mẫu (13.22%)
- True Fake (TP): 105 mẫu (86.78%) -> Hàng 1: $16 + 105 = 121$
- Tổng hàng khớp tuyệt đối ground truth của tập Test.
#### 4.4. Đánh giá Năng lực Tổng quát hóa & Độ vững chắc
- Che tên cơ quan báo chí & URL: F1 chỉ biến thiên **-0.26%**, chứng minh mô hình hiểu ngữ nghĩa, không phụ thuộc thương hiệu nguồn tin.
- Đánh giá chéo nguồn (ViFN $\to$ Mạng xã hội): F1 giảm **-17.44%** do độ dịch chuyển phong cách viết (Domain Shift).
#### 4.5. Thảo luận chuyên sâu về Rò rỉ Nhãn (Target Leakage)
- Chứng minh việc gán like/share phân tách theo nhãn gây F1 ảo (0.997 - 1.000).
- Quyết định chuẩn mực: chuyển hoàn toàn sang thuần văn bản để đảm bảo liêm chính khoa học.
#### 4.6. Phân tích lỗi (Error Analysis)
- Phân tích chi tiết 21 ca dự đoán sai (16 FN, 5 FP) từ `error_analysis.csv`.

---

### CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG DEMO VÀ CHATBOT AI
#### 5.1. Kiến trúc ứng dụng Web Gradio
- Phân hệ Kiểm tra đơn lẻ (URL & Text + Chunking + XAI + MC Dropout).
- Phân hệ Phân tích hàng loạt (Batch CSV).
- Phân hệ Đối chiếu 2 nguồn tin.
- Chatbot AI tư vấn kiểm chứng.
- Dashboard Thống kê và Quản trị Deep Learning.
#### 5.2. Quản lý dữ liệu và Bảo mật
- Lưu trữ SQLite có phân quyền người dùng.
- Tắt mặc định `share=True`, đóng tiến trình theo PID an toàn.

---

### KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
1. Kết luận: Đạt toàn bộ mục tiêu, số liệu minh bạch, có thể tái lập 100%.
2. Hướng phát triển: Đa phương thức (Multimodal), RAG, Nén mô hình (Knowledge Distillation).
