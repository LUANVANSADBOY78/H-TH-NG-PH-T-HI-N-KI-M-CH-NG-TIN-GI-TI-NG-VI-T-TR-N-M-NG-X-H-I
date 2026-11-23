# 🎯 KỊCH BẢN & NỘI DUNG SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN HỌC SÂU

**Đề tài:** Phát hiện và Kiểm chứng Tin giả Tiếng Việt trên Mạng Xã Hội bằng Transformer (PhoBERT-base-v2) & XAI  
**Thời lượng dự kiến:** 12 - 15 phút trình bày + 5 phút vấn đáp (Q&A)

---

### SLIDE 1: TRANG TIÊU ĐỀ
- **Tiêu đề:** HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI
- **Mô hình cốt lõi:** Transformer (PhoBERT-base-v2 Thuần Văn Bản) + Class-Weighted Loss + Explainable AI (XAI)
- **Học phần:** Học Sâu (Deep Learning) - Khoa Công Nghệ Thông Tin
- **Người thực hiện:** [Họ và tên sinh viên]
- **Giảng viên hướng dẫn:** [Tên Giảng viên]
> 🎙️ **Lời thoại thuyết trình (Speaker Notes):**  
> *"Kính thưa quý Thầy/Cô trong Hội đồng chấm đồ án môn Học Sâu. Hôm nay em xin phép được trình bày đề tài: 'Nghiên cứu và xây dựng hệ thống phát hiện tin giả tiếng Việt trên mạng xã hội sử dụng mô hình Transformer PhoBERT thuần văn bản kết hợp giải thích trí tuệ nhân tạo (XAI)'."*

---

### SLIDE 2: BỐI CẢNH & TÍNH CẤP THIẾT
- **Thực trạng:** Mạng xã hội Việt Nam (Facebook, TikTok, Zalo) bùng nổ tin sai lệch về y tế, tài chính và lừa đảo trực tuyến.
- **Thách thức ngôn ngữ:** Ngôn ngữ tiếng Việt phi chuẩn mực: teencode, từ lóng, lối viết giật gân (*clickbait*), từ ghép có cấu trúc đa âm tiết.
- **Hạn chế của phương pháp cũ:** Kiểm duyệt thủ công quá tải; Machine Learning truyền thống (TF-IDF + SVM) chỉ bắt từ khóa bề mặt, không nắm bắt được ngữ cảnh ngữ nghĩa sâu.
- **Giải pháp:** Sử dụng mô hình Transformer chuyên biệt cho tiếng Việt **PhoBERT-base-v2** để hiểu ngữ cảnh hai chiều của toàn bộ câu văn.

---

### SLIDE 3: MỤC TIÊU NGHIÊN CỨU & NGUYÊN TẮC LIÊM CHÍNH KHOA HỌC
1. **Xây dựng đường ống xử lý dữ liệu chuẩn:** Chuẩn hóa Unicode NFC, ánh xạ teencode và tách từ ghép tiếng Việt bằng `underthesea`.
2. **Nghiên cứu & Đối sánh thực nghiệm minh bạch:** So sánh khách quan giữa mô hình Đa số, Machine Learning truyền thống (Naive Bayes, Logistic Regression, Random Forest, Linear SVM) với Deep Learning Transformer (**PhoBERT-base-v2**).
3. **Phát hiện & Loại bỏ Rò rỉ Nhãn (Target Leakage):** Chứng minh việc dùng metadata mô phỏng sinh theo nhãn dẫn đến F1 ảo (0.99+), kiên quyết tập trung vào **Thuần Văn Bản** để bảo vệ tính liêm chính học thuật.
4. **Kiểm tra độ vững chắc (Robustness):** Đánh giá chéo nguồn (*Cross-domain*) và Che tên cơ quan báo chí (*Source Masking*).
5. **Sản phẩm ứng dụng thực tiễn:** Hệ thống Web Gradio tương tác hoàn chỉnh kèm Chatbot AI tư vấn.

---

### SLIDE 4: TẬP DỮ LIỆU & QUY TRÌNH TIỀN XỬ LÝ
- **Quy mô tập dữ liệu:** **2.594 bài viết** tổng hợp từ ViFN Dataset Ver 3 (Viện Hàn lâm KH&CN) và dữ liệu cào mạng xã hội.
  - Phân bố: 1.788 tin thật (68.9%) - 806 tin giả (31.1%).
  - Phân chia phân tầng (Stratified Split 70/15/15):
    - **Train:** 1.815 mẫu
    - **Val:** 389 mẫu
    - **Test Độc lập:** **390 mẫu** (Gồm đúng **269 tin thật** và **121 tin giả**).
- **Baseline Đa số (Majority Baseline):** Dự đoán tất cả là Thật $\to$ Accuracy đạt **68.97%**.
- **Quy trình tiền xử lý:**
  1. *Unicode NFC:* Tránh lỗi xung đột mã tổ hợp.
  2. *Làm sạch:* Bỏ URL, HTML, icon rác.
  3. *Teencode Mapping:* `ko` $\to$ `không`, `đc` $\to$ `được`, `mn` $\to$ `mọi người`.
  4. *Word Segmentation:* Tách từ ghép bằng `underthesea` (`tin_tức`, `thời_sự`).

---

### SLIDE 5: KIẾN TRÚC MÔ HÌNH PHOBERT THUẦN VĂN BẢN
- **Cốt lõi Transformer (PhoBERT-base-v2):** 12 lớp Transformer Encoder, 768 chiều ẩn, 12 attention heads.
- **Đầu phân loại (Classification Head):** Vector $[CLS]$ 768 chiều $\to$ Dropout (0.1) $\to$ Linear (768 $\to$ 2) $\to$ Softmax.
- **Xử lý mất cân bằng dữ liệu (Weighted CrossEntropy Loss):**
  - Trọng số $w_{\text{real}} = 1.0, \quad w_{\text{fake}} = 2.22$ (nghịch đảo tỷ lệ mẫu).
  - Giúp mô hình không bị "ru ngủ" bởi lớp đa số (tin thật), nâng cao độ nhạy bắt tin giả.

---

### SLIDE 6: BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM THẬT (TEST SET N=390)

| Nhóm Mô Hình | Tên Mô Hình Thực Nghiệm | Accuracy | Macro F1 | Precision | Recall | Nguồn Dữ Liệu |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Đa số** | Predict All Real (Đoán toàn bộ thật) | 68.97% | 0.4082 | 0.3449 | 0.5000 | Phân bố tập Test (269/390) |
| **Truyền thống** | Multinomial Naive Bayes (TF-IDF) | 86.92% | 0.8291 | 0.8946 | 0.8006 | Test Set N=390 |
| | Logistic Regression (TF-IDF) | 87.69% | 0.8593 | 0.8526 | 0.8676 | Test Set N=390 |
| | Random Forest (TF-IDF) | 88.97% | 0.8657 | 0.8851 | 0.8519 | Test Set N=390 |
| | Linear SVM (TF-IDF) | 90.00% | 0.8801 | 0.8917 | 0.8707 | Test Set N=390 |
| **Deep Learning** | **PhoBERT-base-v2 (Thuần Văn Bản)** | **94.62%** | **0.9354** | **0.9487** | **0.9246** | **Checkpoint RTX 3050** |

> 🎙️ **Lời thoại thuyết trình:**  
> *"Nhìn vào bảng số liệu thực nghiệm hoàn toàn khách quan trên 390 mẫu test độc lập, PhoBERT-base-v2 vượt trội rõ rệt: đạt Accuracy 94.62% và Macro F1 93.54%, vượt xa baseline đa số (+25.65%) và vượt mô hình Linear SVM mạnh nhất (+4.62% Accuracy và +5.53% F1). Toàn bộ số liệu này được xuất tự động từ checkpoint huấn luyện thật lưu trong hệ thống."*

---

### SLIDE 7: MA TRẬN NHẦM LẪN & ĐỘ VỮNG CHẮC (ROBUSTNESS)
- **Ma trận nhầm lẫn thực nghiệm trên 390 mẫu Test:**
  $$\begin{bmatrix} 264 & 5 \\ 16 & 105 \end{bmatrix}$$
  - Nhận diện đúng **264 / 269** tin thật (**98.14%**). Chỉ đoán nhầm 5 mẫu.
  - Bắt chính xác **105 / 121** tin giả (**86.78%** Recall, **95.45%** Precision).
  - Tổng các hàng khớp tuyệt đối: $264 + 5 = 269$ và $16 + 105 = 121$.
  - Chỉ số **ROC-AUC đạt 0.9843**.
- **Thực nghiệm Che tên báo & URL (Source Masking):**
  - Che VnExpress, Tuổi Trẻ, Zing, URL $\to$ F1 chỉ giảm nhẹ **0.26%**, chứng minh mô hình hiểu ngữ nghĩa chứ không học vẹt tên báo.
- **Thực nghiệm Đánh giá chéo nguồn (Domain Shift):**
  - Train trên ViFN báo chí $\to$ Test mạng xã hội $\to$ F1 giảm **17.44%**, phản ánh đúng khoảng cách phong cách giữa báo chí và mạng xã hội.

---

### SLIDE 8: BÀI HỌC VỀ RÒ RỈ NHÃN (TARGET LEAKAGE)
- **Thực nghiệm đối chứng:** Khi sinh metadata tương tác (like/share) theo nhãn, mô hình đạt F1 bất thường **0.997 - 1.000**.
- **Bản chất khoa học:** Khoảng chia sẻ của tin giả ($[0.25, 0.75]$) và tin thật ($[0.04, 0.20]$) bị rời rạc hoàn toàn, vô tình đưa "đáp án" vào đầu vào.
- **Quyết định phương pháp luận:** Nhóm nghiên cứu kiên quyết loại bỏ metadata mô phỏng, chỉ giữ mô hình thuần văn bản để đảm bảo **liêm chính học thuật 100%**.

---

### SLIDE 9: GIẢI THÍCH MÔ HÌNH (XAI) & HỆ THỐNG DEMO
- **Minh bạch hóa AI (XAI):** Sử dụng trọng số từ vựng để tô màu trực quan các từ khóa kích động cảm xúc hoặc từ ngữ mang tính kiểm chứng.
- **Ứng dụng Web Gradio hoàn chỉnh:**
  - Kiểm tra đơn lẻ kèm chunking văn bản dài.
  - Phân tích hàng loạt (Batch CSV).
  - Đối chiếu 2 nguồn tin đối lập.
  - Chatbot AI hỗ trợ cộng đồng.
  - Admin Dashboard theo dõi biểu đồ Loss/F1 thực tế.

---

### SLIDE 10: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- **Kết luận:**
  - Hoàn thành xuất sắc toàn bộ mục tiêu đề tài môn Học Sâu.
  - Dữ liệu minh bạch, mã nguồn sạch, số liệu thực nghiệm trung thực và có thể tái lập hoàn toàn.
- **Hướng phát triển:**
  - Nghiên cứu AI đa phương thức (Multimodal Fusion: Văn bản + Hình ảnh) với dữ liệu người dùng thật.
  - Áp dụng kỹ thuật RAG đối chiếu tin tức thời gian thực.
- **Lời cảm ơn:** Em xin chân thành cảm ơn Thầy/Cô đã lắng nghe và rất mong nhận được câu hỏi phản biện từ Hội đồng!
