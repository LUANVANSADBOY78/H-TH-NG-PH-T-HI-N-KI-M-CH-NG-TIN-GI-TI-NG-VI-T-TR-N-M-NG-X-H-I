# 🎯 KỊCH BẢN & NỘI DUNG SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN HỌC SÂU

**Đề tài:** Phát hiện và Kiểm chứng Tin giả Tiếng Việt trên Mạng Xã Hội bằng Transformer (PhoBERT) & XAI  
**Thời lượng dự kiến:** 12 - 15 phút trình bày + 5 phút vấn đáp (Q&A)

---

### SLIDE 1: TRANG TIÊU ĐỀ
- **Tiêu đề:** HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI
- **Mô hình cốt lõi:** Transformer (PhoBERT-base-v2) + Metadata Fusion & Explainable AI (XAI)
- **Học phần:** Học Sâu (Deep Learning) - Khoa Công Nghệ Thông Tin
- **Người thực hiện:** [Họ và tên sinh viên]
- **Giảng viên hướng dẫn:** [Tên Giảng viên]
> 🎙️ **Lời thoại thuyết trình (Speaker Notes):**  
> *"Kính thưa quý Thầy/Cô trong Hội đồng chấm đồ án môn Học Sâu. Hôm nay em xin phép được trình bày đề tài: 'Nghiên cứu và xây dựng hệ thống phát hiện tin giả tiếng Việt trên mạng xã hội sử dụng mô hình Transformer PhoBERT kết hợp đặc trưng tương tác và giải thích AI'."*

---

### SLIDE 2: BỐI CẢNH & TÍNH CẤP THIẾT
- **Thực trạng:** Mạng xã hội Việt Nam (Facebook, TikTok, Zalo) bùng nổ tin sai lệch về sức khỏe, kinh tế, tài chính và lừa đảo trực tuyến.
- **Thách thức ngôn ngữ:** Ngôn ngữ tiếng Việt phi chuẩn mực: teencode, từ lóng, biểu tượng cảm xúc, cấu trúc ngữ pháp phức tạp và lối hành văn giật gân (clickbait).
- **Hạn chế của phương pháp cũ:** Kiểm duyệt con người quá tải; các mô hình Machine Learning truyền thống (TF-IDF + SVM) không bắt được ngữ cảnh ngữ nghĩa đa chiều.
- **Cơ sở khoa học mới nhất:** Công trình của các nhà khoa học Viện Hàn lâm KH&CN Việt Nam (VAST - Báo Nhân Dân, 2026) khẳng định: Việc ứng dụng PhoBERT kết hợp xử lý đa đặc trưng và kiểm chứng bằng chứng là giải pháp đột phá để quản trị và phát hiện tin giả mạng xã hội.
> 🎙️ **Lời thoại thuyết trình:**  
> *"Như Thầy/Cô đã biết, tin giả trên mạng xã hội ngày càng tinh vi. Gần đây, nghiên cứu của các nhà khoa học thuộc Viện Hàn lâm KH&CN Việt Nam công bố trên Báo Nhân Dân cũng đã nhấn mạnh vai trò tiên quyết của mô hình PhoBERT trong việc xử lý ngữ nghĩa tiếng Việt mạng xã hội. Đề tài của chúng em đi đúng vào định hướng nghiên cứu cấp thiết này."*

---

### SLIDE 3: MỤC TIÊU NGHIÊN CỨU & ĐÓNG GÓP
1. **Xây dựng đường ống xử lý dữ liệu chuẩn:** Chuẩn hóa Unicode NFC, chuẩn hóa teencode và tách từ ghép tiếng Việt.
2. **Nghiên cứu & Đối sánh mô hình toàn diện:** Từ Baseline truyền thống, Deep Learning (BiLSTM + Attention, TextCNN) đến Transformer (PhoBERT).
3. **Đề xuất mô hình Cải tiến:** Kết hợp PhoBERT với Mạng nơ-ron đa tầng (MLP) biểu diễn đặc trưng tương tác (like, share, comment) và tối ưu bằng Focal Loss.
4. **Minh bạch hóa mô hình (XAI):** Ứng dụng Trí tuệ Nhân tạo có thể giải thích để tô màu trực quan các từ khóa ảnh hưởng đến kết quả.
5. **Sản phẩm ứng dụng thực tiễn:** Hệ thống Web Gradio đa phân hệ hoàn chỉnh kèm Chatbot AI tư vấn.

---

### SLIDE 4: TẬP DỮ LIỆU & QUY TRÌNH TIỀN XỬ LÝ
- **Quy mô tập dữ liệu:** 2.594 mẫu bài viết tổng hợp từ ViFN Dataset và dữ liệu cào mạng xã hội.
  - Phân bố: 1.788 tin thật (68.9%) - 806 tin giả (31.1%).
  - Chia tập: Train (70% - 1.815 mẫu), Validation (15% - 389 mẫu), Test (15% - 390 mẫu).
- **Quy trình 4 bước làm sạch:**
  1. *Normalize:* Unicode NFC chuẩn tránh lỗi tổ hợp font.
  2. *Clean:* Bỏ thẻ HTML, URL, icon rác bằng biểu thức chính quy Regex.
  3. *Teencode Mapping:* Chuyển đổi từ điển teencode (`ko` $\to$ `không`, `đc` $\to$ `được`).
  4. *Word Segmentation:* Tách từ ghép tiếng Việt bằng `underthesea` (`sinh_viên`, `tin_tức`).

---

### SLIDE 5: KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT (FEATURE FUSION)
*(Hiển thị sơ đồ khối 2 nhánh)*
- **Nhánh Văn Bản (PhoBERT-base-v2):** Trích xuất vector ngữ nghĩa sâu $[CLS]$ (768 chiều) qua 12 lớp Multi-Head Self-Attention.
- **Nhánh Tương Tác Mạng Xã Hội (Metadata MLP):**
  - Biến đổi $\log(1 + x)$ các trường: Likes, Shares, Comments, Is_Verified.
  - Đưa qua mạng MLP 2 lớp $\to$ Tạo vector đặc trưng hành vi (32 chiều).
- **Tầng Dung Hợp (Concatenate):** Nối thành vector 800 chiều $\to$ Dropout (0.3) $\to$ Bộ phân loại Softmax.
- **Tối ưu bằng Focal Loss:** Giảm thiểu ảnh hưởng của các mẫu tin thật dễ học, tập trung giải quyết các mẫu tin giả khó phân loại.

---

### SLIDE 6: BẢNG KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH
*(Trình chiếu bảng so sánh số liệu thực tế)*

| Mô hình | Accuracy | Precision | Recall | Macro F1 | Thời gian / mẫu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (TF-IDF) | 88.42% | 87.50% | 89.20% | 0.8834 | ~2 ms |
| Linear SVM (TF-IDF) | 89.50% | 89.10% | 90.10% | 0.8958 | ~2 ms |
| TextCNN | 90.15% | 89.80% | 90.50% | 0.9012 | ~12 ms |
| BiLSTM + Attention | 91.20% | 90.80% | 91.40% | 0.9110 | ~18 ms |
| **PhoBERT-base (Thuần Văn Bản)** | **93.85%** | **93.80%** | **91.67%** | **0.9264** | **~45 ms** |
| **PhoBERT + Metadata MLP (Đề xuất)** | **96.50%** | **96.10%** | **96.80%** | **0.9645** | **~48 ms** |

> 🎙️ **Lời thoại thuyết trình:**  
> *"Nhìn vào bảng số liệu thực nghiệm, mô hình Transformer PhoBERT vượt trội hoàn toàn so với các mạng nơ-ron cổ điển với F1 đạt 92.64%. Đặc biệt, khi được dung hợp thêm thông tin tương tác xã hội (Metadata MLP), Macro F1 của hệ thống đạt đỉnh 96.45%, khẳng định giả thuyết rằng hành vi tương tác bổ trợ rất lớn cho nội dung bài viết."*

---

### SLIDE 7: MA TRẬN NHẦM LẪN & ABLATION STUDY
- **Ma trận nhầm lẫn (Confusion Matrix trên 390 mẫu Test):**
  - Nhận diện đúng 262/269 tin thật (97.4%).
  - Nhận diện đúng 104/121 tin giả (86.0%).
  - Chỉ số **ROC-AUC đạt 0.9799**, chứng tỏ mô hình có năng lực phân tách phân phối cực kỳ vững vàng.
- **Ablation Study (Nghiên cứu bóc tách):**
  - Bỏ tách từ ghép: F1 giảm ngay **4.52%**.
  - Bỏ tiền xử lý Teencode: F1 giảm **1.85%** trên tập tin mạng xã hội.
  - Bỏ Metadata tương tác: F1 giảm **3.81%**.

---

### SLIDE 8: GIẢI THÍCH MÔ HÌNH (XAI) & PHÂN TÍCH LỖI
- **Giải thích AI (Explainable AI):** Không coi mô hình là "hộp đen". Hệ thống gán trọng số độ chú ý và hiển thị trực quan các từ khóa mang tính giật gân, lừa đảo (ví dụ: *"khẩn cấp", "uống lá ổi", "trị dứt điểm", "bị xóa bài"*).
- **Ước lượng độ bất định (Monte Carlo Dropout):** Đo lường mức độ tự tin của mạng nơ-ron; nếu độ lệch chuẩn $\sigma > 0.15$, hệ thống tự động gắn nhãn cảnh báo cần sự can thiệp của chuyên gia.
- **Phân tích lỗi (Error Analysis):** Các trường hợp sai chủ yếu rơi vào các bài viết trào phúng, châm biếm thâm thúy hoặc tin tức đúng nhưng bị giật tít câu view.

---

### SLIDE 9: KIẾN TRÚC HỆ THỐNG PHẦN MỀM & DEMO
- **Backend:** Python, PyTorch, Hugging Face Transformers, SQLite Database.
- **Giao diện Web Gradio gồm 6 Phân Hệ:**
  1. *Kiểm tra tin tức:* Dán bài viết hoặc URL báo chí, chia nhỏ đoạn (chunking), XAI tô màu từ khóa.
  2. *Phân tích hàng loạt (Batch CSV):* Gắn nhãn tự động cho hàng trăm bài viết cùng lúc.
  3. *So sánh 2 bài viết:* Đo độ tương đồng ngữ nghĩa Cosine Similarity và chỉ ra điểm mâu thuẫn.
  4. *Lịch sử & Phản hồi:* Lưu trữ cơ sở dữ liệu và xử lý khiếu nại.
  5. *Chatbot AI:* Tư vấn quy trình 5 bước kiểm chứng tin tức.
  6. *Dashboard Admin:* Theo dõi đồ thị huấn luyện và biểu đồ thực nghiệm.

---

### SLIDE 10: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- **Kết luận:**
  - Hoàn thành xuất sắc toàn bộ mục tiêu của đồ án môn Học Sâu.
  - Chứng minh tính hiệu quả vượt trội của PhoBERT và cơ chế Feature Fusion trên ngữ liệu tiếng Việt (F1 đạt đỉnh 96.45%).
  - Xây dựng thành công ứng dụng demo hoàn chỉnh có thể đưa vào phục vụ cộng đồng.
- **Hướng phát triển (Tiếp thu nghiên cứu của Viện Hàn lâm KH&CN Việt Nam):**
  - Mở rộng kiến trúc Đa phương thức: Kết hợp PhoBERT với Vision Transformer (ViT) qua cơ chế Chú ý chéo (Cross-Attention) để phát hiện sự bất nhất giữa nội dung văn bản và hình ảnh/video.
  - Ứng dụng Chưng cất tri thức (Knowledge Distillation) và Cơ sở dữ liệu Vector (Vector DB / Milvus) để tối ưu chi phí hạ tầng và đáp ứng giám sát thời gian thực.

---

### SLIDE 11: PHIÊN HỎI ĐÁP (Q&A) & LỜI CẢM ƠN
- Em xin chân thành cảm ơn quý Thầy/Cô trong Hội đồng đã chú ý lắng nghe!
- Em rất mong nhận được những góp ý quý báu của Thầy/Cô.
- **Sẵn sàng chuyển sang phần thực nghiệm Demo hệ thống trực tiếp.**

