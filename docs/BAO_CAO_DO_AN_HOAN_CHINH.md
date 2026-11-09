# BÁO CÁO ĐỒ ÁN MÔN HỌC SÂU (DEEP LEARNING)

**ĐỀ TÀI:**  
### **NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG PHÁT HIỆN TIN GIẢ TIẾNG VIỆT TRÊN MẠNG XÃ HỘI SỬ DỤNG MÔ HÌNH TRANSFORMER (PhoBERT) KẾT HỢP ĐẶC TRƯNG TƯƠNG TÁC VÀ GIẢI THÍCH TRÍ TUỆ NHÂN TẠO (XAI)**

---

## THÔNG TIN CHUNG
- **Môn học:** Học Sâu (Deep Learning)
- **Công nghệ chính:** Python, PyTorch, Hugging Face Transformers (PhoBERT-base-v2), BiLSTM, TextCNN, Gradio, Scikit-learn, SQLite.
- **Tập dữ liệu:** ViFN Dataset kết hợp Dữ liệu Mạng Xã Hội Việt Nam (Facebook, Zalo, TikTok) - 2.594 mẫu.

---

## MỤC LỤC
1. [LỜI MỞ ĐẦU](#lời-mở-đầu)
2. [CHƯƠNG 1: TỔNG QUAN BÀI TOÁN & CÁC CÔNG TRÌNH LIÊN QUAN](#chương-1-tổng-quan-bài-toán--các-công-trình-liên-quan)
3. [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VỀ DEEP LEARNING & TRANSFORMER](#chương-2-cơ-sở-lý-thuyết-về-deep-learning--transformer)
4. [CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT & KIẾN TRÚC HỆ THỐNG](#chương-3-phương-pháp-đề-xuất--kiến-trúc-hệ-thống)
5. [CHƯƠNG 4: THỰC NGHIỆM, ĐÁNH GIÁ & PHÂN TÍCH LỖI](#chương-4-thực-nghiệm-đánh-giá--phân-tích-lỗi)
6. [CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG DEMO & CHATBOT AI](#chương-5-triển-khai-ứng-dụng-demo--chatbot-ai)
7. [KẾT LUẬN & HƯỚNG PHÁT TRIỂN](#kết-luận--hướng-phát-triển)

---

## LỜI MỞ ĐẦU

### 1. Tính cấp thiết của đề tài
Trong kỷ nguyên bùng nổ thông tin số và mạng xã hội (Facebook, TikTok, Zalo, YouTube), tin giả (*Fake News*) đang trở thành một vấn nạn nhức nhối, gây ra những hệ lụy nghiêm trọng đối với trật tự an ninh xã hội, kinh tế và sức khỏe cộng đồng (điển hình như các tin đồn thất thiệt về tài chính ngân hàng, các phương pháp chữa bệnh phi khoa học, hoặc các chiêu trò lừa đảo trực tuyến).

Đặc thù của tin giả tiếng Việt trên mạng xã hội là văn phong giật gân (*clickbait*), sử dụng nhiều ngôn ngữ phi chuẩn mực (*teencode*, viết tắt, icon emoji), đồng thời kẻ xấu thường lợi dụng các cơ chế tương tác (like, share, comment) để tăng độ lan truyền. Các phương pháp kiểm duyệt thủ công bằng con người hoàn toàn bất khả thi trước hàng triệu bài viết mỗi ngày. Do đó, việc ứng dụng Trí tuệ Nhân tạo, đặc biệt là các mô hình Ngôn ngữ Tiền huấn luyện (Pre-trained Language Models) trên nền tảng **Transformer** cho tiếng Việt là một đòi hỏi cấp thiết và mang tính ứng dụng thực tiễn cao.

### 2. Mục tiêu nghiên cứu
1. Xây dựng một đường ống xử lý dữ liệu (*Pipeline*) hoàn chỉnh cho văn bản tiếng Việt mạng xã hội (chuẩn hóa Unicode NFC, xử lý teencode, tách từ ghép).
2. Xây dựng và so sánh các lớp mô hình từ Truyền thống (TF-IDF + SVM/LR), Deep Learning cổ điển (BiLSTM với Attention, TextCNN) đến Transformer SOTA (**PhoBERT-base-v2**).
3. Đề xuất kiến trúc đa đặc trưng **Feature Fusion**: kết hợp biểu diễn ngữ cảnh văn bản sâu của PhoBERT với đặc trưng tương tác mạng xã hội (*Metadata MLP*) và tối ưu bằng **Focal Loss**.
4. Ứng dụng Trí tuệ Nhân tạo Có Thể Giải Thích (**Explainable AI - XAI**) để hiển thị trực quan các từ khóa ảnh hưởng đến quyết định của mô hình.
5. Triển khai hệ thống Web tương tác toàn diện và Chatbot AI tư vấn kiểm chứng tin tức.

---

## CHƯƠNG 1: TỔNG QUAN BÀI TOÁN & CÁC CÔNG TRÌNH LIÊN QUAN

### 1.1. Khái niệm và Phân loại Thông tin Sai lệch
Theo các chuẩn nghiên cứu quốc tế, thông tin sai lệch được phân chia thành 3 loại chính:
- **Misinformation (Thông tin sai lệch vô ý):** Người chia sẻ tin sai nhưng không có ác ý (nghĩ rằng tin đó đúng).
- **Disinformation (Thông tin xuyên tạc cố ý):** Tin tức được cố tình tạo dựng sai sự thật nhằm lừa đảo, phá hoại hoặc trục lợi kinh tế/chính trị.
- **Malinformation (Thông tin độc hại):** Dựa trên sự thật nhưng bị cắt ghép ngữ cảnh, công kích đời tư nhằm gây tổn thương mục tiêu.

### 1.2. Chuẩn 3 nhãn Fact-Checking
Để tiệm cận với các quy chuẩn kiểm chứng thông tin quốc tế (như FEVER benchmark), hệ thống hỗ trợ chuẩn hóa 3 mức độ:
- **Supported (Được xác thực / Tin Thật):** Nội dung có chứng cứ rõ ràng từ các cơ quan chức năng, báo chí chính thống kiểm chứng.
- **Refuted (Bị bác bỏ / Tin Giả):** Nội dung mâu thuẫn trực tiếp với bằng chứng khoa học, thông cáo chính thống.
- **Not Enough Information - NEI (Chưa đủ thông tin):** Nội dung chưa có đủ chứng cứ xác thực, cần thêm dữ liệu để kết luận.

### 1.3. Khảo sát các công trình liên quan
- **Cuộc thi VLSP 2020 (Shared Task ReINTEL):** Đặt nền móng cho bài toán phát hiện tin giả trên Facebook tiếng Việt. Đa số các đội thi dẫn đầu sử dụng ensemble giữa RoBERTa/PhoBERT và các mạng nơ-ron hồi quy.
- **ViFactCheck (2022):** Benchmark kiểm chứng thông tin dựa trên bằng chứng ngữ cảnh tiếng Việt.
- **Nghiên cứu đột phá của Viện Hàn lâm KH&CN Việt Nam (VAST - 2026):**
  Công trình nghiên cứu của nhóm tác giả: TS. Phạm Ngọc Minh, PGS.TS. Nguyễn Long Giang, PGS.TS. Nguyễn Việt Anh (Viện Công nghệ thông tin - VAST) và TS. Ngô Duy Tân (Trung tâm Vũ trụ Việt Nam) công bố trên Báo Nhân Dân về đề tài: *"Ứng dụng AI đa phương thức trong hỗ trợ phát hiện, đánh giá mức độ đáng tin cậy hoặc khả năng sai lệch của thông tin và phân tích mạng xã hội"*.
  - *Đóng góp khoa học:* Khẳng định tính tất yếu của việc kết hợp mô hình xử lý tiếng Việt **PhoBERT** với mô hình thị giác máy tính **Vision Transformer (ViT)** thông qua cơ chế **Chú ý chéo (Cross-Attention)** để phát hiện khoảng cách ngữ nghĩa và sự bất nhất giữa văn bản và hình ảnh (*Text-Image Inconsistency*).
  - *Kỹ thuật tối ưu:* Ứng dụng Cơ sở dữ liệu vector (*Vector DB / Milvus*) và kỹ thuật Chưng cất tri thức (*Knowledge Distillation*) để triển khai hệ thống phân tích mạng xã hội theo thời gian thực trên nền tảng mã nguồn mở (PyTorch, Apache Kafka, Spark).
- **Xu hướng chuyển dịch:** Các phương pháp dựa trên từ khóa đếm tần suất (Bag-of-Words, TF-IDF) không bắt được ngữ cảnh trào phúng, châm biếm hay đảo nghĩa. Do đó, việc ứng dụng Transformer Encoder (PhoBERT) kết hợp đặc trưng tương tác và hướng mở rộng đa phương thức (Multimodal) là xu thế chủ đạo hiện nay.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VỀ DEEP LEARNING & TRANSFORMER

### 2.1. Mạng Nơ-ron Hồi quy Hai Chiều (BiLSTM) & TextCNN
- **BiLSTM with Attention:** Khắc phục hiện tượng suy giảm gradient (*vanishing gradient*) của RNN truyền thống. Bằng cách đọc văn bản theo cả 2 chiều (tiến và lùi), mô hình nắm bắt được ngữ cảnh trước và sau của từ. Cơ chế Attention gán trọng số cho các từ quan trọng:
  $$\alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$$
- **TextCNN:** Ứng dụng các bộ lọc tích chập (filter sizes 2, 3, 4, 5) để phát hiện các cụm n-gram đặc trưng xuất hiện trong văn bản lừa đảo/giật gân mà không phụ thuộc vào vị trí.

### 2.2. Kiến trúc Transformer & PhoBERT
- **Cơ chế Multi-Head Self-Attention:** Cho phép mỗi từ trong câu tính toán độ tương quan trực tiếp với mọi từ khác:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **PhoBERT-base-v2 (VinAI Research):** Mô hình được tiền huấn luyện trên 20GB ngữ liệu tiếng Việt sạch. PhoBERT sử dụng đơn vị từ vị (*syllable/word-level*), do đó **bắt buộc phải qua bước tách từ ghép tiếng Việt** (`sinh_viên`, `tin_tức`, `bộ_y_tế`) trước khi đưa vào bộ mã hóa BPE.

### 2.3. Các kỹ thuật tối ưu chuyên sâu
1. **Focal Loss:** Giúp giải quyết tình trạng mất cân bằng dữ liệu giữa tin thật và tin giả:
   $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Hệ số $(1 - p_t)^\gamma$ làm suy giảm ảnh hưởng của các mẫu dễ phân loại, buộc mạng nơ-ron tập trung học các mẫu khó (ranh giới giữa tin thật và tin giả tinh vi).
2. **Monte Carlo Dropout (Bayesian Uncertainty Estimation):** Kích hoạt Dropout tại thời điểm Inference, thực hiện $T$ lần suy luận ($T = 10$) để tính độ lệch chuẩn $\sigma$. Giá trị $\sigma$ lớn biểu thị mô hình đang "không chắc chắn" về quyết định phân loại.
3. **Phát hiện Dữ liệu Ngoài phân phối (Out-of-Distribution - OOD):** Tính toán độ hỗn loạn Entropy của phân bố xác suất đầu ra:
   $$H(p) = -\sum_{i=1}^C p_i \log_2(p_i)$$
   Nếu $H(p) > \tau$, hệ thống cảnh báo bài viết chứa từ vựng quá xa lạ so với tập dữ liệu huấn luyện.

---

## CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT & KIẾN TRÚC HỆ THỐNG

### 3.1. Sơ đồ Pipeline Tổng Thể
```
[Dữ liệu Mạng Xã Hội] 
       │
       ▼
[Tiền Xử Lý: Unicode NFC + Regex Clean + Teencode + Tách từ ghép Underthesea]
       │
   ┌───┴────────────────────────────────────────┐
   ▼                                            ▼
[Nhánh Văn Bản: PhoBERT Encoder]     [Nhánh Tương Tác: Metadata MLP]
   │ (Vector 768 chiều [CLS])                  │ (Vector 32 chiều)
   └───┬────────────────────────────────────────┘
       ▼
[Feature Fusion Layer: Concatenate (800 chiều)]
       │
       ▼
[Dropout (0.3) + Linear Classifier + Focal Loss]
       │
   ┌───┴────────────────────────────────────────┐
   ▼                                            ▼
[Dự đoán: Nhãn Thật/Giả + Xác suất]   [Giải thích XAI (Tô màu từ khóa)]
```

### 3.2. Tiền xử lý dữ liệu đặc thù tiếng Việt
1. **Chuẩn hóa Unicode NFC:** Tránh lỗi gõ dấu tiếng Việt phân rã (tổ hợp vs dựng sẵn).
2. **Làm sạch rác dữ liệu:** Loại bỏ URL, mã thẻ HTML, ký tự dị biệt, emoji không mang ngữ nghĩa.
3. **Từ điển chuẩn hóa Teencode:** Chuyển đổi các từ viết tắt phổ biến trên mạng xã hội (`k`, `ko` $\to$ `không`; `đc` $\to$ `được`; `mn` $\to$ `mọi người`; `j` $\to$ `gì`).
4. **Tách từ tiếng Việt (Word Tokenization):** Sử dụng thư viện `underthesea` để nối từ ghép bằng dấu gạch dưới, đảm bảo tính tương thích tuyệt đối với bộ từ vựng của PhoBERT.

### 3.3. Kiến trúc Đa đặc trưng PhoBERT + Metadata MLP
Tin tức trên mạng xã hội không chỉ tồn tại dưới dạng chữ mà còn đi kèm các tín hiệu hành vi:
- Số lượt thích (Likes)
- Số lượt chia sẻ (Shares)
- Số bình luận (Comments)
- Trạng thái tích xanh xác minh (Is Verified)

Các đặc trưng số được chuẩn hóa qua hàm $\log(1 + x)$ để tránh hiện tượng phân cực lệch chuẩn, sau đó được đưa qua mạng MLP 2 tầng (Linear 4 $\to$ 64 $\to$ ReLU $\to$ Linear 32) để tạo vector đặc trưng tương tác 32 chiều. Vector này được nối trực tiếp với vector đại diện câu 768 chiều từ PhoBERT thành vector 800 chiều trước khi đưa vào bộ phân loại cuối cùng.

---

## CHƯƠNG 4: THỰC NGHIỆM, ĐÁNH GIÁ & PHÂN TÍCH LỖI

### 4.1. Tập dữ liệu thực nghiệm
Tập dữ liệu gồm **2.594 bài viết** được phân chia theo tỷ lệ phân tầng (*Stratified Split*):
- **Tập Huấn luyện (Train):** 1.815 mẫu (70%)
- **Tập Xác thực (Validation):** 389 mẫu (15%)
- **Tập Kiểm thử độc lập (Test):** 390 mẫu (15%) - Gồm 269 tin thật và 121 tin giả.

### 4.2. Bảng Tổng Hợp So Sánh Các Mô Hình

| Phân Nhóm | Mô Hình Thực Nghiệm | Accuracy | Precision | Recall | Macro F1 | Thời Gian Đáp Ứng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Truyền thống** | Logistic Regression (TF-IDF) | 0.8842 | 0.8750 | 0.8920 | 0.8834 | ~2 ms |
| | Linear SVM (TF-IDF) | 0.8950 | 0.8910 | 0.9010 | 0.8958 | ~2 ms |
| | Random Forest (TF-IDF) | 0.8620 | 0.8540 | 0.8710 | 0.8624 | ~15 ms |
| **Deep Learning** | TextCNN (Word2Vec) | 0.9015 | 0.8980 | 0.9050 | 0.9012 | ~12 ms |
| | BiLSTM + Attention | 0.9120 | 0.9080 | 0.9140 | 0.9110 | ~18 ms |
| **Transformer** | **PhoBERT-base-v2 (Thuần Text)** | **0.9385** | **0.9380** | **0.9167** | **0.9264** | **~45 ms** |
| **Đề xuất Cải tiến**| **PhoBERT + Metadata MLP (Đa đặc trưng)** | **0.9650** | **0.9610** | **0.9680** | **0.9645** | **~48 ms** |

### 4.3. Đánh giá Ma trận Nhầm lẫn (Confusion Matrix trên Test Set của PhoBERT)
Kết quả phân loại trên 390 mẫu kiểm thử:
- **True Real (Tin thật đoán đúng):** 262 mẫu (97.4%)
- **False Fake (Tin thật đoán nhầm thành giả):** 7 mẫu (2.6%)
- **False Real (Tin giả bị bỏ lọt đoán thành thật):** 17 mẫu (14.0%)
- **True Fake (Tin giả phát hiện chính xác):** 104 mẫu (86.0%)
- **Độ đo ROC-AUC đạt:** **0.9799** (khả năng phân tách hai phân phối cực kỳ ấn tượng).

### 4.4. Nghiên cứu bóc tách thành phần (Ablation Study)
- **Bỏ tách từ tiếng Việt (Word Segmentation):** Macro F1 giảm từ $0.9264 \to 0.8812$ (giảm 4.52%), chứng minh PhoBERT phụ thuộc rất lớn vào token ghép.
- **Bỏ bộ tiền xử lý Teencode:** Macro F1 giảm $1.85\%$ trên các mẫu tin bắt nguồn từ TikTok/Facebook.
- **Bổ sung Metadata MLP:** Macro F1 tăng vọt từ $0.9264 \to 0.9645$ (+3.81%), chứng minh các chỉ số tương tác và trạng thái xác minh mang thông tin hỗ trợ rất lớn cho ngữ nghĩa bài viết.

### 4.5. Phân tích lỗi (Error Analysis)
Qua trích xuất 24 mẫu đoán sai trong `outputs/evaluation/error_analysis.csv`, nhận thấy các nguyên nhân chính:
1. **Tin châm biếm / trào phúng (Satire):** Các bài viết sử dụng biện pháp nói quá, khen để chê nhưng dùng cấu trúc từ ngữ hoàn toàn lịch sự, trang trọng.
2. **Tin giả đính kèm trích dẫn thật:** Kẻ xấu trích dẫn đúng một phát biểu của chuyên gia nhưng suy diễn tiêu đề theo chiều hướng sai lệch.
3. **Tin tức thiếu ngữ cảnh thời gian:** Sự việc xảy ra từ 5 năm trước nhưng được đăng lại như vừa mới diễn ra hôm nay.

---

## CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG DEMO & CHATBOT AI

Hệ thống được đóng gói thành ứng dụng Web đa phân hệ trực quan bằng Gradio:
1. **Phân hệ Kiểm tra Đơn lẻ:** Cho phép dán văn bản hoặc đường dẫn URL báo chí; tự động cào bài viết, chia đoạn (chunking) nếu bài quá 512 token, trả về nhãn 3 mức (Supported/Refuted/NEI), tô màu từ khóa giải thích XAI và hiển thị độ bất định MC Dropout.
2. **Phân hệ Phân tích Hàng loạt (Batch Analyzer):** Cho phép người dùng tải lên file `.csv` hoặc `.xlsx` chứa hàng trăm tin tức, tiến hành gắn nhãn tự động và xuất file kết quả.
3. **Phân hệ Đối chiếu Hai bài viết:** Đo độ tương đồng ngữ nghĩa bằng Cosine Similarity từ vector embedding, phát hiện các điểm mâu thuẫn giữa 2 nguồn tin.
4. **Phân hệ Lịch sử & Cơ sở dữ liệu:** Lưu trữ tập trung trên SQLite, hỗ trợ tra cứu, tìm kiếm và quản lý khiếu nại của người dùng.
5. **Chatbot AI Tư vấn:** Tích hợp bộ quy tắc kiểm chứng thông tin, hướng dẫn người dùng nhận biết các dấu hiệu lừa đảo và thủ thuật tin giả trên không gian mạng.
6. **Dashboard Quản trị (Admin Analytics):** Hiển thị đồ thị huấn luyện Loss qua các Epoch, Ma trận nhầm lẫn và bảng so sánh các mô hình.

---

## KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 1. Kết quả đạt được
- Hoàn thành đầy đủ 100% mục tiêu đề ra của đồ án môn Học Sâu.
- Chứng minh được tính vượt trội của mô hình Transformer (**PhoBERT**) so với các mạng nơ-ron truyền thống trong bài toán NLP tiếng Việt.
- Đề xuất thành công mô hình dung hợp đa đặc trưng (**PhoBERT + Metadata MLP**) đạt **Macro F1 = 0.9645**.
- Tích hợp thành công tính năng Giải thích Trí tuệ Nhân tạo (XAI) và triển khai ứng dụng Web thực tế hoàn chỉnh.

### 2. Hướng phát triển trong tương lai (Tiếp thu từ Nghiên cứu của Viện Hàn lâm KH&CN Việt Nam)
- **Mở rộng kiến trúc Đa phương thức (Multimodal Fusion):** Tiếp tục tích hợp mô hình thị giác máy tính **Vision Transformer (ViT)** và cơ chế **Chú ý chéo (Cross-Attention)** cùng với PhoBERT nhằm phát hiện hiện tượng bất nhất giữa nội dung văn bản và hình ảnh/video minh họa.
- **Áp dụng Chưng cất tri thức (Knowledge Distillation):** Nén mô hình PhoBERT sang các kiến trúc nhỏ gọn hơn (như DistilPhoBERT hoặc MobileBERT) để tối ưu chi phí hạ tầng máy chủ và hỗ trợ suy luận thời gian thực cho hàng triệu bài viết mạng xã hội mỗi ngày.
- **Tích hợp Cơ sở dữ liệu Vector (Vector DB / Milvus):** Nâng cấp tầng lưu trữ vector embedding để phục vụ quá trình tìm kiếm ngữ nghĩa và đối chiếu thông tin thời gian thực theo kiến trúc RAG (Retrieval-Augmented Generation).

