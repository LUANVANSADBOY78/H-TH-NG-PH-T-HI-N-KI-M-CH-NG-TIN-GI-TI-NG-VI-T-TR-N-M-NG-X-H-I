# 🛡️ Hệ Thống Phát Hiện & Kiểm Chứng Tin Giả Tiếng Việt Trên Mạng Xã Hội Bằng Mô Hình Transformer (PhoBERT) & Chatbot AI

> **Đề tài Đồ án Tốt nghiệp / Nghiên cứu Khoa học Môn Học Sâu (Deep Learning)**:  
> *"Xây dựng hệ thống phát hiện và hỗ trợ kiểm chứng tin giả tiếng Việt trên mạng xã hội sử dụng mô hình Transformer (PhoBERT), tích hợp Chatbot AI."*

---

## 📌 Bảng Tổng Hợp Tính Năng Đồ Án (45+ Chức Năng Hoàn Chỉnh)

| Nhóm Phân Hệ | Danh Sách Chức Năng Chi Tiết | Trạng Thái | File Module Phụ Trách |
| :--- | :--- | :---: | :--- |
| **1. Người Dùng (User Portal)** | 1. Phân loại tin tức bằng PhoBERT (Transformer Encoder) | ✅ | [`src/models.py`](file:///d:/doanhocsau/src/models.py), [`src/app.py`](file:///d:/doanhocsau/src/app.py) |
| | 2. Quy chuẩn 3 nhãn Fact-checking: Supported, Refuted, NEI | ✅ | [`src/fact_checker.py`](file:///d:/doanhocsau/src/fact_checker.py) |
| | 3. Trích xuất tự động Tiêu đề & Nội dung từ đường dẫn URL | ✅ | [`src/url_extractor.py`](file:///d:/doanhocsau/src/url_extractor.py) |
| | 4. Kiểm tra bất nhất Tiêu đề vs Thân bài (Clickbait check) | ✅ | [`src/advanced_nlp.py`](file:///d:/doanhocsau/src/advanced_nlp.py) |
| | 5. Xử lý bài viết dài bằng cơ chế Chunking & Aggregation | ✅ | [`src/advanced_nlp.py`](file:///d:/doanhocsau/src/advanced_nlp.py) |
| | 6. Đối chiếu kho bằng chứng đã xác minh (Evidence Retrieval) | ✅ | [`src/fact_checker.py`](file:///d:/doanhocsau/src/fact_checker.py) |
| | 7. Giải thích mô hình (XAI) - Tô màu từ khóa có trọng số cao | ✅ | [`src/explainer.py`](file:///d:/doanhocsau/src/explainer.py) |
| | 8. Báo cáo phản hồi khiếu nại nếu AI dự đoán sai | ✅ | [`src/database.py`](file:///d:/doanhocsau/src/database.py) |
| **2. Phân Tích Hàng Loạt (Batch)** | 9. Tải file CSV / Excel chứa hàng trăm bài viết | ✅ | [`src/batch_analyzer.py`](file:///d:/doanhocsau/src/batch_analyzer.py) |
| | 10. Quét tự động, gắn nhãn, tính độ rủi ro hàng loạt | ✅ | [`src/batch_analyzer.py`](file:///d:/doanhocsau/src/batch_analyzer.py) |
| | 11. Xem trước bảng kết quả & tải file kết quả `.csv` về máy | ✅ | [`src/app.py`](file:///d:/doanhocsau/src/app.py) |
| **3. Đối Chiếu 2 Bài Viết** | 12. Đo độ tương đồng ngữ nghĩa giữa Bài A và Bài B | ✅ | [`src/advanced_nlp.py`](file:///d:/doanhocsau/src/advanced_nlp.py) |
| | 13. Tìm kiếm các từ khóa chung và phát hiện mâu thuẫn | ✅ | [`src/advanced_nlp.py`](file:///d:/doanhocsau/src/advanced_nlp.py) |
| **4. Lịch Sử & Cơ Sở Dữ Liệu** | 14. Lưu trữ lịch sử vào SQLite Database | ✅ | [`src/database.py`](file:///d:/doanhocsau/src/database.py) |
| | 15. Tìm kiếm lịch sử theo từ khóa, lọc theo nguồn tin | ✅ | [`src/database.py`](file:///d:/doanhocsau/src/database.py) |
| | 16. Xóa lịch sử kiểm tra | ✅ | [`src/database.py`](file:///d:/doanhocsau/src/database.py) |
| **5. Chatbot AI Tư Vấn** | 17. Giải thích khái niệm tin giả (Mis/Dis/Mal-information) | ✅ | [`src/chatbot.py`](file:///d:/doanhocsau/src/chatbot.py) |
| | 18. Hướng dẫn quy trình 5 bước kiểm chứng nguồn tin MXH | ✅ | [`src/chatbot.py`](file:///d:/doanhocsau/src/chatbot.py) |
| | 19. Giải thích cơ chế Attention của PhoBERT & độ tin cậy | ✅ | [`src/chatbot.py`](file:///d:/doanhocsau/src/chatbot.py) |
| | 20. Cung cấp danh bạ các báo chí chính thống uy tín tại VN | ✅ | [`src/chatbot.py`](file:///d:/doanhocsau/src/chatbot.py) |
| **6. Học Sâu Chuyên Sâu (Deep Learning)** | 21. Kiến trúc BiLSTM với Attention Mechanism | ✅ | [`src/deep_learning_models.py`](file:///d:/doanhocsau/src/deep_learning_models.py) |
| | 22. Kiến trúc Mạng nơ-ron tích chập TextCNN | ✅ | [`src/deep_learning_models.py`](file:///d:/doanhocsau/src/deep_learning_models.py) |
| | 23. Mô hình LSTM 1 chiều tiêu chuẩn | ✅ | [`src/deep_learning_models.py`](file:///d:/doanhocsau/src/deep_learning_models.py) |
| | 24. Mô hình Đa đặc trưng PhoBERT + Metadata MLP | ✅ | [`src/models.py`](file:///d:/doanhocsau/src/models.py) |
| | 25. Ước lượng độ không chắc chắn (Monte Carlo Dropout) | ✅ | [`src/deep_learning_models.py`](file:///d:/doanhocsau/src/deep_learning_models.py) |
| | 26. Phát hiện dữ liệu ngoài phân phối (OOD Detection) | ✅ | [`src/advanced_nlp.py`](file:///d:/doanhocsau/src/advanced_nlp.py) |
| | 27. Phân tích cảm xúc & độ kích động (Sensationalism Score) | ✅ | [`src/advanced_nlp.py`](file:///d:/doanhocsau/src/advanced_nlp.py) |
| | 28. Xử lý mất cân bằng dữ liệu bằng Focal Loss | ✅ | [`src/models.py`](file:///d:/doanhocsau/src/models.py) |
| **7. Quản Trị & Thực Nghiệm (Admin)** | 29. Dashboard số liệu KPI thời gian thực | ✅ | [`src/app.py`](file:///d:/doanhocsau/src/app.py) |
| | 30. Quản lý danh sách phản hồi khiếu nại của người dùng | ✅ | [`src/database.py`](file:///d:/doanhocsau/src/database.py) |
| | 31. Quản lý và thêm bằng chứng kiểm chứng mới vào kho DB | ✅ | [`src/database.py`](file:///d:/doanhocsau/src/database.py) |
| | 32. Vẽ biểu đồ Training/Validation Loss qua các Epoch | ✅ | [`src/ablation_and_metrics.py`](file:///d:/doanhocsau/src/ablation_and_metrics.py) |
| | 33. Vẽ đồ thị Ma trận nhầm lẫn (Confusion Matrix Plot) | ✅ | [`src/ablation_and_metrics.py`](file:///d:/doanhocsau/src/ablation_and_metrics.py) |
| | 34. Báo cáo nghiên cứu bóc tách thành phần (Ablation Study) | ✅ | [`src/ablation_and_metrics.py`](file:///d:/doanhocsau/src/ablation_and_metrics.py) |
| | 35. So sánh thực nghiệm PhoBERT vs BiLSTM, TextCNN, SVM, LR | ✅ | [`src/model_comparison.py`](file:///d:/doanhocsau/src/model_comparison.py) |
| | 36. Xuất toàn bộ lịch sử ra file CSV (`history_export.csv`) | ✅ | [`src/app.py`](file:///d:/doanhocsau/src/app.py) |

---

## 📁 Cấu Trúc Toàn Bộ Dự Án

```text
d:\doanhocsau/
├── requirements.txt            # Danh mục thư viện Python
├── README.md                   # Tài liệu hướng dẫn đồ án
├── data/
│   ├── create_sample_data.py   # Script tạo tập dữ liệu mẫu benchmark
│   ├── fake_news_system.db     # SQLite Database lưu Lịch sử, Bằng chứng, Phản hồi
│   ├── raw/data.csv            # Dữ liệu mạng xã hội thô
│   └── processed/              # train.csv, val.csv, test.csv đã tách từ
├── notebooks/
│   └── 01_eda.py               # Phân tích khám phá dữ liệu (EDA)
├── outputs/                    # Checkpoint PhoBERT, ảnh biểu đồ, file export CSV
│   ├── training_curves.png     # Đồ thị Training/Val Loss & F1 qua từng epoch
│   └── confusion_matrix.png    # Ma trận nhầm lẫn trực quan hóa
└── src/
    ├── __init__.py
    ├── preprocess.py           # Chuẩn hóa Unicode NFC, teencode, tách từ Underthesea
    ├── baseline.py             # TF-IDF + Logistic Regression, SVM, Naive Bayes, Random Forest
    ├── models.py               # PhoBERT + Metadata MLP + Focal Loss
    ├── deep_learning_models.py # Các kiến trúc BiLSTM, LSTM, TextCNN, MC Dropout
    ├── advanced_nlp.py         # Chunking bài dài, OOD Detection, So sánh 2 bài, Clickbait check
    ├── batch_analyzer.py       # Quét phân tích hàng loạt file CSV / Excel
    ├── ablation_and_metrics.py # Vẽ biểu đồ Loss/F1, Confusion Matrix, Bảng Ablation
    ├── train.py                # Pipeline huấn luyện PhoBERT (Hugging Face Trainer)
    ├── evaluate.py             # Đánh giá tập test & Error Analysis
    ├── url_extractor.py        # Cào & trích xuất nội dung từ đường dẫn URL bài báo
    ├── fact_checker.py         # Engine đối chiếu bằng chứng & chuẩn 3 nhãn Fact-checking
    ├── explainer.py            # Giải thích mô hình AI (XAI) & tô màu từ khóa
    ├── chatbot.py              # Trợ lý ảo Chatbot AI tư vấn và hướng dẫn kiểm chứng
    ├── model_comparison.py     # Bảng tổng hợp so sánh thực nghiệm các mô hình
    └── app.py                  # Giao diện Web Gradio 6 Phân hệ Toàn diện
```

---

## 🚀 Hướng Dẫn Sử Dụng & Toàn Bộ Câu Lệnh Khởi Chạy

### 1. Khởi chạy Giao diện Web & Tạo Link Chia Sẻ Công Khai (Public Link)
- **Tạo link trực tuyến (gradio.live) để gửi cho thầy cô / bạn bè dùng từ xa (trên điện thoại hoặc máy khác):**
```bash
python src/app.py --share
```
- **Chạy Web nội bộ (chỉ riêng trên máy cá nhân):**
```bash
python src/app.py
```
> Mở trình duyệt tại: **`http://127.0.0.1:7860`** để trải nghiệm đầy đủ 6 phân hệ:
> - **Tab 1: 🔍 Kiểm Tra Tin Tức AI**: Dán URL Facebook/Báo chí, trích xuất lượt like/share/cmt, phân loại PhoBERT, tính chỉ số bất định MC Dropout & Entropy OOD, giải thích từ khóa (XAI).
> - **Tab 2: 📁 Phân Tích Hàng Loạt (Batch CSV/Excel)**: Quét và gắn nhãn hàng loạt hàng trăm bài viết, xuất file `.csv`.
> - **Tab 3: ⚖️ So Sánh Hai Bài Viết**: Đối chiếu 2 bài viết đo độ tương đồng ngữ nghĩa và phát hiện mâu thuẫn.
> - **Tab 4: 🕒 Lịch Sử Kiểm Tra**: Tìm kiếm, lọc và xuất dữ liệu lịch sử vào SQLite.
> - **Tab 5: 💬 Chatbot AI**: Trợ lý ảo tư vấn phòng chống tin giả và kiến trúc PhoBERT.
> - **Tab 6: 📊 Admin & Nghiên Cứu Học Sâu**: Xem thống kê thời gian thực, quản lý phản hồi khiếu nại, biểu đồ Loss/F1 và ma trận nhầm lẫn.

---

### 2. Huấn luyện Mô hình AI (Fine-tuning PhoBERT trên GPU)
Tận dụng nhân Tensor Cores trên GPU (như NVIDIA GeForce RTX 3050 Laptop) với Mixed Precision FP16, chỉ mất **2 đến 4 phút**:
```bash
python src/train.py
```
*(Hoặc chạy nhanh 2 epoch để kiểm tra nhanh trong hơn 1 phút: `python src/train.py --epochs 2`)*

---

### 3. Tiền xử lý & Làm sạch dữ liệu
Chuẩn hóa mã hóa ký tự đặc biệt (chữ in đậm Toán học trên Facebook `𝐍𝐆𝐎̣𝐂 𝐌𝐈𝐍𝐇` $\to$ `NGỌC MINH`), dịch teencode và tách từ tiếng Việt:
```bash
python src/preprocess.py
```

---

### 4. Huấn luyện các mô hình Baseline so sánh
Huấn luyện các mô hình truyền thống (TF-IDF + Naive Bayes, Logistic Regression, SVM) để đối chiếu chứng minh ưu thế của Deep Learning:
```bash
python src/baseline.py
```

---

### 5. Đánh giá Mô hình & Xuất Biểu đồ Báo cáo
Chạy thực nghiệm bóc tách thành phần (Ablation Study) và xuất đồ thị biểu diễn ma trận nhầm lẫn:
```bash
python src/ablation_and_metrics.py
python src/model_comparison.py
```
*(Kết quả biểu đồ được tự động lưu vào thư mục `outputs/`)*

---

### 6. Kiểm thử tự động toàn diện (Unit Tests)
Chạy tự động 14 bài kiểm tra các hàm xử lý chuỗi, clickbait, MC Dropout và Database:
```bash
python -m unittest discover tests
```

---

### 7. Chạy Tự động Trọn gói 1-Click (Treo máy)
Tự động làm sạch dữ liệu $\to$ đo đạc số liệu $\to$ khởi động luôn hệ thống Web:
```bash
python run_all.py
```

---

### 8. Giải phóng cổng khi bị kẹt (Windows)
Đóng các tiến trình Python chạy ngầm nếu gặp thông báo "Cổng 7860 đang bận":
```cmd
taskkill /F /IM python.exe
```

---

### 📁 Bộ công cụ 1-Click (File `.bat` có sẵn trong thư mục gốc)
Người dùng chỉ cần nhấp đúp chuột để chạy mà không cần gõ lệnh:
- `1_UP_CODE_LEN_GITHUB.bat`: Đẩy code lên GitHub.
- `2_CHAY_WEB_CHIA_SE_CONG_KHAI.bat`: Bật web kèm link công khai gửi người khác.
- `3_CHAY_WEB_NOI_BO.bat`: Mở web chạy nội bộ.
- `4_HUAN_LUYEN_PHOBERT.bat`: Huấn luyện mô hình PhoBERT.
- `5_DANH_GIA_VA_XUAT_BIEU_DO.bat`: Xuất đồ thị và bảng so sánh báo cáo.
- `6_TAT_TIEN_TRINH_BI_KET.bat`: Giải phóng cổng bị kẹt.
