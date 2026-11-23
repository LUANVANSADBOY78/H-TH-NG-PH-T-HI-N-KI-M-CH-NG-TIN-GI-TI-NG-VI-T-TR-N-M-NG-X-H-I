# 🛡️ Hệ Thống Phát Hiện & Kiểm Chứng Tin Giả Tiếng Việt Trên Mạng Xã Hội Bằng Mô Hình Transformer (PhoBERT) & Chatbot AI

> **Đề tài Đồ án Môn Học Sâu (Deep Learning)**:  
> *"Nghiên cứu và xây dựng hệ thống phát hiện tin giả tiếng Việt trên mạng xã hội sử dụng mô hình Transformer (PhoBERT-base-v2) thuần văn bản và giải thích AI (XAI)."*

---

## 📌 Bảng Tổng Hợp Kết Quả Thực Nghiệm Thực Tế (Test Set N=390)

Toàn bộ các số liệu dưới đây được sinh tự động từ thực nghiệm thật và lưu tại `outputs/evaluation/test_metrics.json` và `outputs/evaluation/baseline_metrics.json`:

| Nhóm Mô Hình | Tên Mô Hình Thực Nghiệm | Accuracy | Macro F1 | Precision | Recall | Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Đa số** | Predict All Real (Đoán toàn bộ thật) | 68.97% | 0.4082 | 0.3449 | 0.5000 | Phân bố Test (269/121) |
| **Truyền thống** | Multinomial Naive Bayes (TF-IDF) | 86.92% | 0.8291 | 0.8946 | 0.8006 | Thực nghiệm Test Set |
| | Logistic Regression (TF-IDF) | 87.69% | 0.8593 | 0.8526 | 0.8676 | Thực nghiệm Test Set |
| | Random Forest (TF-IDF) | 88.97% | 0.8657 | 0.8851 | 0.8519 | Thực nghiệm Test Set |
| | Linear SVM (TF-IDF) | 90.00% | 0.8801 | 0.8917 | 0.8707 | Thực nghiệm Test Set |
| **Deep Learning** | **PhoBERT-base-v2 (Thuần Văn Bản)** | **94.62%** | **0.9354** | **0.9487** | **0.9246** | **Checkpoint RTX 3050** |

- **Ma trận Nhầm lẫn (Confusion Matrix):**
  $$\begin{bmatrix} 264 & 5 \\ 16 & 105 \end{bmatrix}$$
  - Hàng 0 (Tin Thật): $264 + 5 = 269$ mẫu (Recall 98.14%)
  - Hàng 1 (Tin Giả ): $16 + 105 = 121$ mẫu (Recall 86.78%, Precision 95.45%)
  - Tổng hàng khớp $100\%$ phân bố ground truth của tập kiểm thử.
- **ROC-AUC Score:** **0.9843**.
- **Độ vững chắc khi che tên báo & URL:** F1 chỉ biến thiên **-0.26%**, chứng minh mô hình hiểu ngữ nghĩa văn bản.

---

## 📁 Cấu Trúc Toàn Bộ Dự Án

```text
├── config.py                   # Cấu hình tập trung (Đường dẫn, Max length, Class weights, Bảo mật)
├── requirements.txt            # Danh mục thư viện Python đã ghim phiên bản chuẩn
├── README.md                   # Tài liệu hướng dẫn đồ án
├── data/
│   ├── fake_news_system.db     # SQLite Database lưu Lịch sử, Bằng chứng, Phản hồi
│   ├── raw/data.csv            # Dữ liệu tổng hợp thô
│   └── processed/              # train.csv, val.csv, test.csv đã tiền xử lý sạch
├── notebooks/
│   └── Huan_Luyen_PhoBERT_Colab.ipynb # Notebook huấn luyện trên Google Colab T4 miễn phí
├── outputs/                    # Checkpoint PhoBERT, ảnh biểu đồ, file export CSV
│   ├── phobert/best/           # Checkpoint mô hình Transformer tốt nhất
│   ├── phobert/training_history.json # Lịch sử hàm mất mát qua từng epoch
│   └── evaluation/             # test_metrics.json, confusion_matrix.png, error_analysis.csv
└── src/
    ├── __init__.py
    ├── dataset/
    │   └── build_complete_dataset.py # Xây dựng dữ liệu sạch thuần văn bản (không rò rỉ metadata)
    ├── baseline.py             # TF-IDF + Logistic Regression, SVM, Naive Bayes, Random Forest
    ├── train.py                # Pipeline huấn luyện PhoBERT (Weighted Cross-Entropy Loss)
    ├── evaluate.py             # Đánh giá tập test 1 lần duy nhất & xuất file JSON/CSV/PNG
    ├── cross_domain_eval.py    # Đánh giá chéo nguồn (Domain Shift) & Che tên báo/URL
    ├── ablation_and_metrics.py # Sinh biểu đồ Loss/F1, Confusion Matrix, Bảng Ablation từ JSON
    ├── model_comparison.py     # Bảng tổng hợp so sánh thực nghiệm động từ JSON
    ├── url_extractor.py        # Cào & trích xuất nội dung từ đường dẫn URL bài báo
    ├── fact_checker.py         # Engine đối chiếu bằng chứng
    ├── explainer.py            # Giải thích mô hình AI (XAI) & tô màu từ khóa
    ├── chatbot.py              # Trợ lý ảo Chatbot AI tư vấn và hướng dẫn kiểm chứng
    └── app.py                  # Giao diện Web Gradio 6 Phân hệ Toàn diện
```

---

## 🚀 Hướng Dẫn Sử Dụng & Toàn Bộ Câu Lệnh Khởi Chạy

### 1. Khởi chạy Giao diện Web (Gradio)
- **Chạy Web nội bộ:**
```bash
python src/app.py
```
- **Tạo link chia sẻ công khai (Cloudflare Tunnel an toàn):**
```bash
python src/app.py --share
```
> Mở trình duyệt tại: **`http://127.0.0.1:7860`** để trải nghiệm đầy đủ 6 phân hệ:
> - **Tab 1: 🔍 Kiểm Tra Tin Tức AI**: Dán URL Báo chí hoặc dán văn bản, phân loại PhoBERT, tính chỉ số bất định MC Dropout & Entropy OOD, giải thích từ khóa (XAI).
> - **Tab 2: 📁 Phân Tích Hàng Loạt (Batch CSV/Excel)**: Gắn nhãn hàng loạt hàng trăm bài viết, xuất file `.csv`.
> - **Tab 3: ⚖️ So Sánh Hai Bài Viết**: Đối chiếu 2 bài viết đo độ tương đồng ngữ nghĩa và phát hiện mâu thuẫn.
> - **Tab 4: 🕒 Lịch Sử Kiểm Tra**: Tìm kiếm, lọc và xuất dữ liệu lịch sử vào SQLite.
> - **Tab 5: 💬 Chatbot AI**: Trợ lý ảo tư vấn phòng chống tin giả.
> - **Tab 6: 📊 Admin & Nghiên Cứu Học Sâu**: Xem thống kê thời gian thực, biểu đồ Loss/F1 thật và ma trận nhầm lẫn thật.

---

### 2. Huấn luyện Mô hình AI (Fine-tuning PhoBERT trên GPU)
Tận dụng GPU máy tính hoặc Google Colab với Mixed Precision FP16:
```bash
python src/train.py --epochs 3 --batch_size 8 --learning_rate 2e-5
```

---

### 3. Đánh giá Mô hình trên Tập Test (Xuất file JSON/PNG thật)
```bash
python src/evaluate.py
```

---

### 4. Đánh giá Chéo Nguồn & Che Tên Báo / URL
```bash
python src/cross_domain_eval.py
```

---

### 5. Huấn luyện & Đánh giá các mô hình Baseline
```bash
python src/baseline.py
```

---

### 6. Đóng tiến trình an toàn khi bị kẹt cổng (Windows)
Sử dụng script đóng theo PID an toàn (không gây ảnh hưởng các tiến trình Python khác):
- Chạy file: `6_TAT_TIEN_TRINH_BI_KET.bat`
- Hoặc đóng thủ công theo cổng:
```cmd
for /f "tokens=5" %a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do taskkill /F /PID %a
```

---

### 📁 Bộ công cụ 1-Click (File `.bat`)
- `1_UP_CODE_LEN_GITHUB.bat`: Đẩy code lên GitHub.
- `2_CHAY_WEB_CHIA_SE_CONG_KHAI.bat`: Bật web kèm link công khai.
- `3_CHAY_WEB_NOI_BO.bat`: Mở web chạy nội bộ.
- `4_HUAN_LUYEN_PHOBERT.bat`: Huấn luyện mô hình PhoBERT.
- `5_DANH_GIA_VA_XUAT_BIEU_DO.bat`: Xuất đồ thị và bảng so sánh báo cáo.
- `6_TAT_TIEN_TRINH_BI_KET.bat`: Đóng server an toàn theo PID và giải phóng cổng.
