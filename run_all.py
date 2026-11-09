"""
Master Pipeline Script - Chạy toàn bộ quy trình của đồ án trong một lệnh duy nhất:
1. Chuẩn bị dữ liệu mẫu (Sample Dataset)
2. Tiền xử lý, chuẩn hóa teencode, tách từ tiếng Việt (Underthesea)
3. Huấn luyện & Đánh giá các mô hình Baseline (Logistic Regression, Linear SVM, Naive Bayes, Random Forest)
4. Sinh biểu đồ huấn luyện (Loss Curves), ma trận nhầm lẫn (Confusion Matrix) & Bảng Ablation Study
5. Khởi chạy toàn bộ hệ thống Web Demo đa phân hệ (Gradio) tại http://127.0.0.1:7860
"""

import os
import sys
import subprocess
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_command(cmd, description):
    print("\n" + "=" * 70)
    print(f">>> [BƯỚC] {description}")
    print(f">>> Lệnh thực thi: {cmd}")
    print("=" * 70)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[!] Cảnh báo: Lệnh '{cmd}' kết thúc với mã lỗi {result.returncode}")
    else:
        print(f"-> Hoàn tất bước: {description} thành công!\n")

def main():
    print("""
    ========================================================================
    🛡️ HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT (PHOBERT + XAI)
    Khoa Công Nghệ Thông Tin - Đồ Án Môn Học Sâu (Deep Learning)
    ========================================================================
    """)

    # 1. Tạo dữ liệu mẫu nếu chưa có data/raw/data.csv
    if not os.path.exists("data/raw/data.csv"):
        run_command("python data/create_sample_data.py", "1. Tạo bộ dữ liệu mẫu benchmark")
    else:
        print("[*] Dữ liệu data/raw/data.csv đã tồn tại, tiếp tục sang bước tiền xử lý.")

    # 2. Tiền xử lý dữ liệu
    run_command("python src/preprocess.py", "2. Tiền xử lý, chuẩn hóa teencode & tách từ tiếng Việt")

    # 3. Chạy các mô hình Baseline truyền thống
    run_command("python src/baseline.py", "3. Huấn luyện & Đánh giá mô hình Baseline (LR, SVM, NB, RF)")

    # 4. Sinh đồ thị Loss, Confusion Matrix và Bảng Ablation Study
    run_command("python src/ablation_and_metrics.py", "4. Sinh đồ thị Loss/F1, Confusion Matrix & Ablation Study")

    # 5. Nếu có cờ --train thì chạy thêm fine-tuning PhoBERT
    if "--train" in sys.argv:
        run_command("python src/train.py --epochs 3 --batch_size 16 --learning_rate 2e-5", "5. Fine-tuning PhoBERT Transformer")

    # 6. Khởi chạy Giao diện Web Toàn diện
    print("\n" + "=" * 70)
    print(">>> 🚀 [BƯỚC CUỐI] KHỞI CHẠY GIAO DIỆN WEB DEMO TOÀN DIỆN (6 PHÂN HỆ)")
    print(">>> 🌐 Mở trình duyệt truy cập: http://127.0.0.1:7860")
    print(">>> (Nhấn Ctrl+C trong terminal nếu muốn dừng hệ thống)")
    print("=" * 70 + "\n")
    
    subprocess.run("python src/app.py", shell=True)

if __name__ == "__main__":
    main()

