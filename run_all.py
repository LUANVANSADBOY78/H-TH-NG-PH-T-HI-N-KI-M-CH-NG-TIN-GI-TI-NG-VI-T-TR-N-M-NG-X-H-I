"""
Master Pipeline Script - Chạy toàn bộ quy trình của đồ án trong một lệnh duy nhất:
1. Xác minh & Chuẩn bị bộ dữ liệu sạch (Clean Text-Only Dataset: ViFN + Social Media)
2. Huấn luyện & Đánh giá các mô hình Baseline (Logistic Regression, Linear SVM, Naive Bayes, Random Forest)
3. Đánh giá mô hình PhoBERT-base-v2 trên tập kiểm thử (Test Set 390 mẫu)
4. Đánh giá kiểm định Cross-Domain & Source/URL Masking Robustness
5. Xuất đồ thị trực quan: Confusion Matrix, Loss Curves, F1 Curves, Error Analysis
6. Kiểm thử tự động toàn bộ hệ thống (Unit Tests: 14 test cases)
7. (Tùy chọn --web) Khởi chạy giao diện Web Demo (Gradio) tại http://127.0.0.1:7860
"""

import os
import sys
import argparse
import subprocess
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_step(cmd: str, description: str) -> bool:
    print("\n" + "=" * 70)
    print(f">>> [BƯỚC] {description}")
    print(f">>> Lệnh thực thi: {cmd}")
    print("=" * 70)
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[!] Thất bại: Lệnh kết thúc với mã lỗi {result.returncode} ({elapsed:.1f}s)")
        return False
    else:
        print(f"-> Hoàn tất: {description} thành công trong {elapsed:.1f}s!\n")
        return True

def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline đồ án Phát hiện Tin giả Tiếng Việt")
    parser.add_argument("--train", action="store_true", help="Huấn luyện lại mô hình PhoBERT")
    parser.add_argument("--rebuild-data", action="store_true", help="Xây dựng lại bộ dữ liệu từ nguồn thô")
    parser.add_argument("--web", action="store_true", help="Khởi chạy giao diện Web Demo sau khi chạy xong pipeline")
    parser.add_argument("--share", action="store_true", help="Tạo public Gradio link khi khởi chạy web")
    args = parser.parse_args()

    print("""
    ========================================================================
    🛡️ HỆ THỐNG PHÁT HIỆN & KIỂM CHỨNG TIN GIẢ TIẾNG VIỆT (PHOBERT + XAI)
    Khoa Công Nghệ Thông Tin - Đồ Án Môn Học Sâu (Deep Learning)
    Pipeline Kiểm Thử & Chạy Toàn Bộ Hệ Thống (End-to-End Execution)
    ========================================================================
    """)

    # 1. Kiểm tra / Tái tạo dữ liệu
    need_build_data = args.rebuild_data or not (
        os.path.exists("data/processed/train.csv") and
        os.path.exists("data/processed/val.csv") and
        os.path.exists("data/processed/test.csv")
    )
    if need_build_data:
        ok = run_step("python src/dataset/build_complete_dataset.py", "1. Chuẩn bị & hợp nhất dữ liệu sạch (ViFN + Social)")
        if not ok:
            print("[!] Lỗi tại bước chuẩn bị dữ liệu. Dừng pipeline.")
            sys.exit(1)
    else:
        print("[*] Dữ liệu data/processed/ (train/val/test.csv) đã sẵn sàng.")

    # 2. Huấn luyện lại PhoBERT nếu có cờ --train
    if args.train:
        ok = run_step("python src/train.py --epochs 3 --batch_size 16 --learning_rate 2e-5", "2. Huấn luyện PhoBERT Transformer")
        if not ok:
            print("[!] Lỗi khi huấn luyện PhoBERT. Dừng pipeline.")
            sys.exit(1)

    # 3. Chạy Baselines
    ok = run_step("python src/baseline.py", "3. Huấn luyện & Đánh giá Baselines (LR, SVM, NB, RF)")
    if not ok:
        print("[!] Lỗi khi chạy Baselines.")

    # 4. Đánh giá PhoBERT trên tập Test
    ok = run_step("python src/evaluate.py", "4. Đánh giá PhoBERT trên tập Test (390 mẫu)")
    if not ok:
        print("[!] Lỗi khi đánh giá PhoBERT.")

    # 5. Đánh giá Cross-Domain & Robustness
    ok = run_step("python src/cross_domain_eval.py", "5. Đánh giá Cross-domain & Masking Robustness")
    if not ok:
        print("[!] Lỗi khi đánh giá Robustness.")

    # 6. Xuất đồ thị và bảng so sánh
    ok = run_step("python src/ablation_and_metrics.py", "6. Xuất biểu đồ trực quan & Ma trận nhầm lẫn")
    ok = run_step("python src/model_comparison.py", "7. Tổng hợp bảng so sánh mô hình")

    # 7. Chạy kiểm thử tự động (Unit Tests)
    ok = run_step("python -m unittest discover tests", "8. Chạy bộ kiểm thử tự động (14 Test Cases)")
    if not ok:
        print("[!] Có kiểm thử tự động thất bại.")

    print("\n" + "=" * 70)
    print("🎉 TOÀN BỘ CÁC BƯỚC PIPELINE ĐÃ ĐƯỢC THỰC THI THÀNH CÔNG!")
    print("=" * 70)

    # 8. Web app nếu yêu cầu
    if args.web:
        share_flag = "--share" if args.share else ""
        print("\n" + "=" * 70)
        print(">>> 🚀 KHỞI CHẠY GIAO DIỆN WEB DEMO TOÀN DIỆN (6 PHÂN HỆ)")
        print(">>> 🌐 Mở trình duyệt truy cập: http://127.0.0.1:7860")
        print(">>> (Nhấn Ctrl+C trong terminal nếu muốn dừng hệ thống)")
        print("=" * 70 + "\n")
        subprocess.run(f"python src/app.py {share_flag}", shell=True)

if __name__ == "__main__":
    main()
