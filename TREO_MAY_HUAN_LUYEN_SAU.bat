@echo off
chcp 65001 >nul
title HUẤN LUYỆN MÔ HÌNH CHUYÊN SÂU - TREO MÁY QUA ĐÊM
color 0B

echo ==============================================================================
echo       🧠 QUY TRÌNH HUẤN LUYỆN CHUYÊN SÂU & TỰ ĐỘNG BẬT WEB (TREO MÁY)
echo ==============================================================================
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-ac 15 >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

echo.
echo [%date% %time%] [*] BƯỚC 1: Chuẩn hóa, dịch teencode và xử lý toàn bộ dữ liệu...
python src/preprocess.py

echo.
echo [%date% %time%] [*] BƯỚC 2: Huấn luyện các mô hình Baseline (SVM, LR, Naive Bayes)...
python src/baseline.py

echo.
echo [%date% %time%] [*] BƯỚC 3: Fine-tuning PhoBERT trên GPU RTX 3050...
python src/train.py --epochs 5 --batch_size 8

echo.
echo [%date% %time%] [*] BƯỚC 4: Xuất biểu đồ Loss, Ma trận nhầm lẫn & Báo cáo Ablation...
python src/ablation_and_metrics.py
python src/model_comparison.py

echo.
echo ==============================================================================
echo  ✅ HUẤN LUYỆN HOÀN TẤT! BẮT ĐẦU TREO WEB CHIA SẺ CÔNG KHAI...
echo ==============================================================================
python src/app.py --share
pause
