@echo off
chcp 65001 >nul
echo ========================================================================
echo  📊 ĐANG ĐÁNH GIÁ TOÀN DIỆN MÔ HÌNH VÀ XUẤT ĐỒ THỊ BÁO CÁO...
echo ========================================================================
echo [1/5] Chạy đánh giá Baseline (LR, SVM, NB, RF)...
python src/baseline.py
echo.
echo [2/5] Chạy đánh giá PhoBERT trên tập Test...
python src/evaluate.py
echo.
echo [3/5] Chạy đánh giá Cross-domain và Masking Robustness...
python src/cross_domain_eval.py
echo.
echo [4/5] Sinh biểu đồ huấn luyện và Ma trận nhầm lẫn...
python src/ablation_and_metrics.py
echo.
echo [5/5] Sinh bảng so sánh tổng hợp...
python src/model_comparison.py
echo.
echo ========================================================================
echo  ✅ ĐÃ XUẤT TOÀN BỘ KẾT QUẢ ĐÁNH GIÁ & ĐỒ THỊ VÀO THƯ MỤC outputs/
echo ========================================================================
pause
