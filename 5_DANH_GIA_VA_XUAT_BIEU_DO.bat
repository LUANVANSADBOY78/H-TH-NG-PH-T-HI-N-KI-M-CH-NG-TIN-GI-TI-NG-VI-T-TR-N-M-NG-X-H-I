@echo off
chcp 65001 >nul
echo ========================================================================
echo  📊 ĐANG ĐÁNH GIÁ MÔ HÌNH VÀ XUẤT ĐỒ THỊ CHẤM ĐIỂM BÁO CÁO...
echo ========================================================================
python src/ablation_and_metrics.py
python src/model_comparison.py
echo.
echo ========================================================================
echo  ✅ ĐÃ XUẤT CÁC ĐỒ THỊ & BẢNG SO SÁNH VÀO THƯ MỤC outputs/
echo ========================================================================
pause
