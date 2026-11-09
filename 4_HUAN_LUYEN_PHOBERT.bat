@echo off
chcp 65001 >nul
echo ========================================================================
echo  🧠 ĐANG FINE-TUNING PHOBERT TRÊN GPU NVIDIA GEFORCE RTX 3050...
echo  (Đã tối ưu bộ nhớ 4GB VRAM - Hoàn thành sau khoảng 2 đến 4 phút)
echo ========================================================================
python src/train.py
pause

