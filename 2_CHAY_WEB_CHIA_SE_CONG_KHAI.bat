@echo off
chcp 65001 >nul
echo ========================================================================
echo  🌐 ĐANG KHỞI CHẠY HỆ THỐNG VÀ TẠO ĐƯỜNG LINK CHIA SẺ CÔNG KHAI...
echo  (Bất kỳ ai trên máy tính hoặc điện thoại khác đều truy cập được)
echo ========================================================================
python src/app.py --share
pause

