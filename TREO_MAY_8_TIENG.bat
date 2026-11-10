@echo off
chcp 65001 >nul
title HỆ THỐNG KIỂM CHỨNG TIN GIẢ - CHẾ ĐỘ TREO MÁY 8 TIẾNG
color 0A

echo ==============================================================================
echo       🛡️ HỆ THỐNG PHÁT HIỆN TIN GIẢ - CHẾ ĐỘ TREO MÁY 8 TIẾNG (ONLINE 24/7) 🌐
echo ==============================================================================
echo  [*] 1. Đang kích hoạt chế độ chống ngủ cho máy tính (Sleep Disabled)...
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
echo  [*] 2. Màn hình sẽ tự tắt sau 10 phút để tiết kiệm điện và làm mát máy...
powercfg /change monitor-timeout-ac 10 >nul 2>&1

echo  [*] 3. Dọn dẹp các cổng mạng và tiến trình cũ...
taskkill /F /IM python.exe >nul 2>&1

echo.
echo ==============================================================================
echo  🚀 BẮT ĐẦU CHẠY HỆ THỐNG VÀ TẠO ĐƯỜNG LINK CÔNG KHAI (GRADIO.LIVE)
echo  Thời gian bắt đầu treo máy: %date% %time%
echo  (Sau khi hiện link https://xxxx.gradio.live, bạn copy gửi cho người khác)
echo  (Muốn dừng bất cứ lúc nào, chỉ cần nhấn tổ hợp phím Ctrl + C hoặc tắt cửa sổ)
echo ==============================================================================
echo.

python src/app.py --share

pause

