@echo off
chcp 65001 >nul
title HỆ THỐNG KIỂM CHỨNG TIN GIẢ - CHẾ ĐỘ TREO MÁY 20+ TIẾNG
color 0A

echo ==============================================================================
echo       🛡️ HỆ THỐNG PHÁT HIỆN TIN GIẢ - CHẾ ĐỘ TREO MÁY LIÊN TỤC 20 TIẾNG 🌐
echo ==============================================================================
echo  [*] 1. Đang cấu hình Windows chống ngủ (Sleep Disabled khi cắm sạc)...
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
echo  [*] 2. Đã bật chế độ tiết kiệm màn hình (tắt sau 15p nhưng máy vẫn chạy).
powercfg /change monitor-timeout-ac 15 >nul 2>&1

echo  [*] 3. Kiểm tra và dọn dẹp các tiến trình bị kẹt trước đó...
taskkill /F /IM python.exe >nul 2>&1

echo ==============================================================================
echo  🚀 BẮT ĐẦU KHỞI CHẠY HỆ THỐNG VÀ TẠO LINK CHIA SẺ CÔNG KHAI (GRADIO.LIVE)
echo  Link công khai có hiệu lực liên tục 72 giờ (vượt xa 20 tiếng).
echo  Nếu gặp sự cố mạng, kịch bản sẽ TỰ ĐỘNG KHỞI ĐỘNG LẠI sau 5 giây!
echo ==============================================================================
echo.

:LOOP
echo [%date% %time%] >>> Khởi động máy chủ Web AI Fact-checking...
python src/app.py --share

echo.
echo [!] Cảnh báo: Máy chủ vừa tạm dừng hoặc ngắt kết nối vào lúc %time%.
echo [!] Đang tự động kết nối và bật lại sau 5 giây để duy trì treo máy...
timeout /t 5 /nobreak >nul
goto LOOP
