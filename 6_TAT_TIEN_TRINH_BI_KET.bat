@echo off
chcp 65001 >nul
echo ========================================================================
echo  🛑 ĐANG ĐÓNG TIẾN TRÌNH WEB SERVER VÀ GIẢI PHÓNG CỔNG 7860/7861...
echo ========================================================================

REM 1. Đóng theo PID file nếu tồn tại
if exist ".web_server.pid" (
    set /p SERVER_PID=<.web_server.pid
    if defined SERVER_PID (
        echo [*] Đang đóng tiến trình Web Server theo PID: %SERVER_PID%
        taskkill /F /PID %SERVER_PID% 2>nul
    )
    del /f /q .web_server.pid 2>nul
)

REM 2. Đóng tiến trình chiếm cổng 7860
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do (
    echo [*] Phát hiện cổng 7860 đang bị chiếm bởi PID: %%a, tiến hành giải phóng...
    taskkill /F /PID %%a 2>nul
)

REM 3. Đóng tiến trình chiếm cổng 7861
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7861 ^| findstr LISTENING') do (
    echo [*] Phát hiện cổng 7861 đang bị chiếm bởi PID: %%a, tiến hành giải phóng...
    taskkill /F /PID %%a 2>nul
)

echo.
echo ========================================================================
echo  ✅ ĐÃ GIẢI PHÓNG AN TOÀN CỔNG MẠNG (KHÔNG ẢNH HƯỞNG TIẾN TRÌNH PYTHON KHÁC)!
echo ========================================================================
pause
