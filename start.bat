@echo off
chcp 65001 >nul
setlocal

rem 优先使用已装好 OpenCV 的专用环境，其次回退到 default 环境
set PY=C:/Users/86156/.workbuddy/binaries/python/envs/fs39/Scripts/python.exe
if not exist "%PY%" set PY=C:/Users/86156/.workbuddy/binaries/python/envs/default/Scripts/python.exe
if not exist "%PY%" set PY=python

cd /d "%~dp0"

echo.
echo   FaceSwap Studio
echo   ----------------------------------------
echo   后端地址: http://127.0.0.1:8000
echo   浏览器没有自动打开的话，手动访问上面的地址。
echo   关闭这个窗口即可停止服务。
echo.

start "" http://127.0.0.1:8000
"%PY%" backend/app.py

pause
