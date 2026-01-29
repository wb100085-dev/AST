@echo off
chcp 65001 >nul
if not exist "venv\Scripts\streamlit.exe" (
    echo venv에 패키지가 설치되지 않았습니다. 먼저 setup_local.bat 을 실행하세요.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
streamlit run run.py
pause
