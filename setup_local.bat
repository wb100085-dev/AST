@echo off
chcp 65001 >nul
echo [1/3] 가상환경 확인...
if not exist "venv\Scripts\python.exe" (
    echo venv 없음. 생성 중...
    python -m venv venv
    if errorlevel 1 (
        echo 오류: python -m venv 실패. Python이 설치되어 있는지 확인하세요.
        pause
        exit /b 1
    )
)

echo [2/3] 패키지 설치 (requirements.txt)...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo 오류: pip install 실패. 인터넷 연결을 확인하세요.
    pause
    exit /b 1
)

echo [3/3] 앱 실행...
echo 브라우저가 자동으로 열립니다. 종료하려면 이 창에서 Ctrl+C 를 누르세요.
streamlit run run.py

pause
