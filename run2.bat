@echo off
REM Kill any process on port 8502 (separate port from app.py's 8501)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8502 ^| findstr LISTENING') do (
    echo Killing PID %%a on port 8502 ...
    taskkill /F /PID %%a >nul 2>&1
)

echo Starting app2.py (v2 corpus: seed + PCR + LSI only) on port 8502 ...
..\venv\Scripts\streamlit.exe run app2.py --server.port 8502
