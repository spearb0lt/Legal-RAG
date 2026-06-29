@echo off
cd /d "%~dp0"

:: Kill any lingering process on port 8501
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    echo Killing existing process on port 8501 ^(PID %%a^)...
    taskkill /F /PID %%a >nul 2>&1
)

echo Starting Indian Legal RAG...
echo.
echo App will open at: http://localhost:8501
echo Keep this window open while using the app.
echo Press Ctrl+C to stop.
echo.

..\venv\Scripts\streamlit.exe run app.py
