@echo off
cd /d "%~dp0"
echo ========================================================
echo   Launching Hinglish Sentiment Research Command Center
echo ========================================================
"C:\Users\madha\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
pause
