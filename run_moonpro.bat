@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd %~dp0
IF NOT EXIST .venv (
  python -m venv .venv
  CALL .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) ELSE (
  CALL .venv\Scripts\activate
)
start "" /b cmd /c "streamlit run moonpro_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false"
timeout /t 3 >nul
start "" http://localhost:8501
popd
