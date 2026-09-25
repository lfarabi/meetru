@echo off
cd /d "%~dp0"
.venv\Scripts\python main.py
if errorlevel 1 pause
