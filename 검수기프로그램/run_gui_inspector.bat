@echo off
title SNU AI Challenge - GUI Syntax Inspector
cd /d "%~dp0"
if exist "C:\Users\user\anaconda3\python.exe" (
    "C:\Users\user\anaconda3\python.exe" gui_syntax_inspector.py
) else (
    python gui_syntax_inspector.py
)
pause
