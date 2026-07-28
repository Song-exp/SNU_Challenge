@echo off
title SNU AI Challenge - GUI Syntax Inspector (100 Samples Validation)
cd /d "%~dp0"
if exist "C:\Users\user\anaconda3\python.exe" (
    "C:\Users\user\anaconda3\python.exe" gui_syntax_inspector.py --start_idx 0 --end_idx 99 --name Unsupervised_Test
) else (
    python gui_syntax_inspector.py --start_idx 0 --end_idx 99 --name Unsupervised_Test
)
pause
