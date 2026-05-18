@echo off
setlocal EnableDelayedExpansion
title Lab Rat AI - Power BI Helper

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

cd /d "%ROOT%"
python start_powerbi.py
pause
