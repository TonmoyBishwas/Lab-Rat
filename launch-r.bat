@echo off
setlocal EnableDelayedExpansion
title Lab Rat AI - R programming

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

cd /d "%ROOT%"
python start_r.py
pause
