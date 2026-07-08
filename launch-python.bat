@echo off
setlocal EnableDelayedExpansion
title Data Visualization Lab - Python

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

cd /d "%ROOT%"
python start.py dataviz-python
pause
