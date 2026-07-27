@echo off
setlocal EnableDelayedExpansion
title Data Analytics Lab - Python

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

REM ---------------------------------------------------------------
REM  Find a Python. The lab PC may have Python installed but NOT on
REM  PATH, which makes a bare "python" fail even though it is there.
REM  None of this needs admin rights - we only ever RUN things.
REM ---------------------------------------------------------------
set "PY="

where python >nul 2>&1 && set "PY=python"

if not defined PY (
    py -3 --version >nul 2>&1 && set "PY=py -3"
)

if not defined PY (
    for %%D in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%ProgramFiles%\Python313\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "%ProgramFiles%\Python310\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
        "%USERPROFILE%\anaconda3\python.exe"
        "%USERPROFILE%\miniconda3\python.exe"
        "C:\ProgramData\Anaconda3\python.exe"
        "%ROOT%\python\python.exe"
    ) do (
        if not defined PY if exist %%D set "PY=%%~D"
    )
)

if not defined PY (
    echo.
    echo  ============================================================
    echo   No Python found on this PC.
    echo  ============================================================
    echo.
    echo   Lab Rat needs Python to run. Try, in order:
    echo.
    echo    1. Open a terminal and type:  py -3 --version
    echo       If that works, this launcher has a bug - tell Claude.
    echo.
    echo    2. Look for Anaconda / Jupyter in the Start menu. If the
    echo       lab has Jupyter for the Data Analytics course, Python
    echo       IS installed - find its folder and copy python.exe's
    echo       folder path, then drop a portable copy into:
    echo          %ROOT%\python\
    echo.
    echo    3. Worst case, unzip a portable Python into %ROOT%\python\
    echo       (python-3.12-embed-amd64.zip from python.org needs no
    echo       installer and no admin rights).
    echo.
    pause
    exit /b 1
)

echo  Using Python: %PY%
%PY% start.py da-python
pause
