@echo off
setlocal EnableDelayedExpansion
title Lab Rat AI - Gemma 4 E4B

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "SERVER=%ROOT%\llama-cpp\llama-server.exe"
set "MODEL=%ROOT%\models\gemma-4-e4b-it-Q4_K_M.gguf"
set "UI=%ROOT%\ui\index.html"
set "PORT=11434"

:: Gemma 4 E4B: ~3.5GB model — more RAM available for context
set "THREADS=8"
set "CTX=8192"

echo.
echo  =====================================================
echo   Lab Rat AI  ^|  Gemma 4 E4B  ^|  CPU mode
echo  =====================================================
echo.

if not exist "%SERVER%" (
    echo  ERROR: llama-server.exe not found in llama-cpp\
    echo  Run update-llama-cpp.bat to get the binaries.
    pause & exit /b 1
)

if not exist "%MODEL%" (
    echo  ERROR: Model not found at:
    echo    %MODEL%
    echo.
    echo  Run download-model.bat and choose option 2.
    pause & exit /b 1
)

:: Kill any previous instance on this port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo  Starting server... (first load takes 20-60 seconds)
echo  Model: Gemma 4 E4B Q4_K_M
echo  Threads: %THREADS%  ^|  Context: %CTX% tokens
echo.

start "llama-server" /B "%SERVER%" ^
  --model "%MODEL%" ^
  --threads %THREADS% ^
  --ctx-size %CTX% ^
  --batch-size 512 ^
  --ubatch-size 128 ^
  --port %PORT% ^
  --host 127.0.0.1 ^
  --flash-attn ^
  --no-mmap

set /a TRIES=0
:WAITLOOP
    timeout /t 3 /nobreak >nul
    curl -s -f "http://127.0.0.1:%PORT%/health" >nul 2>&1
    if %errorlevel% == 0 goto READY
    set /a TRIES+=1
    if %TRIES% GEQ 60 (
        echo.
        echo  Timeout after 3 minutes.
        pause & exit /b 1
    )
    set /a ELAPSED=TRIES*3
    echo  Loading... !ELAPSED!s
    goto WAITLOOP

:READY
echo.
echo  Server ready! Opening browser...
echo.
start "" "%UI%"

echo  =====================================================
echo   AI is running. Keep this window open.
echo   Close it (or press Ctrl+C) to stop.
echo  =====================================================
echo.

:LOOP
timeout /t 60 /nobreak >nul
goto LOOP
