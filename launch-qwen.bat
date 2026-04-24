@echo off
setlocal EnableDelayedExpansion
title Lab Rat AI - Qwen2.5

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PORT=11434"
set "THREADS=8"
set "CTX=8192"

echo.
echo  =====================================================
echo   Lab Rat AI  ^|  Qwen2.5  ^|  CPU mode
echo  =====================================================
echo.

if not exist "%ROOT%\llama-cpp\llama-server.exe" (
    echo  ERROR: llama-server.exe not found in llama-cpp\
    pause & exit /b 1
)

:: Auto-detect which model is present (prefer larger)
set "MODEL="
if exist "%ROOT%\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf" set "MODEL=Qwen2.5-14B-Instruct-Q4_K_M.gguf" & set "CTX=4096"
if exist "%ROOT%\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"  set "MODEL=Qwen2.5-7B-Instruct-Q4_K_M.gguf"
if exist "%ROOT%\models\Qwen2.5-7B-Instruct-Q2_K.gguf"    set "MODEL=Qwen2.5-7B-Instruct-Q2_K.gguf"

if not defined MODEL (
    echo  ERROR: No model found in models\
    echo  Run: python download-model.py
    pause & exit /b 1
)

:: Kill any previous instances on both ports
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1

echo  Model:   %MODEL%
echo  Threads: %THREADS%  ^|  Context: %CTX% tokens
echo  Starting server... (first load takes 30-90 seconds)
echo.

:: pushd changes working directory so relative paths work (avoids spaces-in-path issues with start)
pushd "%ROOT%"
start "" /B llama-cpp\llama-server.exe --model "models\%MODEL%" --threads %THREADS% --ctx-size %CTX% --batch-size 512 --ubatch-size 128 --port %PORT% --host 127.0.0.1 --flash-attn --no-mmap
popd

set /a TRIES=0
:WAITLOOP
    timeout /t 3 /nobreak >nul
    curl -s -f "http://127.0.0.1:%PORT%/health" >nul 2>&1
    if %errorlevel% == 0 goto READY
    set /a TRIES+=1
    if %TRIES% GEQ 60 (
        echo.
        echo  Timeout: server did not start after 3 minutes.
        pause & exit /b 1
    )
    set /a ELAPSED=TRIES*3
    echo  Loading... !ELAPSED!s
    goto WAITLOOP

:READY
echo.
echo  Server ready! Starting UI server on http://localhost:8080...

:: Serve ui\ via Python HTTP server so fetch() calls work (file:// blocks them)
pushd "%ROOT%"
start "" /B python -m http.server 8080 --directory ui
popd

:: Small wait for Python server to bind
timeout /t 1 /nobreak >nul

:: Open in browser — VS Code intercepts localhost links into Simple Browser panel
:: if workbench.browser.openLocalhostLinks is enabled in .vscode/settings.json
start "" "http://localhost:8080"
echo.
echo  =====================================================
echo   AI is running. Keep this window open.
echo   Close it or press Ctrl+C to stop.
echo  =====================================================
echo.

:LOOP
timeout /t 60 /nobreak >nul
goto LOOP
