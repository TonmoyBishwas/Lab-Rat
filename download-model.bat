@echo off
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "MODELS=%ROOT%\models"

echo.
echo  =====================================================
echo   Lab Rat AI  ^|  Model Downloader
echo  =====================================================
echo.
echo  Select a model to download:
echo.
echo  [1] Qwen2.5-7B Instruct Q4_K_M  (~4.7 GB)  RECOMMENDED
echo      Best for Python + data viz. 8-15 tok/s on i7/i5 12th gen.
echo.
echo  [2] Qwen2.5-14B Instruct Q4_K_M  (~8.9 GB)  BETTER QUALITY
echo      Smarter answers. Only use if you have 16GB free RAM.
echo.
echo  [3] Qwen2.5-7B Instruct Q2_K  (~2.9 GB)  STORAGE-SAVER
echo      Lower quality but smallest download.
echo.
set /p CHOICE=Enter choice (1, 2, or 3) then press Enter:

if "%CHOICE%"=="1" goto OPT1
if "%CHOICE%"=="2" goto OPT2
if "%CHOICE%"=="3" goto OPT3
echo  Invalid choice: "%CHOICE%"
pause
exit /b 1

:OPT1
set "FILENAME=Qwen2.5-7B-Instruct-Q4_K_M.gguf"
set "URL=https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
set "LAUNCHER=launch-qwen.bat"
goto DOWNLOAD

:OPT2
set "FILENAME=Qwen2.5-14B-Instruct-Q4_K_M.gguf"
set "URL=https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
set "LAUNCHER=launch-qwen.bat"
goto DOWNLOAD

:OPT3
set "FILENAME=Qwen2.5-7B-Instruct-Q2_K.gguf"
set "URL=https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q2_K.gguf"
set "LAUNCHER=launch-qwen.bat"
goto DOWNLOAD

:DOWNLOAD
if not exist "%MODELS%" mkdir "%MODELS%"

if exist "%MODELS%\%FILENAME%" (
    echo.
    echo  File already exists: %FILENAME%
    set /p OVERWRITE=Delete and re-download? (y/n):
    if /i "%OVERWRITE%"=="y" (
        del "%MODELS%\%FILENAME%"
        echo  Deleted.
    ) else (
        echo  Keeping existing file.
        pause
        exit /b 0
    )
)

echo.
echo  Downloading: %FILENAME%
echo  Saving to:   %MODELS%\%FILENAME%
echo.
echo  TIP: If it stops, run this script again to resume.
echo.

curl --progress-bar -f -L -C - -o "%MODELS%\%FILENAME%" "%URL%"

if %errorlevel% NEQ 0 (
    echo.
    echo  Download failed. Check your internet connection and try again.
    if exist "%MODELS%\%FILENAME%" del "%MODELS%\%FILENAME%"
    pause
    exit /b 1
)

echo.
echo  =====================================================
echo   Download complete!
echo   File: %MODELS%\%FILENAME%
echo.
echo   Run %LAUNCHER% to start the AI.
echo  =====================================================
echo.
pause
