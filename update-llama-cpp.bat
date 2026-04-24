@echo off
setlocal EnableDelayedExpansion
title Lab Rat AI - Update llama.cpp

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "DEST=%ROOT%\llama-cpp"

echo.
echo  =====================================================
echo   Lab Rat AI  ^|  Update llama.cpp Binaries
echo  =====================================================
echo.
echo  Current binaries were copied from Aug 2025 build.
echo  Updating gives better CPU performance + new features.
echo.
echo  MANUAL METHOD (always works):
echo  1. Go to: https://github.com/ggerganov/llama.cpp/releases/latest
echo  2. Download: llama-bXXXX-bin-win-avx2-x64.zip
echo     (avx2 = correct for Intel 12th gen)
echo  3. Open the zip, copy all .exe and .dll files into:
echo     %DEST%\
echo  4. Overwrite existing files.
echo.
echo  AUTO METHOD (needs internet, uses PowerShell + curl):
echo.
set /p AUTO=Auto-download now? (y/n):
if /i not "%AUTO%"=="y" goto DONE

echo.
echo  Fetching latest release info from GitHub...
curl -s "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" > "%TEMP%\llama_release.json"

if %errorlevel% NEQ 0 (
    echo  Could not reach GitHub. Check internet connection.
    goto DONE
)

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Content \"%TEMP%\\llama_release.json\" | ConvertFrom-Json).tag_name"') do set TAG=%%A

if not defined TAG (
    echo  Failed to parse release info.
    goto DONE
)

echo  Latest release: %TAG%
set "ZIPNAME=llama-%TAG%-bin-win-avx2-x64.zip"
set "ZIPURL=https://github.com/ggerganov/llama.cpp/releases/download/%TAG%/%ZIPNAME%"
set "TMPZIP=%TEMP%\llama_latest.zip"
set "TMPDIR=%TEMP%\llama_extracted"

echo  Downloading %ZIPNAME%...
curl -L --progress-bar -o "%TMPZIP%" "%ZIPURL%"

if %errorlevel% NEQ 0 (
    echo  Download failed.
    goto DONE
)

echo  Extracting...
if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
powershell -NoProfile -Command "Expand-Archive -Path '%TMPZIP%' -DestinationPath '%TMPDIR%' -Force"

echo  Copying binaries to llama-cpp\...
powershell -NoProfile -Command "Get-ChildItem '%TMPDIR%' -Recurse -Include '*.exe','*.dll' | Copy-Item -Destination '%DEST%\' -Force"

del "%TMPZIP%" >nul 2>&1
rmdir /s /q "%TMPDIR%" >nul 2>&1

echo.
echo  =====================================================
echo   Update complete! New binaries are in llama-cpp\
echo  =====================================================
echo.

:DONE
pause
