Lab Rat AI - Portable Offline Local AI
========================================

QUICK START
-----------
1. Download a model:
     Run: download-model.bat
     Choose option 1 (Gemma 4 E4B-it Q4_K_M, ~5.0 GB) — recommended.
     Wait for download to finish.

2. Pick a mode and start the AI:
     launch-gemma4.bat   - Python / numpy / pandas / matplotlib / seaborn
                           (the default Data Visualization Lab mode)
     launch-powerbi.bat  - Power BI helper. Reads your open .pbix
                           model and grounds DAX/M answers in your
                           actual tables and relationships.
     launch-r.bat        - R / tidyverse / dplyr / ggplot2 helper
                           (for the R portion of the DV Lab exam)

     A black console window will appear — keep it open.
     Wait 30-90 seconds for the model to load.
     Your browser opens automatically.

3. Use it:
     Type your question and press Enter to send.
     Shift+Enter = new line inside your message.
     Drag & drop a .py, .r, .csv, .txt file onto the browser window.
     The AI will read the file and answer questions about it.

4. Stop:
     Close the black console window, or press Ctrl+C inside it.

Important: the three modes share the same ports (11434 for the AI,
8080 for the UI). Launching one mode automatically stops the other.
You cannot run two modes at the same time.


PORTABLE USE (USB / Lab PC)
----------------------------
- No installation required. No admin rights needed.
- Copy this entire folder to any Windows 10/11 PC and run a launcher
  directly. Nothing is written to the registry.
- Windows 10 version 1803 or later is required (for built-in curl.exe).


FILE ATTACHMENT
---------------
- Drag and drop files onto the browser window
- Or click the paperclip (clip) button in the input bar
- Supported: .py .r .txt .csv .json .md .log .ipynb .java .c .cpp
- Multiple files can be attached to one message
- The file content is read in your browser — nothing is uploaded anywhere


MODELS
------
Primary (recommended):
  Gemma 4 E4B-it Q4_K_M
  - Filename: gemma-4-E4B-it-Q4_K_M.gguf
  - Size:     ~5.0 GB
  - Strong general code quality on Python and R, fits comfortably in
    16 GB RAM, ~10-15 tok/sec on an i5 11/12th gen CPU.

Code specialist (Python-heavy):
  Qwen2.5-Coder-7B Instruct Q4_K_M (~4.7 GB, fastest)
  Qwen2.5-Coder-14B Instruct Q4_K_M (~8.9 GB, highest Python quality
    but slower and tighter on RAM).

Download via: download-model.bat


TWEAKING PERFORMANCE
--------------------
Each launcher .bat invokes one of start.py / start_powerbi.py /
start_r.py. To change threads, context size, or any llama-server
flag, edit the matching .py file (look for the subprocess.Popen
call with --threads, --ctx-size, etc.).

Useful defaults:
  THREADS = 8       - Set to the physical core count on the PC
                      (i7-12700: 8-10; i5-12400: 4-6; i5-11xx: 4).
  CTX     = 16384   - Context window in tokens. Drop to 8192 or 4096
                      if the server crashes or uses too much RAM.

The per-model context table lives in start.py (CTX_FOR dict).


UPDATING LLAMA.CPP
------------------
The AI engine (llama.cpp) occasionally releases faster CPU builds.
Run update-llama-cpp.bat for instructions or auto-download.
Manual: get llama-bXXXX-bin-win-avx2-x64.zip from
  https://github.com/ggml-org/llama.cpp/releases
Extract all .exe and .dll into the llama-cpp\ folder.


TROUBLESHOOTING
---------------
"Server not connected" shown in browser:
  - The launcher bat file is not running, or the model is still loading.
  - Wait a bit longer (Gemma 4 E4B can take 60s on first load).
  - Check that the model file exists in models\

"Model not found" error in the black window:
  - Run download-model.bat first.
  - Make sure the model file lives in models\ with the original filename.

Browser shows a blank page or nothing:
  - Open the URL from the black console window manually in Chrome/Edge.
  - Use a private/incognito window if a stale cache is suspected.

Download interrupted mid-way:
  - Re-run download-model.bat — it overwrites the partial file.


FOLDER STRUCTURE
----------------
Lab Rat\
  llama-cpp\           AI engine binaries (llama-server.exe etc.)
  models\              GGUF model files go here
  ui\
    index.html         Default Python / data-viz UI
    powerbi.html       Power BI helper UI
    r_lang.html        R programming UI
  start.py             Default launcher (used by launch-gemma4.bat /
                       launch-qwen.bat)
  start_powerbi.py     Power BI launcher (with msmdsrv schema bridge)
  start_r.py           R launcher
  launch-gemma4.bat    Start the default Python mode
  launch-qwen.bat      Start the default mode, prefers Qwen models
  launch-powerbi.bat   Start the Power BI helper
  launch-r.bat         Start the R helper
  download-model.bat   Download a model from Hugging Face
  update-llama-cpp.bat Update the AI engine binaries
  evaluations\         Exam-style question banks and model-response logs
  old\                 Previous version (kept for reference)
