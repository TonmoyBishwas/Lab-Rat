Lab Rat AI - Portable Offline Local AI
========================================

QUICK START
-----------
1. Download a model:
     Run: download-model.bat
     Choose option 1 (Qwen 3.5-9B, ~5.7 GB)
     Wait for download to finish.

2. Start the AI:
     Run: launch-qwen.bat
     A black window will appear — keep it open.
     Wait 30-90 seconds for the model to load.
     Your browser opens automatically.

3. Use it:
     Type your question and press Enter to send.
     Shift+Enter = new line inside your message.
     Drag & drop a .py or .txt file onto the browser window.
     The AI will read the file and answer questions about it.

4. Stop:
     Close the black console window, or press Ctrl+C inside it.


PORTABLE USE (USB / Lab PC)
----------------------------
- No installation required. No admin rights needed.
- Copy this entire folder to any Windows 10/11 PC and run
  launch-qwen3.bat directly. Nothing is written to the registry.
- Windows 10 version 1803 or later is required (for built-in curl.exe).


FILE ATTACHMENT
---------------
- Drag and drop files onto the browser window
- Or click the paperclip (clip) button in the input bar
- Supported: .py .txt .csv .json .md .log .ipynb .r .java .c .cpp
- Multiple files can be attached to one message
- The file content is read in your browser — nothing is uploaded anywhere


MODELS
------
Primary (recommended):
  Qwen2.5-7B Instruct Q4_K_M
  - Launcher: launch-qwen.bat
  - Size: ~4.7 GB
  - Best for Python and data visualization questions

Upgrade option:
  Qwen2.5-14B Instruct Q4_K_M
  - Launcher: launch-qwen.bat (auto-detects model)
  - Size: ~8.9 GB
  - Smarter answers, needs 16GB RAM free

Download via: download-model.bat


TWEAKING PERFORMANCE
--------------------
Open launch-qwen3.bat in Notepad and change:

  THREADS=8    Set to the number of physical cores on the PC
               (i7-12700: try 8-10, i5-12400: try 4-6)

  CTX=8192     Context window in tokens. Reduce to 4096 if the
               server crashes or uses too much RAM.


UPDATING LLAMA.CPP
------------------
The AI engine (llama.cpp) occasionally releases faster CPU builds.
Run update-llama-cpp.bat for instructions or auto-download.
Manual: get llama-bXXXX-bin-win-avx2-x64.zip from
  https://github.com/ggerganov/llama.cpp/releases
Extract all .exe and .dll into the llama-cpp\ folder.


TROUBLESHOOTING
---------------
"Server not connected" shown in browser:
  - The launcher bat file is not running, or the model is still loading.
  - Wait a bit longer (9B model can take 90s on first load).
  - Check that the model file exists in models\

"Model not found" error in the black window:
  - Run download-model.bat first.
  - Make sure the model filename matches what the launcher expects.
  - You can edit the MODEL= line in the .bat file to point to your file.

Browser shows a blank page or nothing:
  - Open ui\index.html manually in your browser.
  - Or copy the full path from Windows Explorer into the browser address bar.

Download interrupted mid-way:
  - Just run download-model.bat again — it will resume from where it stopped.


FOLDER STRUCTURE
----------------
Lab Rat\
  llama-cpp\       AI engine binaries (llama-server.exe etc.)
  models\          GGUF model files go here
  ui\              Browser UI (index.html)
  launch-qwen3.bat Start Qwen 3.5-9B
  launch-gemma4.bat Start Gemma 4 E4B
  download-model.bat Download models
  update-llama-cpp.bat Update the AI engine
  old\             Previous version (kept for reference)
