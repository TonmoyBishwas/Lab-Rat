Lab Rat AI - Portable Offline Local AI
========================================

QUICK START
-----------
1. Download a model (at home, needs internet):
     Run: download-model.bat
     Option 1 (Gemma 4 E4B, ~5.0 GB) is the eval-validated safe default.
     Option 2 (Mellum 2, ~8.1 GB) is the faster/stronger candidate - run
     the eval bank before trusting it in an exam (see eval\ folder).

2. Pick a course and start the AI (fully offline, no admin needed):
     launch-python.bat   - Data Viz Lab, Python / numpy / matplotlib mode
     launch-r.bat        - Data Viz Lab, R / tidyverse / ggplot2 mode
     launch-powerbi.bat  - Power BI helper. Reads your open .pbix model
                           and grounds DAX/M answers in your actual tables.

     A black console window will appear - keep it open.
     Wait 30-90 seconds for the model to load.
     Open the URL it prints (http://localhost:8080) in Chrome/Edge.

3. Use it:
     Type your question and press Enter to send.
     Shift+Enter = new line inside your message.
     Drag & drop a .py, .r, .csv, .txt file onto the browser window.

4. Stop:
     Close the black console window, or press Ctrl+C inside it.

All modes share ports 11434 (AI) and 8080 (UI). Launching one mode
automatically stops the other. You cannot run two modes at once.


COURSES (the configurable part)
-------------------------------
Each mode is a folder under courses\:
    courses\<id>\course.json   - name, browser title, temperature, etc.
    courses\<id>\prompt.md     - the system prompt (the actual value here)
    courses\<id>\evals\        - question banks + eval run logs

launch-*.bat just runs:  python start.py <course-id>

To add a NEW course (Data Analytics, Data Structures, ...):
    1. Copy courses\_template to courses\<new-id> and fill both files in.
    2. Copy launch-python.bat to launch-<name>.bat, change the course id.
    3. Tune the prompt with the eval loop: eval\BLINDSPOT_WORKFLOW.md.
No Python changes are needed.


PORTABLE USE (USB / Lab PC)
----------------------------
- No installation required beyond Python (see SETUP_FOR_EXAM.md).
- No admin rights needed. No internet needed after models are downloaded.
- Copy this entire folder to any Windows 10/11 PC and run a launcher.
- Windows 10 version 1803 or later (for built-in curl.exe).


MODELS
------
Registry: models.json (priority order, context size, per-model flags).
Downloader: download-model.bat

  gemma-4-E4B-it-Q4_K_M      ~5.0 GB  SAFE DEFAULT (eval-validated 25/25)
  Mellum2-12B-A2.5B-Instruct ~8.1 GB  CODE CANDIDATE (MoE, fast on CPU,
                                      much stronger coding scores - needs
                                      eval validation before exam use)
  Qwen3.5-9B                 ~5.7 GB  REASONING CANDIDATE (thinking disabled
                                      via flag; slower - dense 9B)
  Qwen2.5-Coder-7B           ~4.7 GB  LEGACY FALLBACK

A model is picked automatically: course.json "models" list first, then
models.json "priority" order - first file that exists in models\ wins.


EVALUATIONS (finding the model's blindspots)
--------------------------------------------
  python eval\run_eval.py dataviz-python
runs the course question bank against the local model and writes a
timestamped log to courses\<id>\evals\runs\. Grade the log and patch the
prompt with a Claude Code instance following eval\BLINDSPOT_WORKFLOW.md.
Old (pre-redesign) eval logs live in evaluations\ - frozen history.


TWEAKING PERFORMANCE
--------------------
Threads auto-detect to half the logical core count (min 4). Context size
is per-model in models.json - drop it if the server crashes on 16 GB.


UPDATING LLAMA.CPP
------------------
Run update-llama-cpp.bat, or manually get
llama-bXXXX-bin-win-avx2-x64.zip from
  https://github.com/ggml-org/llama.cpp/releases
and extract all .exe/.dll into llama-cpp\. Current build: b8914.
Qwen3.5 needs >= b8000; --reasoning-budget needs >= b8148.


TROUBLESHOOTING
---------------
"Server not connected" in browser:
  - The launcher is not running, or the model is still loading. Wait.
  - Check the model file exists in models\
"Model not found":
  - Run download-model.bat first.
Blank browser page:
  - Open http://localhost:8080 manually; try a private window.
Download interrupted:
  - Re-run download-model.bat - it overwrites the partial file.


FOLDER STRUCTURE
----------------
Lab Rat\
  llama-cpp\           AI engine binaries (llama-server.exe etc.)
  models\              GGUF model files
  courses\             per-course config + prompt + evals  <- THE VALUE
  ui\index.html        generic chat UI (config-driven)
  ui\powerbi.html      Power BI mode UI
  eval\                eval runner + blindspot workflow
  start.py             generic launcher (python start.py <course-id>)
  start_powerbi.py     Power BI launcher (live .pbix schema bridge)
  launch-*.bat         double-click entry points
  models.json          model registry
  download-model.bat   model downloader (needs internet)
  update-llama-cpp.bat AI engine updater (needs internet)
  evaluations\         pre-redesign eval logs (frozen)
  old\                 previous versions (kept for reference)
