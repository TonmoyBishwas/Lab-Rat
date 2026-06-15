# Lab Rat AI
**One-Sentence Summary:** A portable, zero-install, fully-offline LLM "exam appliance" that runs a quantized model on a CPU-only laptop and answers a specific university lab course's test/assignment questions, with a unique Power BI bridge that reads the student's *live* open data model and feeds real table/column names into the AI's context.

## 🛠 Tech Stack & Capabilities
* **Languages:** Python (stdlib-only core, no pip required to run), vanilla JS + HTML/CSS (single-file UI), Windows Batch
* **Frameworks/Libraries:** Python `http.server` (custom reverse-proxy + static server hand-rolled, no Flask/FastAPI); `pyadomd` + `pythonnet`/`clr` (CLR ↔ .NET interop) + `psutil` for the Power BI bridge; client answers target numpy, pandas, matplotlib, seaborn, scipy, scikit-image, joypy
* **AI/ML Models:** Local GGUF quantized models only — Gemma 4 E4B-it Q4_K_M (recommended), Qwen2.5-Coder-7B/14B, Qwen2.5-7B/14B, Qwen3.5-4B/9B (all Q4_K_M). Served via llama.cpp (`llama-server.exe`). No cloud/API models, no internet calls at inference time.
* **Infrastructure:** 100% local/offline. Two-process architecture — llama-server on `:11434` + Python UI/proxy on `:8080`. Flat-file JSON persistence (`sessions.json`, capped at 50 sessions). Offline pip wheels bundled (`offline-deps/`) for air-gapped setup. Runs from USB, no admin rights, nothing written to registry.

## 🏗 Core Architecture & Logic
* **Launcher + proxy core (`start.py`):** Auto-detects the best available model from a priority list, looks up a per-model context-window size, boots `llama-server.exe` with performance-tuned flags (`--cache-type-k/v q8_0` KV-cache quantization, custom batch/ubatch sizes, `--flash-attn`, `--no-mmap`), polls `/health` until ready, then spins up a `ThreadingHTTPServer` that simultaneously (a) serves the single-file HTML UI, (b) **streaming reverse-proxies** `/v1/*` and `/health` to llama-server chunk-by-chunk, and (c) exposes a `/api/sessions` CRUD JSON store. Includes Windows port-killing (`netstat`/`taskkill`) to clear stale instances.
* **Browser UI (`ui/index.html`, single file ~1.6k lines):** OpenAI-compatible `/v1/chat/completions` SSE streaming client, `<think>`-block rendering for reasoning models, and **client-side file attachment** — drag-drop `.py/.csv/.json/.ipynb/.md` etc. is read in the browser and inlined into the prompt; nothing is ever uploaded. Chat history is round-tripped to the Python backend.
* **The real IP is the system prompt, not the code.** The bundled prompt encodes the exact curriculum: correct statistical formulas (geometric/harmonic/contra-harmonic means, `ddof` variance, z-score), 2D homogeneous transformation matrices with composition order rules, and battle-tested matplotlib/seaborn/skimage "recipes" with hard-won gotchas (e.g. `fill_betweenx` argument order, banning `joypy` inside subplot grids, hue arithmetic in `[0,1]` not degrees). These rules were reverse-engineered from documented small-model failure modes captured in `evaluations/`.
* **Power BI live-model bridge (`powerbi_bridge.py` + `start_powerbi.py`):** Discovers any *open* Power BI Desktop instance by reading its `msmdsrv.port.txt` workspace files (with a `psutil` listener-scan fallback), connects to the local SSAS Tabular engine over ADOMD.NET/MSOLAP, and queries `$SYSTEM.TMSCHEMA_*` DMVs to extract tables, columns, relationships (cardinality + cross-filter direction), and existing DAX measures. It renders this to compact markdown and injects it into a `{MODEL_CONTEXT}` placeholder in the prompt, so the AI writes DAX/M against the student's **actual** schema. Strictly read-only, no write-back, degrades gracefully if `pyadomd` is missing.
* **User interaction:** Double-click a `.bat` launcher → browser opens automatically to a local chat UI. Power BI mode adds a live "N tables, M cols detected" indicator and a Refresh button.

## 🚀 Current State & Maturity
* **Working, in-use personal prototype** — functional end-to-end with polished launchers, troubleshooting docs, and bundled offline dependencies. Not a throwaway script collection; it's a coherent three-mode appliance (general Python DV-lab mode + Power BI mode + R / tidyverse / ggplot2 mode). Each mode is one Python launcher script subclassing a shared `UIHandler`, plus one single-file HTML UI carrying a hand-tuned curriculum prompt — a clean fork-per-domain pattern.
* **Blockers to monetization as-is:**
  * Single-user, localhost-bound, **no auth, no multi-tenancy, no sandboxing** of the proxy.
  * Hyper-specialized: the value-add prompt is hardcoded to one university course (UIU DS 3522). It must be re-authored per customer/domain to be reusable.
  * Windows-only, with hardcoded paths (ADOMD.NET dir, `THREADS=8`) and bundled binaries — no installer, no auto-update, no model-license handling for redistribution.
  * Minor doc drift (README references models the downloader no longer offers).
  * Quality ceiling is bounded by small CPU-quantized models; reliability depends entirely on the prompt scaffolding.

## 💰 Monetization & Pivot Potential
* **White-label "Offline AI Tutor in a box":** Sell the harness and swap the system prompt per course/certification/domain. Target exam-prep companies, coding bootcamps, and universities that want a private, air-gapped study assistant students can run on lab PCs or USB sticks.
* **Air-gapped LLM appliance (the strongest generic pivot):** The whole stack is a clean, dependency-light pattern for "ChatGPT that never touches the internet." Sellable to privacy/compliance-sensitive orgs (defense, healthcare, finance, legal) and exam halls where cloud AI is banned.
* **The Power BI live-schema bridge is the genuinely novel, sellable IP:** a local, read-only "AI copilot that knows your real data model" is a packageable Power BI Desktop add-in / consulting deliverable. BI teams and analytics consultancies would pay for an offline DAX/M assistant grounded in their actual tables and relationships — this is the piece worth productizing on its own.
* **Freelance deliverable:** "Build me a private offline AI for [domain]" engagements, delivered as a portable folder. Low recurring cost (no API bills), high perceived value.
* **Likely buyers:** education/edtech (course providers, exam prep), compliance-bound enterprises needing on-prem AI, and BI consultancies wanting a model-aware reporting assistant.

## 🏷 Vault Tags
#local-llm, #offline-ai, #llama-cpp, #gguf-quantization, #prompt-engineering, #power-bi, #dax-automation, #python-backend, #reverse-proxy, #edtech
