# Lab Rat AI

Portable, fully-offline LLM "exam appliance": llama.cpp + a quantized model on
a CPU-only Windows laptop, answering UIU lab-course questions. **The IP is the
per-course system prompt, not the code** — treat `courses/*/prompt.md` as the
most valuable files in the repo.

## Hard constraints (never violate)

- **Hardware:** i7 11th-gen, 16 GB RAM, NO GPU. Models must be ≤ ~9 GB GGUF
  Q4-class; dense models above 9B params are too slow (< 5 tok/s).
- **Lab PC:** no admin rights, no internet. Everything must run by
  double-clicking a `launch-*.bat` from this folder. No pip installs at
  runtime (Python stdlib only in `start*.py` and `eval/`), nothing written
  outside the folder, no registry.
- **No cloud calls at inference time.** The internet is only used at home for
  downloads (models, llama.cpp builds).

## Architecture

```
start.py                 generic launcher: python start.py <course-id>
                         boots llama-server (:11434) + UI/proxy server (:8080)
                         serves /api/config from the course folder
models.json              model registry: priority order, ctx, extra flags
courses/<id>/
    course.json          name, disguise title, tagline, temp, max_tokens,
                         optional per-course model pin
    prompt.md            THE system prompt (served to the UI via /api/config)
    evals/questions.md   eval bank(s) — format: ## Q<n> headings
    evals/runs/          timestamped answer logs from eval/run_eval.py
    evals/findings.md    what failed → what patch → did it hold
ui/index.html            single generic chat UI; fetches /api/config at load
ui/powerbi.html          special-mode UI (live .pbix schema, {MODEL_CONTEXT})
start_powerbi.py         special launcher for the Power BI bridge
eval/run_eval.py         runs a bank against the local model, logs answers
eval/check_answers.py    EXECUTES every ```python block in a run log and
                         reports which ones actually run (stdlib-only runner;
                         the answers themselves need pandas etc. on the dev box)
eval/BLINDSPOT_WORKFLOW.md   THE tuning loop — read before touching a prompt
launch-<course>.bat      double-click entry points
data/                    vendored seaborn CSVs — sns.load_dataset() needs the
                         network on first call, so the exam PC must either
                         pre-warm its cache or read these. See SETUP_FOR_EXAM.md
evaluations/             historical (pre-redesign) eval logs, kept as record
```

Adding a course = copy `courses/_template`, write `prompt.md`, add a
`launch-*.bat`, then run the eval loop. No Python changes needed.

## The tuning workflow (why you were probably launched)

The user runs one fresh Claude Code instance per course/topic to tune its
prompt. Follow `eval/BLINDSPOT_WORKFLOW.md` exactly. Key rules distilled from
months of eval rounds:

1. **Prompt-first, model-second.** Never suggest a model swap as the first
   response to wrong output — patch the prompt, re-eval, escalate only after
   rung-3 (worked example) fails twice.
2. **Escalation ladder** for repeated failures: anti-pattern bullet →
   structural ban inside the recipe → literal worked-example code.
3. **Recipes beat rule lists.** The model follows recipe blocks; it drifts
   from anti-pattern bullets when they conflict with the question's phrasing.
   Critical constraints go inside the recipe next to the code they guard.
4. **Patch systematic failures only** (same mistake ≥2 questions). One-off
   slips are noise; patching them bloats the prompt.
5. **Verify API signatures before pinning them in a prompt** (e.g. the joypy
   `ax=` saga: the "fix" was defensive over-engineering — the kwarg actually
   works). Check the installed package source when in doubt.
6. Small-model failure modes repeat across topics: formula hallucination,
   signature drift, over-aggregation, gratuitous imports, second-draft
   spirals. Author decoy questions for each when building new banks.

## Models (July 2026 state)

- `gemma-4-E4B-it-Q4_K_M` — **primary**. Eval-validated 25/25 on DV Lab
  (June 2026) and re-confirmed by a 3-way bake-off on `da-python` (July 2026).
- `Mellum2-12B-A2.5B-Instruct-MXFP4_MOE` — fastest (20.4 vs 11.3 tok/s) but
  **lost the bake-off**: weak instruction-following (IFEval 75.8 vs Gemma's
  96.7) makes it follow decoy questions and state a rule then violate it.
  Keep as a speed fallback; it would need its own tuning round.
- `granite-4.1-8b-Q4_K_M` — best published HumanEval that fits, but measured
  only 7.4 tok/s (4.9 warm through llama-server) and failed the ordinal
  decoy. Not competitive on this hardware.
- **Lesson from the bake-off:** for this appliance, *instruction-following*
  beats *code benchmarks*. The job is not "write hard algorithms", it is
  "follow a strict format and refuse a badly-worded question". Rank
  candidates on IFEval-like behaviour first, HumanEval second.
- `Qwen3.5-9B-Q4_K_M` — strongest reasoning per GB; thinking must be killed
  with `--reasoning-budget 0` (already in `models.json` flags; needs
  llama.cpp ≥ b8148). Dense 9B → slow; niche use only.
- Promotion rule: a challenger must pass every bank the incumbent passes,
  then flip `models.json` priority. A model swap invalidates ALL prompt
  calibration — plan a full re-eval, not a spot check.
- llama.cpp: repo is `ggml-org/llama.cpp`, current local build **b10148**
  (upgraded from b8914 July 2026; the b8914 tree is kept at
  `old/llama-cpp-b8914` for rollback). Mellum2 needs ≥ b9482 or it fails with
  `unknown model architecture`. `--flash-attn` takes a value in new builds.
  Update via `update-llama-cpp.bat`.
- Measured decode on the DEV box (Ryzen 9 7950X, `llama-bench -p 512 -n 128`),
  16 / 6 threads — the exam i7 will be slower, but the RANKING holds because
  decode is memory-bandwidth-bound:
  Mellum2 MXFP4 20.4 / 22.1 · Gemma 4 E4B 11.3 / 11.9 · Granite 4.1 8B 7.4 / 7.6.
  Note decode barely moves with thread count while prefill halves — so a big
  system prompt costs wall-clock on the FIRST question of a session only
  (llama-server reuses the cached prefix afterwards).

## Conventions

- Windows paths, batch launchers, `python` on PATH (installed per
  `SETUP_FOR_EXAM.md` on exam machines).
- Commit eval run logs and findings together with the prompt patch they
  motivated: `"Close <topic> eval loop: N/N pass"`.
- `evaluations/` is frozen history — add new logs under
  `courses/<id>/evals/runs/` instead.
- Auto-memory holds durable lessons (`system_prompt_calibration.md`,
  `model_choice.md`). Update it when an eval round teaches something new.
