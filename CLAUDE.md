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
eval/BLINDSPOT_WORKFLOW.md   THE tuning loop — read before touching a prompt
launch-<course>.bat      double-click entry points
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

- `gemma-4-E4B-it-Q4_K_M` — incumbent, eval-validated 25/25 (June 2026).
- `Mellum2-12B-A2.5B-Instruct-Q4_K_M` — challenger: JetBrains MoE, 2.5B
  active (fast on CPU), LiveCodeBench 69.9 vs Gemma's ~52, no
  chain-of-thought. **Not yet eval-validated** — compare with
  `python eval\run_eval.py <course> --model ...` before promoting.
- `Qwen3.5-9B-Q4_K_M` — strongest reasoning per GB; thinking must be killed
  with `--reasoning-budget 0` (already in `models.json` flags; needs
  llama.cpp ≥ b8148). Dense 9B → slow; niche use only.
- Promotion rule: a challenger must pass every bank the incumbent passes,
  then flip `models.json` priority. A model swap invalidates ALL prompt
  calibration — plan a full re-eval, not a spot check.
- llama.cpp: repo is `ggml-org/llama.cpp`, current local build b8914.
  `--flash-attn` takes a value in new builds. Update via `update-llama-cpp.bat`.

## Conventions

- Windows paths, batch launchers, `python` on PATH (installed per
  `SETUP_FOR_EXAM.md` on exam machines).
- Commit eval run logs and findings together with the prompt patch they
  motivated: `"Close <topic> eval loop: N/N pass"`.
- `evaluations/` is frozen history — add new logs under
  `courses/<id>/evals/runs/` instead.
- Auto-memory holds durable lessons (`system_prompt_calibration.md`,
  `model_choice.md`). Update it when an eval round teaches something new.
