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
                         serves /api/config, /api/datasets, /api/scan
dataset_scan.py          stdlib CSV scanner -> compact schema block appended to
                         the END of the system prompt (prefix-cache friendly).
                         Reads EVERY row (costs no context) but emits <=500
                         tokens. Exists because the model cannot see the file:
                         .map({'Male':0}) on a file holding MALE silently NaN-s
                         the column, and hand-typed one-hot names KeyError.
                         Also emits the DETECTED DELIMITER in its read_csv
                         hint — bank.csv is ';'-separated and reading it as CSV
                         yields one column and kills every task.
models.json              model registry: priority order, ctx, extra flags
courses/<id>/
    course.json          name, disguise title, tagline, temp, max_tokens,
                         optional per-course model pin
    prompt.md            THE system prompt (served to the UI via /api/config)
    evals/questions.md   eval bank(s) — format: ## Q<n> headings
    evals/runs/          timestamped answer logs from eval/run_eval.py
    evals/findings.md    what failed → what patch → did it hold
ui/index.html            single generic chat UI; fetches /api/config at load.
                         Remembers the Dataset box in localStorage (the exam
                         workflow is one question block per NEW TAB, and a tab
                         used to start with no schema), and WARMS THE MODEL on
                         load with a 1-token request so the ~21k-token prefill
                         happens while the student reads the paper instead of
                         in front of question 1.
ui/powerbi.html          special-mode UI (live .pbix schema, {MODEL_CONTEXT})
start_powerbi.py         special launcher for the Power BI bridge
eval/run_eval.py         runs a bank against the local model, logs answers
eval/grade_class_test.py EXECUTES an answer to the REAL Class Test paper and
                         checks 25 properties against values verified from the
                         CSV. Bank: courses/da-python/evals/class_test_1.md,
                         a real paper rather than a synthetic one (diamonds,
                         REGRESSION target, no missing values).
eval/grade_bank_ct2.py   The THIRD real paper, and the first from the NEW
                         Sept-2026 syllabus (Parts 3-4: ML + DL). bank.csv,
                         SEMICOLON-delimited, target ~88.5/11.5 imbalanced,
                         and the paper SUPPLIES STARTER CODE. Its defining
                         feature: it sends each question block as a SEPARATE
                         chat with NO history, because that is how the student
                         sits it (one block per new tab). Then concatenates the
                         three answers and executes them as one notebook, so a
                         rebuild or a renamed variable is a real traceback.
                         Bank: courses/da-python/evals/class_test_3_bank.md
                         Truth: eval/ref_bank_ct2.py (regenerates the key)
                         Port override: LABRAT_AI_PORT.
eval/grade_titanic_mldl.py  The GENERALIZATION bank — NOT a real sitting.
                         Written from the Part 3/4 handouts to answer "is the
                         prompt tuned to the bank paper or to the syllabus?".
                         Differs from bank CT-2 on every memorisable axis:
                         titanic (text + real NaNs), DecisionTree/KNN, (16,8),
                         10-fold, Purples, an architecture sweep, a REQUIRED
                         ColumnTransformer, and above all an INVERTED
                         threshold (T=0.35, so false positives must RISE where
                         the bank paper's 0.65 made them fall). Its key is
                         cross-checked against the figures the handouts
                         themselves print. Bank:
                         courses/da-python/evals/class_test_4_titanic.md
                         Truth: eval/ref_titanic_mldl.py
eval/grade_penguins_ct.py  The SECOND real paper (penguins, CLASSIFICATION
                         target, real NaNs, derived feature). Runs BOTH the
                         one-shot and the task-by-task flow, then appends a
                         PROBE to the generated code and inspects the live
                         df / X_train / y_train — so a column that silently
                         became all-NaN is caught by measurement, not reading.
                         Bank: courses/da-python/evals/class_test_2_penguins.md
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
7. **A rule that keeps drifting may be contradicted, not too weak.** Before
   adding an escalation rung, grep the prompt for the thing you are banning
   and check nothing else still shows it. The ladder assumes self-consistency.
8. **Grade by executing, not by reading.** The worst defect found so far — the
   target leaking into X, so "which feature correlates most with price"
   answers "price" at r=1.000 — produces code that runs clean and reads as
   correct. `eval/grade_class_test.py` exists because reading missed it.
9. **Check the config before blaming the prompt.** Answers truncating
   mid-question was ctx, not wording: a 12.4k-token prompt inside ctx 16384
   left 3.1k for output. Keep ctx well above prompt + longest answer, and use
   `--parallel 1` — llama-server's auto default opened 4 slots and re-prefilled
   the system prompt on each.
10. **One paper is not the course. Overfitting is the DEFAULT outcome.** The
   July 2026 round drove a 25-check grader to 24/25 on the diamonds paper; the
   student then sat a penguins paper and scored roughly 15/30. Every defect it
   exposed lived in a step diamonds does not have — real NaNs, case-sensitive
   categories, a derived feature, a classification target. Always tune against
   at least two real papers on different datasets, and treat a single-paper
   score as evidence about that paper only.
11. **Some facts cannot be prompted, only injected.** No escalation rung can
   teach the model that the CSV holds `MALE` and not `Male` — it has never seen
   the file. `dataset_scan.py` exists for exactly the class of bug that prompt
   engineering cannot reach. Reach for injection, not a fourth rung, whenever
   the missing knowledge is *about the data*.
12. **Split derivable from semantic.** Structure (column names, dtypes, missing
   counts, category values, ranges) belongs to the scanner. Meaning and
   counter-intuitive results (Fair < Ideal is an ordering, "Ideal is the
   CHEAPEST cut") belong in the prompt. Keeping derivable facts in the prompt
   is what made it 47 KB.
13. **The UI can manufacture what looks like a model failure.** `newChat()`
   opened with `if (isGenerating) return;`, so a slow generation silently made
   the "+" button a no-op, and there was no Stop control — the only escape was
   killing the console. Two reported "bugs", one cause. Check the client before
   re-tuning the prompt.
14. **A student under exam pressure types tasks one at a time.** They cannot
   paste a printed paper. Any workflow that only works when the whole paper
   arrives in one message will not survive the exam hall — the prompt must
   handle follow-up turns (continue, don't re-derive; always repeat imports).
15. **THE SYLLABUS CHANGES UNDER YOU, AND A STALE BAN IS WORSE THAN A GAP.**
   Sept 2026 the course moved to Parts 3-4 (ML + DL). The prompt still said
   *"sklearn is allowed ONLY for MinMaxScaler, StandardScaler,
   train_test_split"*, *"the labs stop at 'ready for a model'"*, and
   *"never import SimpleImputer / OneHotEncoder / ColumnTransformer"* — i.e.
   it explicitly forbade the exact APIs the new paper is built on. An omission
   makes the model improvise; a ban makes it refuse. When new material lands,
   FIRST grep the prompt for every API that material teaches and fix the
   contradictions, THEN add recipes. This is rule 7 at syllabus scale.
16. **A FRESH CHAT IS NOT A FRESH NOTEBOOK.** The student now opens a new tab
   per question block, so Q2 and Q3 arrive with zero history — no dataset
   description, no starter code, no memory of Q1. The model cannot recover
   that from context, only from the question's own wording ("Q3", "your
   trained model"). What makes it safe is a **canonical variable-name
   contract** (`X_train_s`, `mlp`, `rf`, `cm`, `pipe`) published in the
   prompt: the block that creates the name and the block that consumes it are
   both written by the same prompt, so they agree by construction. Pair it
   with "re-derive the cheap (predictions), never the expensive (read_csv,
   split, fit)" and a `# continues the notebook — uses: ...` header that makes
   a mismatch visible before the student runs the cell.
17. **Injection has to cover the DELIMITER, not just the column names.**
   `bank.csv` is semicolon-separated; `pd.read_csv('bank.csv')` returns ONE
   column named `age;"job";"marital";...` and every later task dies on a
   KeyError. The scanner sniffed the delimiter correctly for its own parsing
   and then emitted a comma-shaped `read_csv` hint anyway. Whatever the
   scanner knows about HOW TO LOAD the file has to reach the prompt too.
18. **A grader must be tested in both directions.** Score the hand-written
   gold answer (catches false positives — three slipped through in July) AND a
   deliberately broken one (catches false negatives — a check that never fires
   is worse than no check, because it reads as a pass).
19. **Check what else is on the port.** `llama-server` failed to bind because
   Ollama squats on 11434 by default. `start.py` sends its stderr to DEVNULL,
   so on such a machine the launcher would look like it started and the UI
   would just never connect. Eval scripts take `LABRAT_AI_PORT`.
20. **A discriminator rule will fire on the wrong text.** The rule "start a new
   notebook when the message quotes a dataset description" was written to catch
   Q1. It fired on Q3, whose printed form opens with a *"Context:"* paragraph
   about class imbalance and call budgets — so the model dutifully reloaded the
   CSV and rebuilt everything, and the score fell 51 -> 36. When a rule keys off
   "does the text mention X", check it against every question in the bank, not
   just the one it was written for. Prefer a signal the student cannot
   accidentally trip: **the question NUMBER**, which is unambiguous and
   overrides all the prose heuristics.
21. **Two banks that differ in one sentence beat two runs of one bank.** The
   typo bank scored 51/53 in the same session the clean bank scored 36/53, and
   the only relevant difference is that a student retyping Q3 in a hurry does
   not copy the Context paragraph. That turned "maybe sampling noise" into a
   located cause without a single extra model call. Keep the rushed-typing
   variant of every paper.
22. **A subtle check needs a unit test.** The "prose gets the False-Positive
   direction right" check was wrong twice: the CORRECT answer is phrased
   *"reduces False Positives and increases False Negatives"*, which a proximity
   match flags, and *"Increasing the threshold"* trips a clause match. It now
   ships with `_selftest_fp()` — three sentences that must pass, five that must
   fail. Six grader defects this round and three last round: **grader code is
   code, and it earns trust the same way the prompt does.**

23. **A TEST HARNESS IS PART OF THE SYSTEM UNDER TEST.** Three prompt patches
   in a row failed to move an unseen paper (36 -> 34 -> 33) because
   `load_bank()` stripped the `## Q2` heading before sending. The prompt rule
   being debugged says "the question number decides whether this continues" -
   and the harness was deleting the question number. Send the question exactly
   as the student pastes it, label included. Suspect the harness when patches
   stop working.
24. **An offline fallback only fails on the machine that has no internet.**
   `try: sns.load_dataset(X)[cols] / except: pd.read_csv(...)` applies the
   subset on the try line only. On the dev box the try branch wins and it looks
   right; on the EXAM PC the except branch is the ONLY one that ever runs, so
   the subset silently vanishes and titanic's 15 columns - including `alive`,
   the target in words - walk into X. Put every post-load step AFTER the
   try/except. Test the branch the target machine will take.
25. **Separate knowledge failures from reliability failures before reading a
   score.** On the unseen paper 9 checks failed in every run - all of them
   probe checks, which only run if the notebook executes. That is ONE failure
   counted nine times. Meanwhile 34 checks held in every run, including the
   inverted-threshold prose check that a memorising model cannot pass. Tally
   per-check across runs; a single score hides which half is broken.
26. **Build one adversarial bank per topic, not just real papers.** The titanic
   ML/DL bank exists only to attack memorisation: same difficulty, different
   dataset, different models, and an INVERTED answer (threshold 0.35, so false
   positives rise where the tuned-on paper's 0.65 made them fall). It found
   four real bugs the real paper structurally could not - including 24 above.
   A bank that shares a blind spot with the prompt cannot detect it.

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
