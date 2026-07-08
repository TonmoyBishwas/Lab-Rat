# Blindspot-finding & prompt-tuning workflow

The reliability of Lab Rat comes from the per-course system prompt, not the
model. This is the loop that produced 25/25 on the DV Lab banks. Run it once
per course topic, and re-run it whenever the model or the syllabus changes.

Each numbered step says WHO does it: **you** (the student) or **Claude Code**
(start a fresh instance per topic so its context stays focused).

## The loop

### 1. Build a question bank — Claude Code
Give Claude Code the course materials (lab notebooks, past papers, the topic
list) and ask it to author `courses/<id>/evals/questions.md`:

- One question per `## Q<n> — DIFFICULTY · topic` heading.
- 8–12 questions per topic batch. Mix: ~40% EASY/MED mirroring likely exam
  phrasing, ~40% HARD composites (multi-part, subplot grids, "do all four
  techniques"), ~20% TRICK adversarial decoys.
- Adversarial decoys probe rules the prompt enforces: bait the model to use a
  banned import, a wrong formula, degrees instead of fractional hue, groupby
  on one-row-per-entity data, etc. A decoy PASSES when the model refuses the bait.
- End each question with an italic `*Probes ...*` note (the runner strips it
  before sending to the model — it's for the grader).
- Seed ideas from `courses/<id>/evals/weaknesses.md` if it exists.

### 2. Run the bank — you (or Claude Code, hands-off)
```
python eval\run_eval.py <course-id>
python eval\run_eval.py <course-id> --bank courses\<id>\evals\stress_test.md
```
Requires the model in `models\` and ~5–15 min per bank on the 16 GB CPU
machine. The runner boots/stops llama-server itself if none is running.

### 3. Grade the run — Claude Code (fresh instance)
Point it at the run log in `courses/<id>/evals/runs/` plus the question bank,
and ask for a verdict per question: **PASS / FAIL / PARTIAL** written into the
log's `### Verdict` slots. Grading criteria:

- Would the code run without error? (mentally execute it; check signatures,
  imports, undefined names)
- Does it do what was asked (right plot type, right formula, right count)?
- Did it respect course conventions and the response format?
- For decoys: did it refuse the bait?

Then have it summarize NEW failure modes (not one-off slips) at the bottom of
the log.

### 4. Patch the prompt — Claude Code (same instance as step 3)
Edit `courses/<id>/prompt.md` following the **escalation ladder** — go one rung
up each time the same failure survives a re-eval:

1. **Anti-pattern bullet** — one "Never ..." line in the anti-pattern list.
2. **Structural ban** — a conditional rule inside the relevant recipe
   ("if the question says X, technique Y is OFF-LIMITS, use Z").
3. **Worked example** — replace the abstract rule with literal correct code
   inside the recipe block.

Lesson learned (May 2026): the model follows the **recipe blocks** more
faithfully than the anti-pattern list. When a rule conflicts with something
the recipe implies, the recipe wins. Put critical constraints inside the
recipe, next to the code they guard.

Also distinguish **systematic failures** (same mistake ≥2 questions → patch)
from **one-off slips** (patching them is over-engineering — note and skip).

### 5. Targeted re-eval — you or Claude Code
```
python eval\run_eval.py <course-id> --only Q3,Q7
```
Repeat 3→5 until the bank passes. If a failure survives two patch rounds at
ladder rung 3, consider a model change for that course (see `models.json`) —
but prompt-first is the rule: model swaps reset ALL calibration.

### 6. Record — Claude Code
- Append the round's findings to `courses/<id>/evals/findings.md`
  (what failed → what patch → did it hold).
- Update auto-memory if a durable lesson emerged.
- Commit: prompt patch + run log + findings in one commit, message like
  `"Close <topic> eval loop: N/N pass"`.

## Bootstrapping a NEW course (e.g. Data Analytics, Data Structures)
1. `xcopy /E courses\_template courses\<new-id>` and fill in `course.json`.
2. Give Claude Code the course materials and ask it to draft `prompt.md` using
   `courses/dataviz-python/prompt.md` as the structural gold standard:
   role line → import discipline → response format → formula/reference blocks
   → recipes with worked examples → style rules → anti-patterns.
3. Add a `launch-<name>.bat` (copy `launch-python.bat`, change the course id).
4. Run the full loop above. Do NOT trust an untuned prompt in an exam —
   the DV Lab prompt needed 3 eval rounds before it held.

## Model changes
When a new model looks promising (see `models.json` notes):
1. Download it, then run the SAME banks with `--model <file>` while the old
   model's runs are still in `runs/` for side-by-side comparison.
2. A model change is accepted only if it passes every bank the incumbent
   passes. Speed (tok/s is in every run log) breaks ties.
3. Flip the order in `models.json` `priority` (or pin per-course in
   `course.json` `models`) only after that.
