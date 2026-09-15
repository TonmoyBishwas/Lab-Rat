# Lab Rat AI — Data Analytics Lab (Python) evaluation findings

Course bootstrapped 2026-07-27. Prompt: `courses/da-python/prompt.md`.
Bank: `courses/da-python/evals/questions.md` (20 questions).
Grading per `eval/BLINDSPOT_WORKFLOW.md`, plus `eval/check_answers.py` which
executes every generated code block.

The exam questions for this course are UNKNOWN — only Lab1, Lab2 and the
assignment exist as signal. So the bank deliberately over-weights composite
questions and adversarial decoys rather than reproductions of the handouts.

---

## Pre-round: facts verified before the prompt was written

Rather than trusting the PDFs, every number pinned into `prompt.md` was checked
against the installed libraries (python 3.12.12, pandas 3.0.3, numpy 2.4.6,
seaborn 0.13.2, matplotlib 3.11.0, sklearn 1.9.0). All of Lab1's stated values
reproduced exactly (age nulls 177, median 28.0, fare Q1 7.9104 / Q3 31.0 /
upper 65.6344 / 116 outliers).

Five landmines were found that the handouts do not mention and that silently
break plausible answers:

| # | Landmine | Consequence |
|---|---|---|
| 1 | seaborn stores diamonds `cut` best-first (`Ideal…Fair`) and `clarity` `IF…I1` | `.cat.codes` is the EXACT REVERSE of the assignment's required map; correlation flips sign |
| 2 | `df.corr()` on a frame holding category/text columns | raises `ValueError: could not convert string to float: 'Ideal'` |
| 3 | `pd.get_dummies()` defaults to `bool` | assignment expects 0/1 ints, hence `dtype=int` |
| 4 | seaborn `palette=` without `hue=` deprecated (removed 0.14) — the handouts use exactly this | warning now, breakage later |
| 5 | pandas 3.0 reports text columns as `str`, not `object` | any answer asserting "object" is wrong on a modern install |

Three claims drafted into the prompt turned out to be WRONG on verification and
were corrected before any eval ran — worth recording because it validates
CLAUDE.md rule 5 ("verify API signatures before pinning them"):

- crosstab error is `values cannot be used without an aggfunc`, not the reverse.
- `train_test_split` stratify error is "least populated **classes** … have", plural.
- pandas 3.0 does **not** warn when grouping a category column without
  `observed=` — the drafted claim that it does was false.

Separately: `sns.load_dataset()` requires the network on first call. On a lab PC
with the internet cut, every question beginning `sns.load_dataset('titanic')`
fails. Mitigated three ways — CSVs vendored to `data/`, a cache pre-warm step
added to `SETUP_FOR_EXAM.md`, and a `try/except pd.read_csv` fallback taught in
the prompt.

---

## Round 1 — Mellum2-12B-A2.5B-Instruct-MXFP4_MOE

Runs: `runs/20260727-2058_…` (Q2, Q16) and `runs/20260727-2130_…` (Q16-Q18).

### Round 1a — plumbing check (Q2, Q16)

| Q | Verdict | Finding |
|---|---|---|
| Q2 missing values | PARTIAL | Correct code, correct median/mode, offline fallback used. But imported the entire allowed list for a 3-library answer, and a Notes line said the age mean "increased slightly from 29.7 to 29.4" — that is a decrease. |
| Q16 cat.codes decoy | **FAIL** | Followed the bait verbatim, then printed a self-contradictory reading: "a positive correlation indicates that better cuts (lower codes) command higher prices". |

### Patches applied after round 1a

1. **New section `=== WHEN THE QUESTION ITSELF ASKS FOR THE WRONG TECHNIQUE ===`**,
   placed immediately after RESPONSE FORMAT. The prompt previously never said
   what to do when the *question's own wording* conflicts with a correctness
   rule — the whole decoy category was undefended. Enumerates the four real
   baits and mandates: write the correct code, do the intended thing, note the
   substitution in one line.
2. **Rung-3 worked example** inside the encoding recipe with the measured
   numbers: `cat.codes` gives `+0.0535`, the correct map gives `-0.0535`, plus
   the reason (Ideal stones average 0.70 carat vs Fair 1.05, and carat drives
   price at r = 0.922) and an explicit ban on the inverted sentence.
3. **Import discipline sharpened** from a one-line rule to a per-question
   decision table, after gratuitous imports appeared in 2/2 answers.

### Round 1b — did the patches hold? (Q16, Q17, Q18)

| Q | Verdict | Finding |
|---|---|---|
| Q16 cat.codes decoy | PARTIAL | **Prose fixed, behaviour not.** Notes now reproduce the correct −0.0535 and the carat confounder — the worked example was absorbed as *facts*. But the code still calls `.cat.codes`, and Assumptions say "use cut.cat.codes directly as per instruction". Printed interpretation still inverted. |
| Q17 scale-then-split decoy | **FAIL** | Assumptions state "scaling must be fit on train only to avoid data leakage", then the code calls `fit_transform` on the full frame anyway with an approving comment. Says the right thing, does the wrong thing. Also dropped the Notes block entirely and invented an unrequested cleaning step. |
| Q18 triple decoy | **FAIL** | 1 of 3 baits refused (`numeric_only=True` used — patch held). Mean imputation and row deletion both followed. Worse: **fabricated statistics** — "std from 13.02 to 13.02", and `fare (-0.011)`, which is diamonds' depth-vs-price value recited into a Titanic answer. |

### New failure modes identified in round 1b

- **Facts absorbed, behaviour not.** A rung-3 worked example taught Mellum2 the
  numbers and the explanation but not the action it was meant to demonstrate.
  Escalation past rung 3 does not obviously exist for this model.
- **Stated-intent / executed-code divergence.** The model writes the correct
  principle in Assumptions and then violates it in the very next block. This is
  distinct from not knowing the rule — it demonstrably knows it.
- **Recitation of pinned numbers into the wrong context.** A fact-dense
  DATASET FACTS block gives the model plausible numerals to assert without
  computing them. Needs a rule: never state a numeric result the code did not
  print.
- All three answers **execute cleanly** — `check_answers.py` reports 3/3 RUNS.
  Confirms that runtime checking and correctness checking are independent axes
  and neither substitutes for the other.

---

## Model bake-off

Measured decode, `llama-bench -p 512 -n 128`, dev box (Ryzen 9 7950X, 32 GB).
The exam PC (i7 12th gen, 16 GB) will be slower in absolute terms, but decode
is memory-bandwidth-bound so the ranking transfers.

| Model | GGUF | decode @16t | decode @6t | prefill @16t |
|---|---|---|---|---|
| Mellum2-12B-A2.5B MXFP4_MOE | 6.55 GiB | **20.4 tok/s** | **22.1** | 225.9 |
| gemma-4-E4B-it Q4_K_M | 4.62 GiB | 11.3 | 11.9 | 251.9 |
| granite-4.1-8b Q4_K_M | 4.98 GiB | 7.4 | 7.6 | 146.8 |

Decode barely moves between 16 and 6 threads while prefill halves — so on the
exam PC the ~7k-token system prompt costs wall-clock on the FIRST question of a
session only; llama-server reuses the cached prefix afterwards. Observed
directly: Q16 took 98 s on a cold server, Q17/Q18 31 s and 45 s warm.

### Quality — identical decoy set, identical prompt

| | Q16 cat.codes | Q17 leakage | Q18 triple | Code that runs (20 Q) |
|---|---|---|---|---|
| **gemma-4-E4B** | PASS | PASS | PARTIAL | **19/20** |
| Mellum2-12B-A2.5B | PARTIAL | FAIL | FAIL | 18/20 |
| granite-4.1-8b | FAIL | — | — | not run (eliminated) |

Granite eliminated on both axes: 4.9 tok/s warm through llama-server on a
machine much faster than the exam PC, and it failed the ordinal decoy while
claiming "as the encoded cut value increases (moving from Fair to Ideal)" —
backwards, since cat.codes makes Ideal 0 and Fair 4. It also quoted −0.053, the
value for the CORRECT map, while its own code would print +0.053.

Mellum2's failures are specifically **instruction-following**, not knowledge:
on Q17 it wrote "scaling must be fit on train only to avoid data leakage" in
Assumptions and then called `fit_transform` on the full frame in the next
block. On Q19 it omitted `import pandas` entirely. This tracks its published
IFEval (75.8) against Gemma's (96.7) far better than it tracks EvalPlus, where
Mellum2 leads.

### DECISION: `gemma-4-E4B-it-Q4_K_M` remains primary

`models.json` priority unchanged. Mellum2 kept at position 2 as a speed
fallback (it is genuinely ~1.8x faster and would need its own tuning round);
Granite added but flagged not-recommended.

**Generalizable lesson, promoted to CLAUDE.md:** for this appliance, rank
candidate models on *instruction-following* first and code benchmarks second.
The workload is not "write a hard algorithm" — it is "obey a strict output
format and refuse a badly-worded question". The research that selected these
three candidates ranked them the opposite way and would have picked wrong.

---

## Round 2 — full 20-question bank, gemma-4-E4B

Run: `runs/20260727-2155_…`. 19/20 code blocks executed; every answer was a
single Assumptions/code/Notes triple (no duplicate-answer drift).

| Issue | Count | Detail |
|---|---|---|
| Crash | 1 | Q6: `pd.get_dummies(columns=['embarked'])` CONSUMES the column, then the next line printed `df[['embarked', …]]` → `KeyError: "['embarked'] not in index"` |
| Missing Notes block | 4 | Q5, Q6, Q10, Q13 ended after the code block |
| False claims in Notes | 4 | see below |

The false claims were the serious finding, because the *code* was right and
only the prose was wrong — the failure is invisible to execution checking:

| Claim | Reality |
|---|---|
| "Ideal cut commands the highest average price" (Q15) | Ideal is the CHEAPEST (3458); Premium is dearest (4584) |
| "The positive correlation confirms better cuts cost more" (Q16) | The correlation is −0.0535, negative |
| "bill depth shows moderate POSITIVE correlation with body mass" (Q19) | −0.472, negative (Simpson's paradox across species) |
| "sex_encoded (0.252) … pclass (−0.189)" (Q11) | Actually 0.543 and −0.338 |

### Patches applied after round 2

4. `get_dummies` consumption caveat, inline in the encoding recipe.
5. All three parts of the response format made explicitly mandatory.
6. Counter-intuitive facts pinned in DATASET FACTS (diamonds mean price by cut,
   penguins bill_depth vs body_mass).
7. **New `=== NOTES DISCIPLINE ===` section.** Root cause: the model never
   executes anything, so every claim about a result is a guess. Rules: no
   numerals in Notes unless pinned for that exact dataset/column; never assert
   a direction or ranking the code did not print; use `idxmax()`/`idxmin()` and
   let the output answer.

### Round 2b — did they hold? (Q5, Q6, Q10, Q11, Q13, Q15, Q16, Q19)

Run: `runs/20260727-2259_…`. **8/8 executed cleanly** (Q6 crash fixed),
**8/8 had a Notes block** (was 4 missing), **0 numerals appeared in any Notes**.
Q16 became fully correct: "The printed negative correlation indicates that
better cuts (higher encoded values) are associated with slightly lower average
prices, as top-cut stones are typically cut from smaller rough diamonds."

One residual: Q15 still asserted "higher quality cuts … associated with higher
prices". Patched at rung 3 with a dedicated *Average price by cut* recipe
(assignment Task 7) printing `idxmax()`/`idxmin()` and stating the true
ordering. Re-eval `runs/20260727-2320_…`: fixed — "while 'Ideal' is the highest
quality, it is associated with the lowest average price, which is
counter-intuitive due to the effect of carat size."

---

## Round 3 — held-out generalization test through the live UI

Asked a question that appears in no bank and no handout, through
`start.py da-python` and the UI proxy rather than the eval runner:

> "Using the tips dataset from seaborn, show the average tip percentage
> (tip/total_bill) by day and by time, and say which combination tips best."

The code was correct and computed the winner with `idxmax()`. **But the Notes
asserted "highest on Saturday during Dinner" — which is the WORST group
(0.1532). The true answer is Friday Lunch (0.1888).** The student would have
copied prose that contradicted their own printed output.

This proved the round-2 Notes rules were insufficient on datasets with no
pinned facts: banning numerals stopped fabricated *numbers* but not fabricated
*directions*. Rung-3 patch — rule 3 of NOTES DISCIPLINE:

> If the code computes the answer, the Notes must NOT restate it. Point at the
> printed line; never name the winner. The only named results allowed are those
> pinned in DATASET FACTS for that exact dataset.

Re-tested with the identical question: **held.** Notes now read "The printed
tables show the average tip percentage for each category. The code identifies
the specific combination … based on the computed data."

### Status

Prompt: 519 lines. Bank: 20 questions, all parsing. Execution: 19/20 on the
full bank, 8/8 and 1/1 on re-evals, no crashes outstanding.

### Residual, intentionally NOT fixed (over-tuning risk)

- Answers run long (~1300 tokens, ~130 s each on the dev box). Acceptable: the
  student confirmed the exam is 1 hour / 20-25 marks and Gemma's pace is fine.
- The prompt is ~8.5k tokens, so the FIRST question of a session costs about a
  minute of prefill. Mitigated by documentation (warm it up before the exam
  starts) rather than by cutting content, since every block earned its place
  from an observed failure.
- Mellum2 remains untuned. If a future exam is time-critical it is the obvious
  candidate, but it needs its own full round — prompt calibration is
  model-specific and none of the above transfers.

---

## Round 4 — held-out test on `mpg`, and the `include='object'` warning
*2026-07-28*

### How it was found

Not by the bank. The student asked for an exam-style question to test the
appliance live, so I wrote an assignment-shaped paper on seaborn's **`mpg`** —
a dataset the prompt had never been tuned on (the bank uses titanic, diamonds,
penguins, tips) — with five traps planted deliberately:

| Trap | Result |
|---|---|
| `name` has 305 unique values in 398 rows — one-hot would explode it to 300+ columns | **caught** — excluded from features, justified as an identifier |
| `cylinders` is int64 with only 5 distinct values — bait to encode it | **caught** — left numeric |
| Scale-before-split leakage | **caught** — split first, fit on train only, explained why |
| Bare `df.corr()` raises on the mixed frame | **caught** — `numeric_only=True` |
| "Strongest **negative** correlation with mpg" — intuition says horsepower | **caught** — computed it, got `weight (-0.832)`, and the Notes refused to name a winner it had not computed |

Every number was verified independently and every one was right: 6 missing in
`horsepower` (1.51%), grouped medians europe 76.5 / japan 75.0 / usa 105.0,
acceleration Q1 13.825 / Q3 17.175 / bounds 8.80–22.20 / 7 outliers,
shapes (318, 9) and (80, 9).

That is the first fully held-out pass: new dataset, new column names, traps it
had never been shown, and not one fabricated statistic.

### The one real defect

`df.describe(include='object')` emits a **`Pandas4Warning`** on pandas 3.0.3:

```
For backward compatibility, 'str' dtypes are included by select_dtypes when
'object' dtype is specified. This behavior is deprecated ...
```

Cosmetic — the output is correct — but it drops a yellow warning block into the
notebook that reads like an error to a marker, and it is **systematic**, not a
one-off: `describe(include='object')` comes straight out of the Lab1/Lab2
handouts, so it would have fired on every profiling answer.

Verified on pandas 3.0.3 before patching (CLAUDE.md rule 5):

| call | result |
|---|---|
| `include='object'` | correct output, **warns** |
| `include=['object','str']` | correct output, silent |
| `include='str'` | correct output, silent |

### The patch

Two rungs of the ladder, because the recipe is what the model actually follows:

1. **Inside the profiling recipe**, next to the code it guards — `include='str'`
   with the reason, plus the same rule extended to `select_dtypes`.
2. **Anti-pattern bullet** — never pass `include='object'` to `describe()` or
   `select_dtypes()`.

Worth noting *why* the existing prompt did not already cover this. It said:

> Do NOT assert a dtype is 'object' — modern pandas reports text columns as 'str'.

That bans *asserting* `'object'`; it says nothing about *passing* `'object'` as a
selector. Two different mistakes, one covered. Cheap lesson: a rule phrased
about the model's prose does not automatically constrain the model's code.

### Targeted re-eval — Q1, Q7, Q12 + Q19 as a regression control

`python eval\run_eval.py da-python --only Q1,Q7,Q12,Q19`
Log: `runs/20260728-2322_already-running-server_questions.md`

| Q | verdict | note |
|---|---|---|
| Q1 | PASS | clean, zero warnings |
| Q7 | PASS | patch landed — emitted `include='str'` |
| Q12 | PASS | all diamonds values verified |
| Q19 | PASS | unseen dataset, correct column names, no invention |

**4/4 execute, 4/4 with zero warnings, 4/4 format-compliant** (one code block,
Assumptions + Notes). Server confirmed as `gemma-4-E4B-it-Q4_K_M` before
trusting the run — the runner attached to an already-running server, so the
model in use was checked via `/v1/models` rather than assumed.

### A bug in the bank, not the model

Q7's expected-value note claimed `embarked` top S freq **646**. The model said
644. The model was right — `titanic['embarked'].value_counts()` gives S 644,
C 168, Q 77. Corrected in `questions.md`.

Standing reminder: when an answer disagrees with the key, verify the key.

### Status

Prompt: 526 lines. Bank: 20 questions. Execution 19/20 on the full bank; 4/4 on
this round plus a held-out `mpg` paper graded 23-24/25. No crashes outstanding.

---

## Round 4b — the round-4 patch was wrong, and the ORIGINAL was worse
*2026-07-28*

### What happened

The round-4 patch changed the profiling recipe to `describe(include='str')`.
While smoke-testing the USB deployment on `tips` — not a bank question, just a
sanity check — the generated code crashed:

```
ValueError: No columns match the specified include or exclude data types
```

seaborn does not store text columns uniformly. Measured across five datasets:

| dataset | text columns stored as |
|---|---|
| titanic, penguins, mpg | `str` |
| **tips, diamonds** | **`category`** |

So `include='str'` crashes on tips and diamonds. And checking the pre-patch
state honestly: **`include='object'` crashes on them too.** The original prompt
was not merely emitting a cosmetic warning — it would have produced a hard
`ValueError` on any "summary statistics for the categorical columns" question
about **diamonds, which is the assignment dataset.** Round 4 replaced one
crashing form with a different crashing form and reported a pass.

Full matrix, measured on pandas 3.0.3:

| include= | titanic | penguins | mpg | tips | diamonds |
|---|---|---|---|---|---|
| `'object'` | warns | warns | warns | **CRASH** | **CRASH** |
| `'str'` | ok | ok | ok | **CRASH** | **CRASH** |
| `['object','str','category']` | ok | ok | ok | ok | ok |

Only the three-type list is correct everywhere. It is also silent, and on
titanic it returns *more* columns (7 vs 5) because it picks up `class`/`deck`,
which are `category` and were being silently dropped from the categorical
summary all along.

### Why the bank missed it

Q7 profiles **titanic** (`str` — passes). Q12 profiles diamonds but never calls
`describe(include=...)`. Q1 asks only for numeric statistics. Every profiling
question in the bank happened to sit on the safe side of the split, so the
round-4 re-eval returned a clean 4/4 on a prompt that crashed on the assignment
dataset.

**Added Q21** — profiling `tips` specifically — so this cannot escape again.

### The fix

`include=['object', 'str', 'category']` in the recipe, with the dtype split
spelled out next to the code and the crash message quoted, plus a rewritten
anti-pattern bullet banning any single-dtype selector for text columns.

### Verified end to end, through the USB deployment

Both questions asked through the real UI proxy on the stripped USB copy, then
every generated block executed:

| question | emitted | result |
|---|---|---|
| tips profiling | `include=['object', 'str', 'category']` | RUNS CLEAN |
| diamonds profiling | `include=['object', 'str', 'category']` | RUNS CLEAN |

### Lessons

1. **A green eval on a tuned bank proves the bank, not the prompt.** Round 4
   reported 4/4 while shipping a crash on the assignment dataset. The escape was
   caught by an unplanned smoke test on an unlisted dataset.
2. **When patching an API detail, enumerate the inputs the rule will meet.**
   "Verify the signature" (CLAUDE.md rule 5) was followed — on titanic only.
   One dataset is not verification when the datasets differ.
3. **Check whether the bug predates the patch.** The instinct was "I introduced
   a regression". Half true: the patch was wrong, and so was the original, in a
   worse way. Recording only the first half would have left the real severity
   unstated.

---

## Round 5 — tuned against the REAL Class Test 1 paper
*2026-07-29*

### The input

A copy of another section's Class Test 1 (60 minutes, 30 marks, diamonds).
Same course, same dataset; the student's paper is expected to match its shape
with different wording. This replaces guesswork with the actual target.

Three things about it that no synthetic question had captured:

1. **It supplies its own loader** — `pd.read_csv('diamonds.csv')` against a file
   on the lab drive, not `sns.load_dataset`. After read_csv the text columns are
   `str`, not `category`: the exact opposite of the seaborn path.
2. **The split happens in Part 1**, so all of Part 2's EDA sits downstream of it
   and must read the target from `y_train`.
3. **Task 5c asks for a correlation over "X_train along with price"** — the
   target has to be put back temporarily.

### Method

Built `eval/grade_class_test.py`: executes the generated code and checks 25
specific properties against values verified from the CSV. This mattered more
than usual, because the decisive failure produces code that **runs clean and
looks right**. Reading the answer would not have caught it.

Baseline, before any tuning: **15/25**.

### What was actually broken

| # | Defect | Consequence |
|---|---|---|
| 1 | `price` left inside `X` | Task 3b wrong; target leaks into every feature |
| 2 | Task 5c printed `price (r = 1.000)` | 2-mark question confidently wrong; answer is carat 0.9216 |
| 3 | Notes said "carat is strongest" while the code printed "price" | asserting a result contradicting its own output |
| 4 | `palette=` without `hue=` | FutureWarning blocks under the cells |
| 5 | answer truncated mid-Task-5b | Tasks 5b and 5c simply missing |

Defect 1 is the root of 2 and 3. The model built features with a hand-written
exclusion list; across runs it variously forgot the target, or dropped the
one-hot `color` columns it had just created. Nothing raises an error either way.

Defect 5 was not a prompt problem at all. The system prompt had grown to
**12,366 tokens**; at ctx 16384 that left 3,115 for output, and a 30-mark answer
needs ~3,500. `max_tokens: 4096` was a fiction — the real ceiling was the
context. A two-turn conversation would have overflowed outright.

### Patches

Prompt (each at the rung its failure had earned):
- split recipe mandates `.drop(columns=[target])`, bans feature lists and
  comprehensions, and states the consequence
- encode and cap **in place**, so no stale text or duplicate column reaches X
- new recipe: `X_train.assign(price=y_train).corr()` to build the matrix,
  `.drop('price')` before `idxmax()` to read it
- new recipe: histogram of the target after a split reads `y_train`
- new **RECIPES PART 3** — the whole class-test pipeline end to end, plus the
  five places marks go missing in that shape
- `palette=` escalated to a structural ban

Config:
- ctx 16384 -> **32768**, max_tokens 4096 -> **8192**. Cost: nothing measurable.
  llama-server private RAM 5.54 GB at 32k vs 5.62 GB at 16k — KV is lazy.
- **`--parallel 1`**. llama-server's auto default allocated FOUR slots and
  rotates requests across them; each new slot re-prefills the 12k-token system
  prompt from cold. One user needs one slot. After: turn 1 296s, turn 2 209s.
- proxy read timeout 300s -> 1800s.

### The palette lesson — contradictions beat escalation

`palette=` kept drifting back after being banned. The cause was not weak
wording: the PART 3 checklist still said "pass hue= alongside palette=" while
the recipe said "omit palette=", and two worked examples elsewhere still showed
it in use. **The model followed whichever it saw last.** Escalating the ban
harder would never have worked. Making the prompt internally consistent — every
categorical-plot example carrying no palette, the checklist agreeing with the
recipe — fixed it immediately.

Before adding a rung, grep the prompt for the thing you are banning and check
nothing else contradicts it.

### Result

| | grader |
|---|---|
| baseline | **15/25** |
| after tuning | **24/25**, repeatable |

Also verified:
- **two-turn workflow** (Part 1, then Part 2 as a follow-up): 7/7 continuity
  checks — Part 2 uses `X_train`/`y_train` from turn 1, does not reload the CSV,
  and the two blocks execute cleanly when concatenated as a notebook would
- **CSV robustness**: run against a copy carrying an `Unnamed: 0` index column
  and against one with 5% NaN injected into carat/price — both execute clean and
  still answer carat

### The one gap, and why it is not being escalated

The model will not emit the `Unnamed: 0` drop line, at three escalation rungs
(recipe aside, mandatory recipe line, anti-pattern bullet, then moved next to
read_csv itself). It reads as defensive code the question did not ask for,
which collides with this prompt's own "answer only what was asked" discipline.

Impact is small and was measured, not assumed: against a CSV that HAS an index
column the code runs clean and still answers carat 0.9216 — it simply carries
one extra column in X, which is arguably compliant with "all other columns as
predictors" anyway. A fourth rung would be the joypy mistake again (defensive
over-engineering against a problem that barely exists), so it is documented as
a ten-second human check in DEPLOY_README.txt instead.

### Status

Prompt: 12.4k tokens. Banks: 21 synthetic questions + the real paper.
Class Test grader 24/25 repeatable, two-turn 7/7, full paper ~8 minutes.

---

## Round 6 — the penguins Class Test (2026-07-30)

### What happened

The student sat the real exam and scored roughly **15-19/30**. Round 5 had
driven the diamonds grader to 24/25; that number turned out to be evidence
about *the diamonds paper*, not about the course. See CLAUDE.md lesson 10.

Reconstructed from the student's submitted notebook, cell by cell:

| Defect | Cost | Owner |
|---|---|---|
| `.map({'Male':0,'Female':1})` on a file holding `MALE`/`FEMALE` — every row NaN, no error | ~2 marks + everything downstream | prompt |
| Hand-typed `island_Boscoe` / `island_Targersen` -> KeyError, cell dead | ~3 marks | prompt |
| 4-column impute list emitted with 3 (`bill_depth_mm` dropped) | ~1 mark + failed the "confirm zero missing" check | prompt |
| `train_test_split` / `MinMaxScaler` used without imports in a follow-up turn | ~4 marks | workflow |
| `.plot(kind='bar')` where the paper named `sns.barplot()` | ~1 mark | prompt |
| top correlation pair printed twice, mirrored | presentation | prompt |
| `palette=` without `hue=` — still drifting after 4 rungs | warning block | prompt |

Only ONE of these (missing imports) came from typing tasks one at a time.
Pasting the whole paper would have fixed that one and none of the others.

### The unfixable-by-prompting class

`penguins.csv` stores `MALE`/`FEMALE`; `sns.load_dataset('penguins')` returns
`Male`/`Female`. No escalation rung can teach a model what is inside a file it
has never read. That is why `dataset_scan.py` exists — it scans every row
(costing zero context) and appends <=250 tokens of real column names, dtypes,
missing counts and category values to the END of the system prompt, where it
does not disturb the cached prefix.

The CSV that reproduces the bug was sitting in `data/` the whole time and no
bank ever touched it.

### Round 6 changes

- `dataset_scan.py` + `/api/scan` + a Dataset box in the UI
- prompt: schema block is authoritative; case-robust `.map()` with a mandatory
  `print("unmapped:", ...)` verification line; derive one-hot names rather than
  typing them; count hand-typed column lists
- prompt: **every block carries its own imports**, always
- prompt: FOLLOW-UP section — continue from prior state, never re-derive
- prompt: output brevity — no preamble, Notes only when the question asks
- prompt: use the plotting function the paper NAMES
- UI: Stop button, `isGenerating` reset in `finally`, `+` cancels in flight
  (`newChat()` opened with `if (isGenerating) return;`, which is why the "+"
  looked broken — one cause, two reported symptoms)

### First measurement against the new bank (pre-patch prompt)

| mode | score | wall clock |
|---|---|---|
| one-shot (whole paper) | **23/31** | 495 s, 7.5 KB, one block |
| multi-turn (task by task) | **30/31** | 5 turns, 72-140 s each, ~1.5-2 KB each |

(Two grader false positives were found and fixed while reading these: a derived
identifier `island_dummy_cols` counted as an invented column name, and the
"highest correlated pair" regex matched the `--- Task 5c ---` banner instead of
the answer line. Numbers above are post-fix. Saved runs can be re-scored without
the model: `python eval\grade_penguins_ct.py regrade <run.md>`.)

**MULTI-TURN BEAT ONE-SHOT, and this reverses the standing advice.** The old
DEPLOY_README said to paste the whole paper and explicitly warned against
task-by-task. On this bank task-by-task scores higher *and* returns runnable
code every ~2 minutes instead of nothing for 8. The FOLLOW-UP rules are why:
each answer is short, carries its own imports, and continues from the existing
variables rather than rebuilding them. Its only genuine failure was the guessed
balance verdict (below) — the one-shot lost 7 further checks to a crash.

All 20 static checks passed in both modes — every new rule landed. Two real
defects remained:

1. **Crash.** The model took the `np.triu` pair recipe and added an
   indirection: `top = pairs.abs().idxmax(); names = top[0]` (a string), then
   `corr.loc[names[0], names[1]]` -> `KeyError: 'l'`. Killed Part 2's output.
   Fix (rung 2, structural): the recipe now uses tuple unpacking,
   `f1, f2 = pairs.abs().idxmax()`, so there is no index to get wrong.

2. **Guessed verdicts, inside `print()`.** Four `print("Observation: ...")`
   lines, none computed, **two false**: "classes appear relatively balanced"
   (they are 152/124/68, ratio 2.24) and "Gentoo exhibits the longest mean bill
   length" (Chinstrap 48.83 > Gentoo 47.50). NOTES DISCIPLINE was scoped to the
   Notes block, so the model moved the guess into a string literal in the code.
   Fix: the rule now explicitly covers `print()`, with computed recipes for the
   three verdict shapes (balance ratio, group idxmax, mean-vs-median skew).
   All three verified against the real data before being pinned.

Multi-turn answers came back 94-108 s and ~1.5 KB each, versus 495 s and 7.5 KB
for the one-shot — the brevity and no-re-derivation rules are working.

### Iteration 2 — the patch fixed one mode and broke the other

| mode | pre-patch | after patch 1 |
|---|---|---|
| one-shot | 23/31 | **29/31** |
| multi-turn | 30/31 | **21/31**  <- regression |

The crash fix held (one-shot now executes clean, `culmen_ratio` present,
train/test 275/69, the pair line prints `flipper_length_mm and body_mass_g`).
But multi-turn fell nine checks, and both causes were mine.

**Cause 1 — I contradicted myself.** CLAUDE.md lesson 7 says a rule that keeps
drifting may be contradicted rather than too weak; this time the contradiction
was in the rule I had just written. To justify repeating imports I wrote:

    "Each answer goes into a SEPARATE notebook cell that may be run in any
     order or after a kernel restart."

That is a licence to reload the data, and the model took it — every follow-up
turn opened with

    # --- Load Data (Assuming df from Task 1 is available and clean) ---
    try:
        df = pd.read_csv('penguins.csv')

which resets df to the raw file and destroys the imputation, the encoding and
the derived column from earlier cells. Concatenated as a notebook runs them,
Task 3 then died with `KeyError: 'culmen_ratio'` and took six probe checks with
it. Note the model even wrote "assuming df from Task 1 is available" and then
reloaded anyway — it was following the letter of my justification.

Fix: imports repeat because they are IDEMPOTENT; cells run top to bottom, in
order, once; `pd.read_csv` appears exactly once, in the first answer. The
"any order / kernel restart" wording is gone.

**Cause 2 — suppressing a symptom without installing the replacement.** Banning
guessed verdicts inside `print()` worked: they stopped appearing there. The
model moved them into `# Notes:` comments and the prose Notes block instead,
still uncomputed and still wrong ("Chinstrap has the largest average flipper
length" — it is Gentoo; "Gentoo having the longest average bill length" — it is
Chinstrap; and a fabricated r = 0.7911 when the code had printed 0.8663).

The rule lived in NOTES DISCIPLINE, a rule-list section. Per CLAUDE.md lesson 3
the model follows RECIPES. So the computation now sits inside the plotting
recipes themselves — countplot ships with a balance ratio, boxplot with
`groupby().mean().idxmax()`, hist with a mean-vs-median skew test — and the ban
covers print strings, comments AND Notes.

**Also fixed:** the model had begun wrapping its Notes in a bare ``` fence,
which made every turn parse as two code blocks. RESPONSE FORMAT now states
exactly one fenced block per answer, containing only runnable Python.

### Iteration 3 — both modes fixed, then a worse defect surfaced

| mode | pre-patch | patch 1 | patch 2 |
|---|---|---|---|
| one-shot | 23/31 | 29/31 | **30/31** |
| multi-turn | 30/31 | 21/31 | **31/31** |

The contradiction fix worked exactly as intended: `read_csv` now appears ONCE
across all five turns, each turn emits exactly one fenced block, and every
verdict is computed (`Max/Min ratio = 2.24 -> IMBALANCED`, `Largest:` via
`idxmax()`, skew from mean-vs-median).

The one remaining one-shot miss is cosmetic: it drew the histogram with
`ax.hist(...)` through `plt.subplots()` rather than `plt.hist(...)` directly.
Same function, 25 bins, correct. Not escalated — a single deviation in the
secondary mode is noise under CLAUDE.md rule 4.

### The defect the grader could not see

Reading the 31/31 run by hand turned up something the harness was structurally
blind to. **The model appends an invented output transcript after the code**,
and the invented values are wrong while the code above them is right:

    Species counts:            <- FABRICATED, and two species are SWAPPED
    Adelie      152
    Chinstrap   124                (really Gentoo 124)
    Gentoo       68                (really Chinstrap 68)
    Largest: Chinstrap         <- WRONG; it is Gentoo, 217.18 vs 197.00

Also fabricated: `mean body_mass_g 3546.7` (really 4202.6) and
`r = 0.9896` (the executed code printed 0.8783).

This is the most dangerous shape any of these defects has taken. The code is
correct and prints the right answer, so every execution-based check passes —
but the student reads the chat window, not their own stdout, and copies the
fabricated winner onto their script. `eval/grade_class_test.py`'s founding
lesson was "grade by executing, not by reading"; the corollary is that
executing alone is also not enough. **Grade the prose too.**

Fixes:
- prompt: an explicit structural ban on writing any output/transcript/"Output:"
  section, quoting the real swapped-species example, in RESPONSE FORMAT where
  the model actually follows structure
- grader: two new prose-only checks that strip the code fences first — one for
  pasted-output tells (`dtype:`, `Name: count`, `Output:`), one asserting no
  wrong winner is named. Re-scoring the "31/31" run against them gives 31/33.

### Iteration 4 — closed

| mode | pre-round | patch 1 | patch 2 | **final** |
|---|---|---|---|---|
| one-shot (whole paper) | 23/31 | 29/31 | 30/31 | **33/33** |
| multi-turn (task by task) | 30/31 | 21/31 | 31/31 | **33/33** |

(33 checks, not 31 — the two prose checks were added after iteration 3 exposed
the fabricated-transcript defect. Both modes pass them.)

Confirmed in the final run: `read_csv` appears exactly once across all five
turns; one fenced block per answer; every verdict computed rather than asserted;
no invented output transcript; `sex` survives encoding with 0 unmapped; the
correlation pair prints `flipper_length_mm and body_mass_g`.

One further grader false positive was fixed while closing: the one-shot wrote
`target = 'species'; X = df.drop(columns=[target])`, which the static regex
rejected for not containing the literal. Correctness was never in doubt — the
probe reported 9 columns in X with `species` absent. Three grader false
positives in total this round (`island_dummy_cols`, the banner-matching pair
regex, and this one); each inflated or deflated a score before being caught by
reading the runs by hand. **A new grader needs its own review pass before its
numbers mean anything.**

Deliberately not escalated: the one-shot draws its histogram via
`plt.subplots()` + `ax.hist(...)` rather than `plt.hist(...)`. Same function,
correct bins, single occurrence in the secondary mode — noise under rule 4.

### Deployed

`E:\Data Analytics` synced and verified byte-identical: `start.py`,
`dataset_scan.py` (new), `ui/index.html`, `courses/da-python/prompt.md`,
`course.json`, `models.json`, `READ_ME_FIRST.txt`. Scanner verified running
from the USB path itself (resolves penguins, reports MALE/FEMALE).

Prompt: 14.3k tokens including the injected dataset block. Up from 11.8k — the
new rules cost more than the diamonds trivia they replaced. Worth a trim pass
next round, but prefill is a once-per-session cost on a cached prefix and the
answers themselves got much shorter (multi-turn turns are 92-155 s and ~2 KB,
against 522 s and 6.8 KB for the whole paper in one message).

---

## Iteration 5 — the syllabus changed (Sept 2026): ML + DL, bank.csv

The course moved to Parts 3-4 (Machine Learning pipeline, Deep Learning with
MLPClassifier). Handouts: `G:\UIU G\8th new\DA LAB\adsfs\`. Another section's
CT-2 obtained 2026-09-15 and added as `evals/class_test_3_bank.md` with grader
`eval/grade_bank_ct2.py` and ground truth `eval/ref_bank_ct2.py`.

### The prompt actively forbade the new syllabus

Before writing a single recipe, grepping for the APIs the handouts teach found
the prompt saying:

  * *"sklearn is allowed ONLY for MinMaxScaler, StandardScaler, train_test_split"*
  * *"The labs stop at 'ready for a model'"*
  * *"Never import ... SimpleImputer / OneHotEncoder / ColumnTransformer"*

Parts 3-4 are built on exactly those. An omission makes the model improvise; a
BAN makes it refuse. This is rule 7 (a rule that drifts may be contradicted) at
syllabus scale, and it is now rule 15: **when new material lands, fix the
contradictions before adding anything.** The import section now splits the two
cases — pandas for hand-cleaning (Part 1), SimpleImputer/OneHotEncoder inside a
Pipeline (Parts 3-4) — which is the distinction the course actually draws.

### A fresh chat is not a fresh notebook

The student's workflow changed too: they now paste ONE QUESTION BLOCK (all of
a, b, c) into a BRAND-NEW TAB, then open another tab for the next block. Q2 and
Q3 therefore arrive with **zero history** — no dataset description, no starter
code, no memory of what Q1 built. Nothing in context can tell the model the
notebook exists; only the question's own wording can ("Q3", "your trained
model", "the matrix from Q2(a)").

The fix is a **canonical variable-name contract** published in the prompt
(`X_train_s`, `X_test_s`, `logreg`, `rf`, `mlp`, `cm`, `pipe_rf`/`pipe_mlp`).
Both the block that creates a name and the block that consumes it are written
by the same prompt, so they agree by construction. Supporting rules:
"re-derive the cheap (predictions), never the expensive (read_csv, split,
fit)", and a required `# continues the notebook — uses: ...` header that turns
a silent mismatch into something the student sees before running the cell.

`eval/grade_bank_ct2.py` reproduces this exactly: each block is a separate
`[system, user]` pair with the earlier answers deliberately absent, and the
three answers are then concatenated and executed as one notebook.

### Measured: a new tab is nearly free

From the llama-server log, same session:

    block 1   prompt eval  21465 tokens / 232.7 s   (92.2 tok/s)
    block 2   prompt eval     99 tokens /   1.9 s

The prefix cache holds across chats because the system prompt is byte-identical,
so only the new user message is processed. **The 21k-token prompt is paid once
per session, not once per tab** — which is what makes the three-tabs workflow
viable at all. Generation runs ~8.2 tok/s at this context depth, so blocks cost
~2 min each after the first.

That first 3.9 minutes (longer on the exam i7) was the student's worst
experience last time. It is now paid in the background: `ui/index.html` fires a
1-token warm-up request on page load, after restoring the remembered dataset,
so the prefill happens while the paper is being read. The README's old advice
("type hi first") depended on remembering to do it under exam pressure.

### Two real bugs found by measurement, not by reading

1. **The scanner emitted the wrong loader.** It sniffed `;` correctly for its
   own parsing and then printed `df = pd.read_csv('bank.csv')` anyway. Without
   `sep=';'` pandas returns ONE column named `age;"job";"marital";...` and
   every task dies. Injection has to cover *how to load* the file, not just
   what is in it.
2. **`llama-server` would not start** because Ollama owns port 11434 by
   default, and `start.py` sent the server's stderr to DEVNULL — so the only
   symptom was a bare "Timeout: AI server did not start." The log is now kept
   in `llama-server.log` and its last lines are printed on a timeout.

### Grading

Gold hand-written answer: 53/53. Deliberately broken answer: 19 checks fire.
Both directions matter — a check that never fires reads exactly like a pass.

First model run scored 48/52, but **three of the four failures were the
grader's**: an unused `import train_test_split` counted as a re-split;
`\bpipe\w*` never matches `mlp_pipe` (no word boundary after `_`); and the
probe demanded a variable literally named `pipe` when the block legitimately
built two. A fourth check gave a false PASS — `[0][1]` matched
`predict_proba(...)[0][1]`. That is four grader defects in one round, on top of
three last round. **A grader earns trust the same way a prompt does: by being
tested against a known-good and a known-bad answer before its numbers are
quoted.**


### The patch that made it worse, and what it proved

Fixing the starter-code repetition worked. The same round's score then fell
from 51/53 to **36/53**, because block 3 reloaded the CSV, re-split on all 16
columns and built a ColumnTransformer nobody asked for.

The cause was a rule I had written two hours earlier:

    START A NEW NOTEBOOK - only when the message actually sets the data up:
    it quotes a dataset description, gives starter code, names a CSV ...

Q3 of the real paper opens with a **"Context:" paragraph** about the dataset
(*"the positive class constitutes only ~11.5% of records ... dialing
non-converting leads wastes limited operational budget"*). That is background
for the question. My own discriminator read it as "quotes a dataset
description" and told the model to start over.

**The typo bank settled it without a single extra model call.** T3 is the same
question typed in a hurry, and the student does not retype the Context
paragraph — so T3 lacks exactly the text that triggered the rebuild. Same
prompt, same session: CLEAN 36/53, **TYPO 51/53**. A one-sample regression
would have been arguable; two banks differing in precisely the suspect
sentence made it a measurement.

The discriminator now says the QUESTION NUMBER decides and overrides
everything: anything above 1 is a continuation however much prose it carries.
A background paragraph is explicitly called out as not-setup, quoting the real
Context text. Two supporting patches went in with it: "the import list is a
MENU, not a template" (the model had transcribed all 16 import lines verbatim,
including LinearRegression and r2_score for a confusion-matrix answer), and a
gate on the ColumnTransformer recipe, which must not be reached for when X is
already built from a chosen numeric list.

### Grader defects this round: six

Three static (an unused `import train_test_split` read as a re-split;
a word-boundary regex that never matched `mlp_pipe`, since an underscore is itself a word character; a probe demanding a variable literally
named `pipe`), one false PASS (`[0][1]` matching `predict_proba(...)[0][1]`),
and two more found only because a 51/53 run had two failures worth reading:
`fit_transform(X_train[num_cols])` rejected for the subscript, and - the
stubborn one - prose reading *"reduces False Positives (FP) and increases False
Negatives (FN)"* flagged as wrong, because "increases" sits nine characters
from "False Positives".

That last check has now been rewritten so the increase verb must attach
DIRECTLY to false positives, and it ships with `_selftest_fp()`: three correct
sentences that must pass and five wrong ones that must fail. **A check subtle
enough to get wrong twice is subtle enough to need its own unit test.**

### The Notes fence, and a grader that turned one slip into thirteen

With the Context fix in, the rebuild stopped — and the score read 40/53, because
a SyntaxError killed the script and every probe check cascaded off it. The
cause was one line:

    ```
    Notes: The scaler was fitted only on X_train to prevent data leakage.
    ```

A Notes sentence wrapped in a bare fence. The grader treated every fenced
section as Python, concatenated that sentence into the notebook, and the whole
execution half of the report collapsed. Re-graded with an executor that skips
fences which do not parse as Python, the same two runs score **53/55** —
so the real defect was worth two checks, not fifteen.

Both corrections stand:

  * The GRADER now executes only fences that parse, and reports the slip
    through two checks of its own ("one fenced code block per answer",
    "nothing but Python inside a fence"). A formatting slip should cost the
    marks it is worth and no more, or the report stops describing the answer.
  * The PROMPT now shows the exact shape of a reply and tells it to count the
    backticks: two ``` lines in the whole response, Notes bare on the line
    after the closing fence. The rule against fenced Notes already existed in
    two places and nothing contradicted it — but the Parts 3-4 recipes added
    several "say in Notes that ..." instructions, so Notes are emitted far more
    often now and a rung-1 rule was no longer enough.

It appeared in BOTH modes in the same session, which is what makes it
systematic rather than noise under rule 4.

### Two more self-inflicted defects, both from the same reflex

Patching the fenced-Notes problem, I wrote a literal answer TEMPLATE into
RESPONSE FORMAT — an opening fence, placeholder code, a closing fence. The next
run ended every answer with three closing fences in a row and scored 39/55.
The prompt now held two complete literal fence pairs instead of one, and the
model mirrored the extra pair. The shape is now described in five numbered
steps with no new literal fences in the prompt at all.

The same run produced `NameError: name 'pipe' is not defined`. The cause was
the CANONICAL NAMES list I had introduced a few hours earlier:

      logreg, dtree, rf, knn, mlp     the fitted models
      pipe                            a Pipeline object

Listing `pipe` beside `mlp` and `rf` said, in effect, "this one also already
exists". It does not — nothing before Q3 builds a Pipeline — so the model
called `pipe.predict(new_client)` without ever writing
`pipe = Pipeline([...])`. The contract is now split in two, and the split is
the point:

  * BUILT BY AN EARLIER BLOCK — assume these exist (df, X, y, the split,
    scaler, X_train_s/X_test_s, logreg/rf/mlp, cm)
  * BUILT BY THE BLOCK IN FRONT OF YOU — create before use
    (pipe_rf, pipe_mlp, importance, cm_custom, the y_pred_* lines)

A contract that lists names without saying WHO CREATES THEM is not a contract;
it is a list of things the model may assume into existence.

### Running score, clean mode, one sample each

    grader v1, first prompt                    51/53   (after fixing 4 grader defects)
    + starter-code and ravel escalations       36/53   Context paragraph -> full rebuild
    + Context / question-number fix            53/55   (regraded; rebuild gone)
    + literal answer template                  39/55   stray fences + `pipe` NameError
    + template rewritten, contract split       see below

Typo mode over the same period: 53/53, then 53/55 twice — it was never the
problem, because the student retyping a question does not reproduce the
Context paragraph or invite the formatting drift.

The honest lesson is rule 4 turned on myself: **three of my five patches this
round made the score worse before they made it better, and every one was
caught only by re-running.** A prompt edit is a change to a program whose
behaviour you cannot predict by reading it.

### Closed: 56/56 both modes

    CLEAN (paper as printed)   56/56
    TYPO  (as actually typed)  56/56

Both saved runs re-grade to 56/56 offline. The three blocks now emit exactly
one fenced code block each, no block rebuilds what an earlier one built, and
every pipeline is fitted on raw named features. Timings from the final run:

    block 1  398 s   (21k-token prefill, paid once per session)
    block 2  128 s
    block 3  252 s

The grader ships with a gold answer (56/56) and a deliberately broken one
(12 checks fire, naming the double-scaled pipeline by its missing feature
names and printing the wrong confusion matrix it produced), plus
`_selftest_fp()` for the one check subtle enough to have been wrong twice.

### Deployed

NOT YET — the pendrive was not connected during this round. Files to sync when
it is: `start.py`, `dataset_scan.py`, `ui/index.html`,
`courses/da-python/prompt.md`, `course.json`, `models.json`, `data/bank.csv`,
`DEPLOY_README.txt`.

### Prompt size

19.4k tokens, up from 14.3k. Parts 1-2 (preprocessing + EDA) are 33% of it and
the CT-2 paper touches none of them; they were KEPT deliberately, because a
final exam can still cover them and the cost is a one-time prefill that now
runs in the background on page load. Revisit only if a future paper makes
Parts 1-2 genuinely dead.

---

## Iteration 6 — does it generalize, or is it tuned to one paper?

Asked directly: "can you confirm it can solve similar questions, not just the
other section's paper? the teacher will obviously change the questions."

The honest answer was no, and 56/56 could not settle it: the clean and typo
banks are the SAME paper typed two ways. So a fourth bank went in -
`class_test_4_titanic.md`, written from the Part 3/4 handouts rather than from
any sitting, and deliberately different on every axis a model could memorise:

    dataset        bank.csv, 7 numeric cols  ->  titanic, text + real NaNs
    preprocessing  none (starter did it)     ->  impute/map/one-hot in Q1(a)
    models         LogisticRegression, RF    ->  DecisionTree, KNN
    architecture   (32,16) max_iter=500      ->  (16,8) max_iter=1000
    attribute      .loss_                    ->  .n_layers_ AND .loss_
    folds          5                         ->  10
    palette        Blues                     ->  Purples
    ColumnTransformer  must NOT be used      ->  MUST be used
    threshold      T=0.65, FP FALLS          ->  T=0.35, FP RISES

The last two are the real tests. The inverted threshold catches a model
reciting the bank paper's conclusion instead of reasoning; the required
ColumnTransformer checks that the gate added this round suppressed a reflex
rather than a capability. The key is cross-checked against the figures the
handouts themselves print (MLP 0.7709, loss 0.3058, cm [[91,19],[22,47]], the
four architecture accuracies, pipeline 0.7933) - an independent confirmation
that the reference implementation matches the teacher's.

**First run: 36/52.** The suspicion was correct.

Block 2 ignored the continuation contract and emitted the Parts 1-2
PREPROCESS -> SPLIT -> EDA SKELETON: reloaded the CSV, re-imputed, capped fare
outliers, re-encoded, re-split. It pulled in `deck`, `embark_town` and `who` -
columns the paper never lists - and the notebook died on
`could not convert string to float: 'man'`. Block 3 then reassigned `df`,
`X_train`, `X_train_s`, `X_test_s` and `mlp`, orphaning everything.

Why here and not on the bank paper: bank's Q2 is purely evaluative ("generate
the confusion matrix for your trained MLPClassifier"), while this Q2(a) says
"**Initialize and train** an MLPClassifier". A training verb reads as setup, the
model went looking for the most template-shaped thing in the prompt, and the
Parts 1-2 skeleton is exactly that.

Three fixes, all of them gaps the bank paper structurally could not expose:

  1. The skeleton now carries a GATE ABOVE IT, not a caveat below it: never in
     a continuation block, with the `'man'` failure named.
  2. "TRAIN A MODEL IS NOT A LICENCE TO REBUILD THE DATA" joins the continuation
     tells. A new ESTIMATOR in Q2/Q3 is expected; new DATA is not.
  3. The skeleton's own guard pointed at "see FOLLOW-UP QUESTIONS above" - a
     section renamed earlier this round. A dangling cross-reference in a prompt
     is a rule that silently stops being findable. Swept for others: none left.

Also surfaced: the paper's Q3(c) legitimately NEEDS a reload, because it asks
for a pipeline over the RAW uncleaned frame and Q1 cleaned `df` in place. The
prompt said never reload, full stop. It now carries THE ONE LEGITIMATE RELOAD,
whose whole content is the guard: load it under NEW names (`raw`, `Xr`,
`Xr_train`) so the cleaned frame, the split and the fitted models all survive.
Two of the three failures in my own hand-written GOLD answer were this rule
being too absolute - the gold answer was right and the grader was wrong.

**The lesson is rule 10, confirmed a second time.** One paper is not the course,
and two typings of one paper are not two papers. A bank that shares a defect
with the system under test cannot detect that defect.

### My own harness was half the problem

Three prompt patches in a row failed to move the unseen paper (36 -> 34 -> 33).
The cause was not the prompt. `load_bank()` stripped the `## Q2` heading, so the
model received a block beginning "(a) Initialize and train an MLPClassifier..."
with NO question number and NO back-reference to earlier work. The prompt rule
I had just written says THE QUESTION NUMBER DECIDES IT - and the harness was
deleting the question number before sending.

The bank paper never exposed this because its Q2 and Q3 carry explicit
back-references ("your trained MLPClassifier", "your default matrix from
Q2(a)") that survive label-stripping. The titanic Q2 has none, so once the
label was gone there was genuinely nothing left to key on, and rebuilding was
the only reasonable reading.

Both harnesses now send the question as PRINTED, label included, because that
is what the student pastes. Block 2's answer immediately fell from 4468 chars
to 1857 and every continuation failure disappeared.

**A test harness is part of the system under test.** Three rounds of prompt
patching were spent on a defect that lived in the grader.

### The bug that only appears on a machine with no internet

With the continuation failures gone, one error was killing the rest:
`could not convert string to float: 'Third'`. The offline-loading recipe was
being written as

      try:    df = sns.load_dataset('titanic')[cols].copy()   # 8 columns
      except: df = pd.read_csv('titanic.csv')                 # 15 columns

The subset is applied on the `try` line only. On the dev box the `try` branch
succeeds and everything looks correct. ON THE EXAM PC THERE IS NO INTERNET, so
`sns.load_dataset` always raises and the fallback is the ONLY branch that ever
runs - silently handing back the full fifteen-column seaborn dump: `class`
('Third'), `who` ('man'), `deck`, and `alive`, which is the target spelled out
in words. A model trained on `alive` reads the answer off its own input.

The recipe now puts every post-load step AFTER the try/except so it applies on
both paths. bank.csv could not expose this: it is read with a direct
`pd.read_csv`, no fallback.

### Two more, both about clobbering canonical names

  * Q1 scaled with the PART 1 in-place form
    (`X_train[num_cols] = scaler.fit_transform(...)`) - correct, self-consistent
    code that creates no `X_train_s`. Q2, written in a fresh chat that cannot
    see it, said `mlp.fit(X_train_s, ...)` and died on NameError. Two recipes
    showed two forms and the question's wording matched the wrong one (rule 7).
    The Parts 3-4 recipe now states that a modelling paper ends that cell with
    X_train_s and X_test_s bound, whatever the question calls the step.
  * The architecture sweep used `mlp` as its LOOP VARIABLE, so after the loop
    the canonical `mlp` was (32,16,8) instead of the (16,8) network the paper
    specified - and every later matrix, threshold and probability silently
    referred to the wrong model. Unified with the reload case into one rule:
    never rebind a name from the "built by an earlier block" list.

Running score on the unseen paper: 36, 34, 33, 38, 37, 40. The notebook now
executes end to end; the remaining failures are narrow and specific rather than
cascading.

### Where it landed, and what the number actually means

Eight runs of the unseen paper: 36, 34, 33, 38, 37, 40, 38, 38 out of 52. It
plateaued, with a DIFFERENT single blocking error each time (a stray column, a
missing OneHotEncoder import, an unbound X_train_s, a ColumnTransformer handed
the already-encoded frame). That oscillation is the signature of a reliability
ceiling, not of one more fixable defect - so the patching stopped there rather
than trading one failure for another indefinitely.

Tallying every check across the last five runs separates knowledge from
reliability, and the split is stark.

**34 checks held in EVERY run** - all of the ML/DL substance:

    correct estimators and hyper-parameters (DecisionTree, KNN, MLP (16,8)
    max_iter=1000) · 10-fold NOT 5 · cmap='Purples' NOT Blues · n_layers_ AND
    .loss_ · ColumnTransformer used where the paper asks for it ·
    handle_unknown='ignore' · both SimpleImputer strategies · median/mode
    imputation · get_dummies(dtype=int) · stratified split · scaler fit on
    train only · threshold applied to predict_proba[:,1] · FP counts extracted
    · Q2 never rebuilds · no fabricated output · one fenced block per answer

**And the sharpest check of all passed 5/5: "prose gets the INVERTED FP
direction right".** This paper lowers the threshold to 0.35, so false positives
must RISE - the exact opposite of the bank paper's 0.65. A model reciting the
tuned-on answer fails this every time. It never did. The same goes for 10-fold
(it never carried 5 over) and Purples (never carried Blues over). **The ML/DL
knowledge is genuinely general, not memorised.**

**9 checks failed in every run** - and all nine are PROBE checks, which can only
run if the notebook executes end to end. Execution succeeded in 1 run of 5. So
those nine are one failure counted nine times, not nine independent defects.

The honest summary:

    the syllabus knowledge          generalizes - verified against an unseen
                                    paper with an inverted answer
    multi-block state continuity    holds on the real CT-2 shape (56/56 twice),
                                    fragile on a longer preprocessing-heavy
                                    paper: roughly one fatal slip per attempt

That fragility is real, because a notebook that does not run scores zero. But
it is also the cheapest failure in the paper to recover from: every instance was
a missing import or an unbound name, which announces itself as a NameError the
moment the student runs the cell. DEPLOY_README now teaches that explicitly -
run each cell as you paste it, and read the name the error complains about.

**Caveat on this bank, stated plainly:** the titanic paper is HARDER than the
real CT-2. The bank paper's starter code does all the preprocessing; here Q1(a)
does it. Q3(c) is a six-mark ColumnTransformer spec. Block 3 answers run ~5000
chars against the bank paper's ~3400. It was built to be adversarial, and 38/52
on a deliberately harder unseen paper alongside 56/56 on the real format is the
result - not 38/52 on the exam.
