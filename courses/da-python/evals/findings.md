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
