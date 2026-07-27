"""
Lab Rat AI - answer execution checker.

Extracts every ```python block from an eval run log and actually RUNS it, so a
grader never has to guess whether generated code would execute. Complements
eval/BLINDSPOT_WORKFLOW.md step 3: this tool answers "does it run?", a human
still answers "does it answer the question?".

Stdlib only (matching the constraint on eval/). The *generated* code obviously
needs pandas/seaborn/sklearn, but those run in a subprocess on the dev machine,
never on the exam appliance.

Usage:
    python eval\\check_answers.py courses\\da-python\\evals\\runs\\20260727-1830_gemma....md
    python eval\\check_answers.py <run-log> --only Q3,Q7
    python eval\\check_answers.py <run-log> --timeout 180 --keep

Exit code is the number of questions whose code failed to run (0 = all clean).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# "## Q3" ... up to the next "## " heading or EOF
Q_RE = re.compile(r"^## (Q\d+)[^\n]*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
# fenced python block; tolerate ```python / ```py / ```Python
CODE_RE = re.compile(r"```(?:python|py|Python)\s*\n(.*?)```", re.DOTALL)
# the answer section only - never grade code that appears in the echoed question
ANSWER_RE = re.compile(r"^### Answer[^\n]*\n(.*?)(?=^### |\Z)", re.MULTILINE | re.DOTALL)


def parse_log(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = []
    for qid, body in Q_RE.findall(text):
        m = ANSWER_RE.search(body)
        answer = m.group(1) if m else ""
        out.append((qid, CODE_RE.findall(answer)))
    return out


def run_block(code, workdir, timeout):
    """Run one code block in a subprocess. Return (ok, detail)."""
    # Force a non-interactive matplotlib backend so plt.show() never blocks.
    preamble = "import matplotlib\nmatplotlib.use('Agg')\n"
    src = os.path.join(workdir, "_snippet.py")
    with open(src, "w", encoding="utf-8") as f:
        f.write(preamble + code)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "ignore"
    try:
        p = subprocess.run(
            [sys.executable, src],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"

    if p.returncode == 0:
        return True, ""

    # Last traceback line is the useful part
    err = (p.stderr or "").strip().splitlines()
    detail = ""
    for line in reversed(err):
        if line.strip() and not line.startswith(" "):
            detail = line.strip()
            break
    return False, detail or f"exit {p.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to an eval run log .md")
    ap.add_argument("--only", default=None, help="comma-separated ids, e.g. Q3,Q7")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per block (default 300)")
    ap.add_argument("--keep", action="store_true", help="keep the temp workdir for debugging")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"No such log: {args.log}")
        return 1

    questions = parse_log(args.log)
    if args.only:
        want = {q.strip().upper() for q in args.only.split(",")}
        questions = [(q, c) for q, c in questions if q.upper() in want]

    if not questions:
        print("No questions parsed. Check the log format (## Q<n> headings).")
        return 1

    workdir = tempfile.mkdtemp(prefix="labrat_check_")
    # Vendored CSVs, if present, let the prompt's offline pd.read_csv fallback work.
    data_dir = os.path.join(ROOT, "data")
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            if fn.endswith(".csv"):
                shutil.copy(os.path.join(data_dir, fn), workdir)

    print(f"\nChecking {len(questions)} questions from {os.path.basename(args.log)}")
    print(f"Workdir: {workdir}\n")
    print(f"  {'Q':<5} {'BLOCKS':<7} {'RESULT':<9} DETAIL")
    print("  " + "-" * 76)

    failures = 0
    multi = []
    for qid, blocks in questions:
        if not blocks:
            print(f"  {qid:<5} {0:<7} {'NO CODE':<9} no python block found in the answer")
            failures += 1
            continue
        if len(blocks) > 1:
            multi.append(qid)

        # Every block must run; report the first that does not.
        ok, detail = True, ""
        for i, code in enumerate(blocks):
            ok, detail = run_block(code, workdir, args.timeout)
            if not ok:
                detail = (f"block {i+1}/{len(blocks)}: " if len(blocks) > 1 else "") + detail
                break

        status = "RUNS" if ok else "ERROR"
        if not ok:
            failures += 1
        print(f"  {qid:<5} {len(blocks):<7} {status:<9} {detail[:60]}")

    print("  " + "-" * 76)
    print(f"  {len(questions) - failures}/{len(questions)} executed cleanly.")
    if multi:
        print(f"\n  WARNING: multiple code blocks in {', '.join(multi)} — the prompt")
        print("  requires exactly ONE Assumptions/code/Notes triple. Treat as a format FAIL.")
    print("\n  Reminder: 'RUNS' only means it executed. Whether it ANSWERS the")
    print("  question is still a human/Claude judgement — see eval/BLINDSPOT_WORKFLOW.md.\n")

    if args.keep:
        print(f"  Workdir kept: {workdir}\n")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    return failures


if __name__ == "__main__":
    sys.exit(main())
