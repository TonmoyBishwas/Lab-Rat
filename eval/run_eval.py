r"""
Lab Rat AI - Evaluation runner.

Runs a markdown question bank against the local model with a course's system
prompt and writes a timestamped answers log. Stdlib only, fully offline.

Usage (from the project root):
    python eval\run_eval.py dataviz-python
    python eval\run_eval.py dataviz-python --bank evaluations\stress_test_questions.md
    python eval\run_eval.py dataviz-python --only Q3,Q7        (targeted re-eval)
    python eval\run_eval.py dataviz-r --model Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf

Bank format: one question per "## Q<n> ..." heading. Everything under the
heading up to the next "## " (or EOF) is the question text, except:
  - "---" horizontal rules are dropped
  - paragraphs that are entirely italic ("*...*") are dropped - those are
    evaluator notes about what the question probes, and sending them to the
    model would leak hints and invalidate the eval.

If llama-server is not already running on port 11434, the runner boots it
itself (same flags as the launchers, via start.py) and shuts it down when done.

Output: courses\<id>\evals\runs\<timestamp>_<model>_<bank>.md
Grade the output with a Claude Code instance - see eval\BLINDSPOT_WORKFLOW.md.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import start  # noqa: E402  (find_model, start_llama_server, wait_healthy, CTX_FOR, AI_PORT)


def parse_bank(path):
    """Return ordered list of (qid, question_text)."""
    text = open(path, encoding="utf-8").read()
    questions = []
    for m in re.finditer(r"^## (Q\d+)[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        qid, body = m.group(1), m.group(2)
        kept = []
        for para in re.split(r"\n\s*\n", body):
            p = para.strip()
            if not p or p == "---":
                continue
            # italic-only evaluator note
            if p.startswith("*") and p.endswith("*") and not p.startswith("**"):
                continue
            kept.append(p)
        questions.append((qid, "\n\n".join(kept).strip()))
    return questions


def server_running():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{start.AI_PORT}/health", timeout=2)
        return True
    except Exception:
        return False


def ask(sys_prompt, question, temperature, max_tokens, timeout=900):
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{start.AI_PORT}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    choice = data["choices"][0]["message"]
    answer = choice.get("content") or ""
    usage = data.get("usage", {})
    return answer, dt, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("course", help="course id under courses\\")
    ap.add_argument("--bank", default=None,
                    help="question bank .md (default: courses\\<id>\\evals\\questions.md)")
    ap.add_argument("--only", default=None, help="comma-separated question ids, e.g. Q3,Q7")
    ap.add_argument("--model", default=None, help="exact gguf filename to require")
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    cdir = os.path.join(ROOT, "courses", args.course)
    course = json.load(open(os.path.join(cdir, "course.json"), encoding="utf-8"))
    sys_prompt = open(os.path.join(cdir, "prompt.md"), encoding="utf-8").read().strip()

    bank_path = args.bank or os.path.join(cdir, "evals", "questions.md")
    if not os.path.exists(bank_path):
        sys.exit(f"Question bank not found: {bank_path}")
    questions = parse_bank(bank_path)
    if args.only:
        wanted = {q.strip().upper() for q in args.only.split(",")}
        questions = [(q, t) for q, t in questions if q.upper() in wanted]
    if not questions:
        sys.exit("No questions matched.")

    temperature = course.get("temperature", 0.3)
    max_tokens = args.max_tokens or course.get("max_tokens", 4096)

    # ── Model / server ──
    booted = None
    if server_running():
        model_name = "(already-running server)"
        if args.model:
            print("NOTE: --model ignored, a server is already running on the port.")
    else:
        if args.model:
            path = os.path.join(ROOT, "models", args.model)
            if not os.path.exists(path):
                sys.exit(f"Model not found: {path}")
            model_name, model_path = args.model, path
        else:
            model_name, model_path = start.find_model(course.get("models"))
            if not model_name:
                sys.exit("No model found in models\\. Run: python download-model.py")
        print(f"Booting llama-server with {model_name} ...")
        booted = start.start_llama_server(model_name, model_path)
        if start.wait_healthy() is None:
            booted.terminate()
            sys.exit("\nllama-server did not become healthy.")
        print("\rServer ready.                    ")

    # ── Run ──
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    bank_tag = os.path.splitext(os.path.basename(bank_path))[0]
    model_tag = re.sub(r"[^A-Za-z0-9.]+", "-", model_name).strip("-")
    out_dir = os.path.join(cdir, "evals", "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stamp}_{model_tag}_{bank_tag}.md")

    header = (f"# Eval run - {args.course}\n\n"
              f"- **Date:** {stamp}\n- **Model:** {model_name}\n"
              f"- **Bank:** {os.path.relpath(bank_path, ROOT)}\n"
              f"- **Temperature:** {temperature}  |  **Max tokens:** {max_tokens}\n"
              f"- **Prompt:** courses/{args.course}/prompt.md "
              f"({len(sys_prompt)} chars)\n\n"
              f"Grading: see eval/BLINDSPOT_WORKFLOW.md. Verdict per question: "
              f"PASS / FAIL / PARTIAL + failure-mode notes.\n")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)

    try:
        for i, (qid, qtext) in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {qid} ...", end="", flush=True)
            try:
                answer, dt, usage = ask(sys_prompt, qtext, temperature, max_tokens)
                ct = usage.get("completion_tokens", 0)
                stat = f"{dt:.0f}s, {ct} completion tokens, {ct/dt:.1f} tok/s" if dt else ""
                print(f" done ({stat})")
            except Exception as e:
                answer, stat = f"**RUNNER ERROR:** {e}", "error"
                print(f" ERROR: {e}")
            with open(out_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"\n---\n\n## {qid}\n\n### Question\n\n{qtext}\n\n"
                        f"### Answer  ({stat})\n\n{answer}\n\n"
                        f"### Verdict\n\n_TODO: PASS / FAIL / PARTIAL - graded by Claude Code_\n")
    finally:
        if booted:
            booted.terminate()
            print("Booted server stopped.")

    print(f"\nRun log: {out_path}")


if __name__ == "__main__":
    main()
