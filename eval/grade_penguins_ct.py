r"""Objectively grade an answer to the REAL penguins Class Test paper.

    python eval\grade_penguins_ct.py            # both modes
    python eval\grade_penguins_ct.py oneshot    # whole paper in one message
    python eval\grade_penguins_ct.py multiturn  # task by task, typos and all

Assumes llama-server is already listening on :11434 (start it with
launch-da.bat, or python start.py da-python).

Why this exists, and why it EXECUTES rather than reads
------------------------------------------------------
In the real sitting the model wrote

    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

against a file containing MALE/FEMALE. That code RUNS CLEAN, READS CORRECT, and
silently turns every value in the column into NaN. No amount of reading the
answer catches it. So after running the generated code we inject a PROBE that
inspects the live variables - df, X_train, y_train - and report on what the
data actually became, not on what the model claimed.

The multi-turn mode concatenates each turn's code block in order, which is
exactly what the student does when pasting one answer per notebook cell. That
is how the missing-import NameError shows up.

Needs pandas/seaborn/sklearn on the grading box - dev machine only.
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dataset_scan

WD = os.environ.get("LABRAT_GRADE_DIR") or os.path.join(ROOT, "eval", "_gradework")
os.makedirs(WD, exist_ok=True)
BANK = os.path.join(ROOT, "courses", "da-python", "evals", "class_test_2_penguins.md")
AI = "http://127.0.0.1:11434/v1/chat/completions"

# Ground truth, computed from data/penguins.csv - never from the model.
KEY = {
    "rows": 344, "cols": 7,
    "n_train": 275, "n_test": 69,
    "sex_counts": {0: 179, 1: 165},
    "dummies": ["island_Biscoe", "island_Dream", "island_Torgersen"],
    "top_flipper": "Gentoo",
    "top_pair": {"body_mass_g", "flipper_length_mm"},
}

PROBE = r'''
# ---- grader probe (appended, not model-written) ----
import json as _json
_o = {}
def _t(k, fn):
    try: _o[k] = fn()
    except Exception as _e: _o[k] = "ERR:" + type(_e).__name__
_t("df_shape",     lambda: list(df.shape))
_t("df_cols",      lambda: [str(c) for c in df.columns])
_t("total_missing",lambda: int(df.isnull().sum().sum()))
_t("sex_nan",      lambda: int(df['sex'].isna().sum()))
_t("sex_vals",     lambda: sorted(str(v) for v in df['sex'].dropna().unique())[:6])
_t("n_train",      lambda: int(len(X_train)))
_t("n_test",       lambda: int(len(X_test)))
_t("X_cols",       lambda: [str(c) for c in X_train.columns])
_t("y_train_vals", lambda: sorted(str(v) for v in set(y_train))[:6])
_t("culmen_ok",    lambda: bool('culmen_ratio' in df.columns))
open("probe.json", "w").write(_json.dumps(_o))
'''


def load_bank():
    txt = open(BANK, encoding="utf-8").read()
    one = re.search(r"^## Q1\n(.*?)^---", txt, re.S | re.M).group(1).strip()
    turns = re.findall(r"^## (T\d)\n(.*?)(?=^## T\d|\n---|\Z)", txt, re.S | re.M)
    return one, [t[1].strip() for t in turns]


def sys_prompt():
    p = open(os.path.join(ROOT, "courses", "da-python", "prompt.md"), encoding="utf-8").read().strip()
    scan = dataset_scan.scan(dataset_scan.resolve("penguins", ROOT))
    # Same order the UI uses: stable prompt first, dataset block appended last.
    return p + "\n\n" + scan["text"] if scan["ok"] else p


def ask(messages, max_tokens=8192, temp=0.3):
    body = json.dumps({"model": "local", "messages": messages, "stream": True,
                       "temperature": temp, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(AI, data=body, headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            try:
                delta = json.loads(d)["choices"][0].get("delta", {})
            except Exception:
                continue
            if delta.get("content"):
                out.append(delta["content"])
    return "".join(out)


def blocks_of(text):
    return re.findall(r"```(?:python)?\n(.*?)```", text, re.S)


def run_code(code, tag):
    src = os.path.join(WD, f"penguins_{tag}.py")
    open(src, "w", encoding="utf-8").write(code + "\n" + PROBE)
    probe = os.path.join(WD, "probe.json")
    if os.path.exists(probe):
        os.remove(probe)
    csv = os.path.join(ROOT, "data", "penguins.csv")
    dst = os.path.join(WD, "penguins.csv")
    if not os.path.exists(dst):
        import shutil; shutil.copy(csv, dst)
    r = subprocess.run([sys.executable, src], capture_output=True, text=True, cwd=WD,
                       env=dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="always"),
                       timeout=600)
    data = {}
    if os.path.exists(probe):
        try: data = json.load(open(probe))
        except Exception: pass
    return r, data


def grade(answers, tag):
    """answers: list of assistant messages, in order."""
    checks = []
    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:120]))

    per_block = [blocks_of(a) for a in answers]
    flat = [b for bs in per_block for b in bs]
    if not flat:
        chk("produced a python code block", False, "none found")
        return checks, {}, ""
    code = "\n\n".join(flat)

    # ---- prose checks (OUTSIDE the code fences) --------------------------
    # Executing the code is not enough. The model can emit correct code and
    # then follow it with an INVENTED output transcript - which is what the
    # student actually reads. Seen for real: correct code printing Gentoo,
    # followed by a fake "Largest: Chinstrap" with the species counts swapped.
    prose = "\n".join(re.sub(r"```.*?```", "", a, flags=re.S) for a in answers)
    tells = [t for t in ("dtype:", "Name: count", "Output:", "0 rows x", "[5 rows x")
             if t in prose]
    chk("no fabricated output transcript", not tells,
        "prose contains pasted-output markers: " + ", ".join(tells))
    # Any winner named in prose must be the true one.
    bad_claim = re.search(r"largest[^\n]{0,40}(chinstrap|adelie)", prose, re.I)
    chk("prose names no wrong winner", not bad_claim,
        bad_claim.group(0)[:90] if bad_claim else "")

    # ---------- static ----------
    chk("reads penguins.csv", "penguins.csv" in code)
    chk("does NOT substitute sns.load_dataset", "load_dataset" not in code)

    # Every block that USES these must also import them (the NameError bug).
    need = {"train_test_split": "from sklearn.model_selection import train_test_split",
            "MinMaxScaler": "from sklearn.preprocessing import"}
    missing = []
    for i, bs in enumerate(per_block):
        for b in bs:
            for sym, imp in need.items():
                if re.search(rf"\b{sym}\s*\(", b) and imp not in b:
                    missing.append(f"turn{i+1}:{sym}")
    chk("every block imports what it uses", not missing, ", ".join(missing))

    # The killer: case-robust category mapping.
    has_map = ".map(" in code
    chk("sex map is case-robust", (not has_map) or (".str.upper()" in code or ".str.lower()" in code),
        "bare .map() on unverified case" if has_map else "no map")
    chk("prints unmapped count after map", (not has_map) or
        re.search(r"unmapped|isna\(\)\.sum\(\)|isnull\(\)\.sum\(\)", code) is not None)

    # All four numeric columns present in whatever list drives imputation.
    four = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    chk("all 4 numeric columns named", all(c in code for c in four),
        "missing " + ", ".join(c for c in four if c not in code))

    chk("get_dummies uses dtype=int", re.search(r"dtype\s*=\s*int", code) is not None)
    # Only QUOTED occurrences can be a hand-typed column name. The recommended
    # derive pattern makes bare identifiers (island_cols, island_dummy_cols)
    # which are variables, not column references - never flag those.
    quoted = re.findall(r"['\"]island_(\w+)['\"]", code)
    unknown = sorted({d for d in quoted if f"island_{d}" not in KEY["dummies"]})
    chk("no invented one-hot column names", not unknown, "invented: " + ", ".join(unknown))

    # Accept either the literal or a target variable bound to 'species' -
    # `target = 'species'; X = df.drop(columns=[target])` is equally correct.
    # Whether the target actually stayed out of X is settled by the probe below.
    drop_ok = (re.search(r"drop\(\s*columns\s*=\s*\[?\s*['\"]species", code) is not None
               or (re.search(r"drop\(\s*columns\s*=\s*\[?\s*target\b", code) is not None
                   and re.search(r"target\s*=\s*['\"]species['\"]", code) is not None))
    chk("X built with drop(columns=[target])", drop_ok)
    chk("no hand-built feature_cols list", "feature_cols" not in code)
    chk("scaler fit on train only",
        "fit_transform(X_train" in code and re.search(r"\.transform\(\s*X_test", code) is not None)
    chk("random_state=42 present", re.search(r"random_state\s*=\s*42", code) is not None)

    pal = re.findall(r"sns\.(?:countplot|boxplot|barplot)\((.*?)\)", code, re.S)
    badpal = [p for p in pal if "palette=" in p and "hue=" not in p]
    chk("no palette= without hue=", not badpal, f"{len(badpal)} call(s)")

    # The paper NAMES these functions; substituting loses the mark.
    chk("uses plt.hist for 4a", "plt.hist(" in code)
    chk("uses sns.countplot for 4b", "sns.countplot(" in code)
    chk("uses sns.boxplot for 4c", "sns.boxplot(" in code)
    chk("uses kind='bar' for 5a", re.search(r"kind\s*=\s*['\"]bar['\"]", code) is not None)
    chk("uses sns.barplot for 5b", "sns.barplot(" in code)
    chk("uses sns.heatmap cmap=coolwarm for 5c",
        "sns.heatmap(" in code and "coolwarm" in code)
    chk("dedups the correlation pair", "triu" in code or "tril" in code
        or re.search(r"get_level_values\(0\)\s*<\s*", code) is not None)

    # ---------- execution ----------
    r, probe = run_code(code, tag)
    out = r.stdout
    chk("code executes without error", r.returncode == 0,
        (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else "")

    warns = [l for l in r.stderr.splitlines()
             if ("FutureWarning" in l or "DeprecationWarning" in l or "Pandas4Warning" in l)
             and "site-packages" not in l]
    chk("no deprecation warnings from OUR code", not warns, warns[0] if warns else "")

    # ---------- probe: what the DATA actually became ----------
    chk("df loaded 344 rows", probe.get("df_shape", [0])[0] == KEY["rows"], probe.get("df_shape"))
    chk("zero missing after imputation", probe.get("total_missing") == 0,
        f"total_missing={probe.get('total_missing')}")
    # THE headline check - a case-mismatched map leaves 344 NaNs here.
    chk("sex column SURVIVED encoding (not all NaN)",
        probe.get("sex_nan") == 0, f"sex_nan={probe.get('sex_nan')} sex_vals={probe.get('sex_vals')}")
    chk("culmen_ratio created", probe.get("culmen_ok") is True)
    chk("train/test = 275/69", probe.get("n_train") == KEY["n_train"]
        and probe.get("n_test") == KEY["n_test"],
        f"{probe.get('n_train')}/{probe.get('n_test')}")
    xc = probe.get("X_cols") or []
    chk("species NOT in X", "species" not in xc, f"{len(xc)} cols")

    # ---------- printed conclusions ----------
    chk("names Gentoo as largest flipper", re.search(r"Gentoo", out) is not None)
    low = out.lower()
    chk("states classes are imbalanced", "imbalanc" in low)
    # Look for a line that names BOTH features. Matching only the first
    # correlat-ish line picks up the "--- Task 5c ---" banner instead of the
    # answer; a tail slice lets a crashed run pass off the matrix dump.
    cand = [l.strip() for l in low.splitlines()
            if "correlat" in l or "pair" in l or "highest" in l]
    hit = next((l for l in cand
                if "flipper_length_mm" in l and "body_mass_g" in l), "")
    chk("top pair = flipper_length_mm & body_mass_g", bool(hit),
        hit[:110] or f"{len(cand)} correlation line(s), none naming both features")
    return checks, probe, out + "\n--- STDERR (tail) ---\n" + r.stderr[-1500:]


def report(title, checks):
    w = max(len(c[0]) for c in checks)
    n = sum(1 for c in checks if c[1])
    print(f"\n{'='*(w+34)}\n {title}\n{'='*(w+34)}")
    for name, ok, det in checks:
        # Detail is diagnostic for a failure; on a PASS it just reads as noise.
        print(f"  {name.ljust(w)}  {'PASS' if ok else 'FAIL'}  {'' if ok else det}")
    print(f"  {'-'*(w+30)}")
    print(f"  {str(n).rjust(len(str(len(checks))))}/{len(checks)} checks passed")
    return n, len(checks)


def regrade(path):
    r"""Re-score a saved run log without touching the model.

        python eval\grade_penguins_ct.py regrade <runs\...-multiturn.md>

    Multi-turn logs store one answer per '---' separated section, in order.
    """
    txt = open(path, encoding="utf-8").read()
    answers = [s for s in re.split(r"\n\n---\n\n", txt) if s.strip()]
    tag = "regrade-" + os.path.basename(path).replace(".md", "")
    c, probe, out = grade(answers, tag[:40])
    report(os.path.basename(path), c)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "regrade":
        regrade(sys.argv[2])
        return
    one, turns = load_bank()
    sp = sys_prompt()
    print(f"system prompt: {len(sp)} chars (~{len(sp)//4} tokens, incl. dataset block)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = os.path.join(ROOT, "courses", "da-python", "evals", "runs")
    os.makedirs(runs, exist_ok=True)
    totals = []

    if mode in ("both", "oneshot"):
        t0 = time.time()
        print("\n[oneshot] whole paper in one message ...", flush=True)
        a = ask([{"role": "system", "content": sp}, {"role": "user", "content": one}])
        print(f"[oneshot] {int(time.time()-t0)}s, {len(a)} chars")
        open(os.path.join(runs, f"{stamp}-penguins-oneshot.md"), "w", encoding="utf-8").write(a)
        c, probe, out = grade([a], "oneshot")
        totals.append(("ONE-SHOT (whole paper)",) + report("ONE-SHOT (whole paper)", c))
        open(os.path.join(WD, "oneshot_stdout.txt"), "w", encoding="utf-8").write(out)

    if mode in ("both", "multiturn"):
        print("\n[multiturn] task by task, with the student's real typos ...", flush=True)
        hist = [{"role": "system", "content": sp}]
        answers = []
        for i, t in enumerate(turns, 1):
            t0 = time.time()
            hist.append({"role": "user", "content": t})
            a = ask(hist)
            hist.append({"role": "assistant", "content": a})
            answers.append(a)
            print(f"  T{i}: {int(time.time()-t0)}s, {len(a)} chars, "
                  f"{len(blocks_of(a))} block(s)", flush=True)
        open(os.path.join(runs, f"{stamp}-penguins-multiturn.md"), "w", encoding="utf-8").write(
            "\n\n---\n\n".join(answers))
        c, probe, out = grade(answers, "multiturn")
        totals.append(("MULTI-TURN (task by task)",) + report("MULTI-TURN (task by task)", c))
        open(os.path.join(WD, "multiturn_stdout.txt"), "w", encoding="utf-8").write(out)

    print("\n" + "=" * 50)
    for name, n, tot in totals:
        print(f"  {name.ljust(28)} {n}/{tot}")
    print("=" * 50)


if __name__ == "__main__":
    main()
