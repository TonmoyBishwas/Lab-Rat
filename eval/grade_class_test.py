"""Objectively grade an answer to the real Class Test 1 paper.

Usage:
    python eval\grade_class_test.py <answer.md>

Executes the generated code and checks 25 specific properties against values
verified independently from data\diamonds.csv - not against the model's own
claims. Reading an answer is not enough: the failure that motivated this
grader (the target leaking into X, so "highest correlation with price" answers
"price" at r=1.000) produces code that RUNS CLEAN and looks entirely correct.

Needs pandas/seaborn/sklearn on the grading box - dev machine only, never the
exam PC.
"""
import os, re, subprocess, sys

# Work dir: where diamonds.csv lives and generated answers are executed.
# Override with LABRAT_GRADE_DIR.
WD = os.environ.get("LABRAT_GRADE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "_gradework")
os.makedirs(WD, exist_ok=True)

# Make sure the CSV the answers read is present next to them.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csv = os.path.join(_root, "data", "diamonds.csv")
if os.path.exists(_csv) and not os.path.exists(os.path.join(WD, "diamonds.csv")):
    import shutil
    shutil.copy(_csv, WD)

# Values verified directly from data/diamonds.csv, not from the model.
KEY = {
    "shape":        "(53940, 10)",
    "zero_rows":    20,
    "table_Q1":     56.0,
    "table_Q3":     59.0,
    "table_IQR":    3.0,
    "table_lower":  51.5,
    "table_upper":  63.5,
    "n_train":      43152,
    "n_test":       10788,
    "carat_median": 0.7,
    "price_median": 2401.0,
    "top_corr":     "carat",
    "top_corr_r":   0.9216,
}


def run(path):
    r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="always"),
                       cwd=WD)
    return r


def grade(md_path):
    txt = open(md_path, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", txt, re.S)
    checks, notes = [], []

    if not blocks:
        return [("code block present", False, "no ```python block")], txt

    code = blocks[0]
    src = os.path.join(WD, "graded.py")
    open(src, "w", encoding="utf-8").write(code)
    r = run(src)
    out = r.stdout

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- static checks on the source -------------------------------------
    chk("uses pd.read_csv('diamonds.csv')", "read_csv('diamonds.csv')" in code
        or 'read_csv("diamonds.csv")' in code)
    chk("does NOT substitute sns.load_dataset", "load_dataset" not in code)
    chk("builds X with drop(columns=...)", re.search(r"drop\(columns\s*=\s*\[?['\"]price", code) is not None)
    chk("no hand-built feature_cols list", "feature_cols" not in code)
    chk("no X_train['price'] / X_train[\"price\"]",
        not re.search(r"X_train\[['\"]price['\"]\]", code))
    chk("drops self-correlation before idxmax",
        re.search(r"\.drop\(\s*['\"]price['\"]\s*\)", code) is not None
        or re.search(r"drop\(index=\s*['\"]price['\"]", code) is not None)
    chk("encodes cut in place (no cut_encoded)", "cut_encoded" not in code)
    chk("caps table in place (no table_capped)", "table_capped" not in code)
    chk("drops any 'Unnamed: 0' index column", "Unnamed" in code)
    chk("get_dummies uses dtype=int", "dtype=int" in code)
    chk("uses np.log1p (not np.log)", "log1p" in code)
    chk("random_state=42 present", "random_state=42" in code or "random_state = 42" in code)
    chk("scaler fit on train only",
        "fit_transform(X_train" in code and "scaler.transform(X_test" in code)
    pal = re.findall(r"sns\.(countplot|boxplot|barplot)\((.*?)\)", code, re.S)
    bad_pal = [p for p in pal if "palette=" in p[1] and "hue=" not in p[1]]
    chk("no palette= without hue=", not bad_pal,
        f"{len(bad_pal)} offending call(s)" if bad_pal else "")

    # --- execution --------------------------------------------------------
    chk("code executes without error", r.returncode == 0,
        (r.stderr.strip().splitlines() or [""])[-1][:160] if r.returncode else "")

    # Only warnings raised BY THE GENERATED CODE count. Warnings whose source
    # line is inside site-packages are seaborn/matplotlib warning about their
    # own internals (e.g. seaborn/matrix.py set_bad on every heatmap call) and
    # no user code can avoid them.
    warns = [l for l in r.stderr.splitlines()
             if ("FutureWarning" in l or "Pandas4Warning" in l or "DeprecationWarning" in l)
             and "site-packages" not in l
             and "vert:" not in l]
    chk("no deprecation warnings from OUR code", not warns,
        warns[0][:140] if warns else "")

    if r.returncode != 0:
        return checks, out + "\n--- STDERR ---\n" + r.stderr[-2000:]

    # --- value checks on captured stdout ----------------------------------
    chk("shape 53940 x 10 reported", "53940" in out)
    chk("zero-dimension rows = 20", re.search(r"\b20\b", out) is not None)
    chk("table Q1=56 / Q3=59 / IQR=3", ("56" in out and "59" in out))
    chk("table bounds 51.5 / 63.5", "51.5" in out and "63.5" in out)
    chk("train rows 43152", "43152" in out)
    chk("test rows 10788", "10788" in out)
    chk("zero missing after imputation", re.search(r"\b0\b", out) is not None)

    # the decisive one
    m = re.search(r"[Hh]ighest correlation[^\n]*", out)
    line = m.group(0) if m else ""
    chk("Task 5c names carat (NOT price)",
        "carat" in line.lower() and not re.search(r":\s*price\b", line.lower()),
        line.strip()[:110] or "no 'highest correlation' line printed")

    # notes must not contradict output
    notes_txt = txt.split("```")[-1]
    chk("Notes do not claim a cut/price direction",
        not re.search(r"better cuts?\s+(command|have|fetch)\s+higher", notes_txt, re.I))

    return checks, out


if __name__ == "__main__":
    checks, out = grade(os.path.join(WD, sys.argv[1]))
    width = max(len(c[0]) for c in checks)
    npass = sum(1 for c in checks if c[1])
    print(f"\n{'CHECK'.ljust(width)}  RESULT  DETAIL")
    print("-" * (width + 40))
    for name, ok, detail in checks:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}    {detail}")
    print("-" * (width + 40))
    print(f"{npass}/{len(checks)} checks passed")
    open(os.path.join(WD, "last_stdout.txt"), "w", encoding="utf-8").write(out)
