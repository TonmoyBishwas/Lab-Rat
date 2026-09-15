r"""GENERALIZATION test: is the prompt tuned to bank.csv, or to the syllabus?

    python eval\grade_titanic_mldl.py
    python eval\grade_titanic_mldl.py regrade <runs\...-titanic.md>

56/56 on the bank paper and its typo variant is evidence about ONE paper typed
two ways. This grader runs a DIFFERENT paper, written from the Part 3 and Part 4
handouts, on a different dataset, with different models, a different
architecture, a different fold count, and - the two that matter most:

  * THE THRESHOLD DIRECTION IS INVERTED. The bank paper raised it to 0.65, so
    False Positives fell. This one LOWERS it to 0.35, so False Positives must
    RISE. A model reciting the other paper's conclusion gets this wrong; only
    one that reasons about the threshold gets it right.
  * COLUMNTRANSFORMER FLIPS FROM BANNED TO REQUIRED. The prompt tells the model
    not to reach for it when X is already a chosen numeric list. Q3(c) asks for
    it outright, so this checks the gate suppressed a reflex, not a capability.

Same three-fresh-chats flow as the bank grader: every block is a separate
[system, user] pair with no history, and the three answers are concatenated and
executed as one notebook.

Needs pandas/seaborn/sklearn on the grading box - dev machine only.
"""
import ast, json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dataset_scan

WD = os.environ.get("LABRAT_GRADE_DIR") or os.path.join(ROOT, "eval", "_gradework")
os.makedirs(WD, exist_ok=True)
BANK = os.path.join(ROOT, "courses", "da-python", "evals", "class_test_4_titanic.md")
AI_PORT = os.environ.get("LABRAT_AI_PORT", "11434")
AI = f"http://127.0.0.1:{AI_PORT}/v1/chat/completions"

# Ground truth from eval/ref_titanic_mldl.py, and cross-checked against the
# figures the Part 3/4 handouts print for the same quantities.
KEY = {
    "n_train": 712, "n_test": 179, "n_feat": 9,
    "shape": [891, 10],
    "cm": [[91, 19], [22, 47]],
    "cm_low": [[85, 25], [15, 54]],
    "fp_default": 19, "fp_low": 25,
    "mlp_loss": 0.3058,
}

# The paper gives this much and no more: Q1(a) has to do the cleaning.
STARTER = """
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
try:
    df = sns.load_dataset('titanic')[cols].copy()
except Exception:
    df = pd.read_csv('titanic.csv')[cols].copy()
print("Loaded:", df.shape)
print(df.isnull().sum())
"""

PROBE = r'''
# ---- grader probe (appended, not model-written) ----
import json as _json, numpy as _np
_o = {}
def _t(k, fn):
    try:
        v = fn()
        _o[k] = v.tolist() if hasattr(v, "tolist") else v
    except Exception as _e:
        _o[k] = "ERR:" + type(_e).__name__ + ": " + str(_e)[:90]

_t("df_shape",  lambda: list(df.shape))
_t("df_missing", lambda: int(df.isnull().sum().sum()))
# sex must be 0/1, not still text, and not silently all-NaN from a bad .map()
_t("sex_vals",  lambda: sorted(set(int(v) for v in df['sex'].dropna().unique())))
_t("sex_nan",   lambda: int(df['sex'].isna().sum()))
_t("emb_cols",  lambda: sorted(c for c in df.columns if str(c).startswith('emb')))
_t("train_shape", lambda: list(X_train.shape))
_t("scaled_shape", lambda: list(_np.asarray(X_train_s).shape))
_t("train_s_mean", lambda: round(float(_np.asarray(X_train_s).mean()), 6))
_t("train_s_std",  lambda: round(float(_np.asarray(X_train_s).std()), 4))
_t("mlp_hidden",  lambda: list(mlp.hidden_layer_sizes))
_t("mlp_maxiter", lambda: int(mlp.max_iter))
_t("mlp_layers",  lambda: int(mlp.n_layers_))
_t("mlp_loss",    lambda: round(float(mlp.loss_), 4))
_t("mlp_nfeat",   lambda: int(mlp.n_features_in_))

from sklearn.metrics import confusion_matrix as _cmf
_t("cm_live", lambda: _cmf(y_test, mlp.predict(X_test_s)).tolist())
_t("cm_custom_var", lambda: [v.tolist() for k, v in list(globals().items())
                             if ("low" in k or "custom" in k or "35" in k)
                             and hasattr(v, "shape") and tuple(v.shape) == (2, 2)][0])

from sklearn.pipeline import Pipeline as _Pl
from sklearn.compose import ColumnTransformer as _CT
_pipes = [v for v in list(globals().values()) if isinstance(v, _Pl)]
_t("pipe_count", lambda: len(_pipes))
# Q3(c) demands a ColumnTransformer INSIDE a pipeline, fitted on the RAW frame.
_has_ct = [p for p in _pipes if any(isinstance(st[1], _CT) for st in p.steps)]
_t("ct_pipe_count", lambda: len(_has_ct))
_fitct = []
for _p in _has_ct:
    try:
        _p.n_features_in_
        _fitct.append(_p)
    except Exception:
        pass
_t("ct_fitted", lambda: len(_fitct))
# raw titanic has 7 feature columns, all still text/NaN-bearing
_t("ct_nfeat", lambda: [int(p.n_features_in_) for p in _fitct])
_t("ct_names", lambda: [[str(c) for c in getattr(p, "feature_names_in_", [])]
                        for p in _fitct])

_json.dump(_o, open("probe.json", "w"))
print("PROBE_OK", len(_o), "keys")
'''


def load_bank():
    txt = open(BANK, encoding="utf-8").read()
    # Keep the "Q2." label. The student pastes the question AS PRINTED, and the
    # number is the strongest signal that a block continues earlier work.
    # Stripping it removed that signal and then blamed the prompt for not
    # seeing it.
    return [f"{lbl}. {b.strip()}" for lbl, b in
            re.findall(r"^## (Q\d)\n(.*?)(?=^## Q\d|\n---\n|\Z)", txt, re.S | re.M)]


def sys_prompt():
    p = open(os.path.join(ROOT, "courses", "da-python", "prompt.md"),
             encoding="utf-8").read().strip()
    scan = dataset_scan.scan(dataset_scan.resolve("titanic", ROOT))
    return p + "\n\n" + scan["text"] if scan["ok"] else p


def ask(messages, max_tokens=8192, temp=0.3):
    body = json.dumps({"model": "local", "messages": messages, "stream": True,
                       "temperature": temp, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(AI, data=body,
                                 headers={"Content-Type": "application/json"})
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


def code_of(text):
    out = []
    for b in blocks_of(text):
        try:
            ast.parse(b)
        except SyntaxError:
            continue
        out.append(b)
    return "\n".join(out)


def prose_fences(text):
    bad = []
    for b in blocks_of(text):
        if not b.strip():
            continue
        try:
            ast.parse(b)
        except SyntaxError:
            bad.append(b.strip()[:60])
    return bad


# A LOWERED threshold must RAISE false positives. The wrong claim here is the
# bank paper's conclusion recited on the wrong paper.
_WRONG_FP = [
    r"false positives?\s*(?:\([^)]*\))?\s*(?:counts?\s*)?"
    r"(?:ha(?:ve|s|d)\s+|were\s+|was\s+|are\s+|is\s+)?"
    r"(?:decreas|reduc|fell|fall|drop|declin|lower|fewer)",
    r"(?:decrease|reduction|drop|decline|fall)\s+in\s+(?:the\s+)?(?:number\s+of\s+)?false positives?",
    r"fewer false positives?",
]


def fp_claim_is_wrong(prose):
    for pat in _WRONG_FP:
        m = re.search(pat, prose, re.I | re.S)
        if m:
            return m.group(0)
    return None


def _selftest_fp():
    ok = [
        "Lowering the threshold to 0.35 increases False Positives (25 vs 19) "
        "and reduces False Negatives.",
        "False Positives rose from 19 to 25 while False Negatives fell to 15.",
        "More passengers are flagged, so false positives go up.",
    ]
    bad = [
        "The number of False Positives decreased after the change.",
        "False Positives (FP) fell from 19 to 25.",
        "There is a reduction in false positives.",
        "This produces fewer false positives.",
    ]
    for s in ok:
        assert fp_claim_is_wrong(s) is None, f"false alarm on: {s}"
    for s in bad:
        assert fp_claim_is_wrong(s), f"missed: {s}"
    return True


def run_code(code, tag):
    src = os.path.join(WD, f"titanic_{tag}.py")
    open(src, "w", encoding="utf-8").write(code + "\n" + PROBE)
    probe = os.path.join(WD, "probe.json")
    if os.path.exists(probe):
        os.remove(probe)
    dst = os.path.join(WD, "titanic.csv")
    if not os.path.exists(dst):
        import shutil
        shutil.copy(os.path.join(ROOT, "data", "titanic.csv"), dst)
    try:
        r = subprocess.run([sys.executable, src], capture_output=True, text=True,
                           cwd=WD, env=dict(os.environ, MPLBACKEND="Agg"), timeout=1200)
    except subprocess.TimeoutExpired:
        class _R:
            returncode, stdout, stderr = 1, "", "TIMEOUT"
        r = _R()
    data = {}
    if os.path.exists(probe):
        try:
            data = json.load(open(probe))
        except Exception:
            pass
    return r, data


def grade(answers, tag):
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:130]))

    a1, a2, a3 = (answers + ["", "", ""])[:3]
    c1, c2, c3 = code_of(a1), code_of(a2), code_of(a3)
    allc = "\n".join([c1, c2, c3])

    # ---- continuation contract, on a paper it has never seen ----------
    # Q2 is a pure continuation: nothing there may reload or re-split.
    chk("Q2 does not re-read the data", not re.search(r"load_dataset|read_csv", c2),
        "reloading resets df and wipes the cleaning Q1 did")
    chk("Q2 does not re-split", not re.search(r"train_test_split\s*\(", c2),
        "a second split invalidates the trained models")
    chk("Q2 does not retrain a model Q1 built",
        not re.search(r"\b(dtree|knn)\s*=", c2), "")

    # Q3(c) is the deliberate EXCEPTION: it asks for the RAW, uncleaned frame
    # and its own raw split, so loading is correct there. What must never
    # happen is CLOBBERING the cleaned state the earlier blocks built - the
    # raw frame belongs in new names.
    clobber = [n for n in ("df", "X_train", "X_test", "y_train", "y_test",
                           "X_train_s", "X_test_s", "mlp", "cm")
               if re.search(rf"^\s*{n}\s*=[^=]", c3, re.M)
               or re.search(rf"^\s*{n}\s*,", c3, re.M)]
    chk("Q3 rebuilds the raw frame WITHOUT clobbering the cleaned one",
        not clobber,
        f"reassigns {clobber} - Q3(c) needs the raw data under NEW names")
    chk("Q1 does not re-emit the starter code",
        not re.search(r"load_dataset\s*\(\s*['\"]titanic", c1)
        and not re.search(r"^\s*cols\s*=\s*\[", c1, re.M),
        "the paper supplies the loader")
    chk("continuation blocks declare what they assume",
        sum(bool(re.search(r"#.*(continue|assume|uses:|from Q)", c, re.I))
            for c in (c2, c3)) >= 1, "")

    # ---- Q1: preprocessing that the bank paper never required ---------
    chk("age imputed with median", re.search(r"age.{0,40}median\(\)", c1), "")
    chk("embarked imputed with mode", re.search(r"embarked.{0,40}mode\(\)\s*\[\s*0\s*\]", c1),
        "mode() returns a Series - [0] is required")
    chk("sex mapped, not label-encoded",
        ".map(" in c1 and "LabelEncoder" not in allc, "")
    chk("one-hot via get_dummies with dtype=int",
        re.search(r"get_dummies\([^)]*dtype\s*=\s*int", c1), "")
    chk("one-hot names derived, never hand-typed",
        not re.search(r"['\"]emb_[CQS]['\"]", c1),
        "hand-typing emb_C/emb_Q/emb_S is the KeyError pattern")
    chk("Q1(b) target dropped, not hand-listed",
        re.search(r"drop\(\s*columns\s*=\s*\[?\s*['\"]survived", c1)
        or re.search(r"drop\(\s*['\"]survived['\"]", c1), "")
    chk("split is stratified with random_state=42",
        re.search(r"stratify\s*=\s*y", c1) and re.search(r"random_state\s*=\s*42", c1), "")
    chk("scaler fit on train only",
        re.search(r"fit_transform\(\s*X_train\b", c1)
        and re.search(r"\.transform\(\s*X_test\b", c1), "")
    chk("DecisionTreeClassifier(random_state=42)",
        re.search(r"DecisionTreeClassifier\([^)]*random_state\s*=\s*42", c1), "")
    chk("KNeighborsClassifier(n_neighbors=5)",
        re.search(r"KNeighborsClassifier\([^)]*n_neighbors\s*=\s*5", c1), "")

    # ---- Q2: the DL block, different attribute and palette ------------
    chk("MLP (16, 8) with max_iter=1000",
        re.search(r"hidden_layer_sizes\s*=\s*\(\s*16\s*,\s*8\s*\)", c2)
        and re.search(r"max_iter\s*=\s*1000", c2), "")
    chk("Q2(a) reports n_layers_", "n_layers_" in c2, "asked for explicitly")
    chk("Q2(a) reports .loss_", re.search(r"mlp\.loss_(?!curve)", c2), "")
    chk("heatmap uses cmap='Purples' as asked",
        re.search(r"cmap\s*=\s*['\"]Purples['\"]", c2),
        "the question names the palette")
    chk("heatmap fmt='d' and both tick labels",
        re.search(r"fmt\s*=\s*['\"]d['\"]", c2)
        and c2.count("Died") >= 2 and c2.count("Survived") >= 2, "")
    chk("classification_report with target_names",
        "classification_report" in c2 and "target_names" in c2, "")
    chk("loss curve plotted and labelled",
        "loss_curve_" in c2 and all(k in c2 for k in ("xlabel", "ylabel", "title")), "")

    # ---- Q3: sweep, 10-fold, inverted threshold, ColumnTransformer ----
    chk("all four architectures compared",
        all(re.search(p, c3) for p in
            (r"\(\s*8\s*,\s*\)", r"\(\s*16\s*,\s*\)",
             r"\(\s*16\s*,\s*8\s*\)", r"\(\s*32\s*,\s*16\s*,\s*8\s*\)")),
        "the question names four designs")
    chk("cross-validation is 10-fold, not 5",
        re.search(r"n_splits\s*=\s*10|cv\s*=\s*10", c3)
        and not re.search(r"n_splits\s*=\s*5|cv\s*=\s*5\b", c3),
        "carrying 5 over from the other paper is the tell")
    chk("CV mean AND std reported",
        re.search(r"\.mean\(\)", c3) and re.search(r"\.std\(\)", c3), "")
    chk("CV runs on the Pipeline with RAW X",
        "cross_val_score" in c3 and not re.search(r"cross_val_score\([^)]*X_train_s", c3), "")
    chk("threshold 0.35 applied to predict_proba[:, 1]",
        re.search(r"predict_proba\([^)]*\)\s*\[\s*:\s*,\s*1\s*\]", c3)
        and "0.35" in c3, "")
    chk("FP counts extracted for the comparison",
        re.search(r"\.ravel\(\)|cm\w*\[0\s*,\s*1\]|cm\w*\[0\]\[1\]", c3), "")
    chk("Q3(c) uses ColumnTransformer as asked",
        "ColumnTransformer" in c3, "the gate must not suppress the capability")
    chk("both SimpleImputer strategies present",
        re.search(r"strategy\s*=\s*['\"]median['\"]", c3)
        and re.search(r"strategy\s*=\s*['\"]most_frequent['\"]", c3), "")
    chk("OneHotEncoder(handle_unknown='ignore')",
        re.search(r"handle_unknown\s*=\s*['\"]ignore['\"]", c3),
        "without it an unseen category raises at predict time")
    chk("new_passenger passed in raw",
        re.search(r"predict\(\s*new_passenger", c3)
        and re.search(r"predict_proba\(\s*new_passenger", c3), "")

    # ---- it has to run ------------------------------------------------
    r, probe = run_code(STARTER + "\n" + allc, tag)
    out = r.stdout
    chk("the whole notebook executes", r.returncode == 0,
        (r.stderr.strip().splitlines() or ["?"])[-1])
    chk("probe reached the end", "PROBE_OK" in out, "script died before the probe")

    def pv(k):
        return probe.get(k)

    chk("cleaned frame is 891 x 10 with nothing missing",
        pv("df_shape") == KEY["shape"] and pv("df_missing") == 0,
        f"{pv('df_shape')} missing={pv('df_missing')}")
    chk("sex really became 0/1 (not silently NaN)",
        pv("sex_vals") == [0, 1] and pv("sex_nan") == 0,
        f"values={pv('sex_vals')} nan={pv('sex_nan')}")
    chk("embarked one-hot produced emb_C/emb_Q/emb_S",
        pv("emb_cols") == ["emb_C", "emb_Q", "emb_S"], pv("emb_cols"))
    chk("split is 712 / 179 with 9 features",
        pv("train_shape") == [KEY["n_train"], KEY["n_feat"]], pv("train_shape"))
    chk("scaler fitted on train only",
        isinstance(pv("train_s_mean"), float) and abs(pv("train_s_mean")) < 1e-6
        and abs((pv("train_s_std") or 0) - 1.0) < 1e-3,
        f"mean {pv('train_s_mean')} std {pv('train_s_std')}")
    chk("live MLP matches the specified architecture",
        pv("mlp_hidden") == [16, 8] and pv("mlp_maxiter") == 1000
        and pv("mlp_layers") == 4,
        f"{pv('mlp_hidden')} max_iter={pv('mlp_maxiter')} layers={pv('mlp_layers')}")
    chk("MLP trained on scaled features (loss matches reference)",
        pv("mlp_nfeat") == KEY["n_feat"] and isinstance(pv("mlp_loss"), float)
        and abs(pv("mlp_loss") - KEY["mlp_loss"]) < 0.05,
        f"loss {pv('mlp_loss')} vs {KEY['mlp_loss']}")
    chk("default confusion matrix matches the reference",
        pv("cm_live") == KEY["cm"], f"{pv('cm_live')} vs {KEY['cm']}")
    chk("relaxed-threshold matrix matches the reference",
        pv("cm_custom_var") == KEY["cm_low"],
        f"{pv('cm_custom_var')} vs {KEY['cm_low']}")
    chk("a ColumnTransformer pipeline was actually fitted",
        pv("ct_fitted") == 1, f"{pv('ct_fitted')} fitted of {pv('ct_pipe_count')} CT pipelines")
    chk("that pipeline got the 7 RAW named columns",
        isinstance(pv("ct_nfeat"), list) and pv("ct_nfeat") == [7]
        and isinstance(pv("ct_names"), list) and pv("ct_names")
        and len(pv("ct_names")[0]) == 7,
        f"nfeat={pv('ct_nfeat')} names={pv('ct_names')}")

    # ---- the prose -----------------------------------------------------
    prose = "\n".join(re.sub(r"```.*?```", "", a, flags=re.S) for a in answers)
    stray = [i for i, a in enumerate((a1, a2, a3), 1)
             if any(not b.strip() for b in blocks_of(a))]
    multi = [i for i, a in enumerate((a1, a2, a3), 1)
             if len([b for b in blocks_of(a) if b.strip()]) > 1]
    chk("one fenced code block per answer", not multi, f"block(s) {multi}")
    chk("no stray closing fence", not stray, f"block(s) {stray}")
    chk("nothing but Python inside a fence", not prose_fences(a1) and not
        prose_fences(a2) and not prose_fences(a3),
        " | ".join(prose_fences(a1) + prose_fences(a2) + prose_fences(a3)))
    tells = [t for t in ("dtype:", "Name: count", "Output:", "macro avg",
                         "precision    recall") if t in prose]
    chk("no fabricated output transcript", not tells, ", ".join(tells))

    # THE headline check of this whole file.
    wrong = fp_claim_is_wrong(prose)
    chk("prose gets the INVERTED FP direction right", not wrong,
        f"recited the bank paper's answer: {wrong!r}")
    chk("Q3(b) answers the rescue-screening 'why'",
        re.search(r"miss|rescue|search|recall|costl|false negativ", prose, re.I), "")

    return checks, probe, out + "\n--- STDERR (tail) ---\n" + r.stderr[-1800:]


def report(title, checks):
    w = max(len(c[0]) for c in checks)
    n = sum(1 for c in checks if c[1])
    print(f"\n{'='*(w+34)}\n {title}\n{'='*(w+34)}")
    for name, ok, det in checks:
        print(f"  {name.ljust(w)}  {'PASS' if ok else 'FAIL'}  {'' if ok else det}")
    print(f"  {'-'*(w+30)}")
    print(f"  {str(n).rjust(len(str(len(checks))))}/{len(checks)} checks passed")
    return n, len(checks)


def regrade(path):
    txt = open(path, encoding="utf-8").read()
    answers = [s for s in re.split(r"\n\n---\n\n", txt) if s.strip()]
    c, probe, out = grade(answers, "regrade")
    report(os.path.basename(path), c)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "regrade":
        regrade(sys.argv[2])
        return
    assert _selftest_fp()
    blocks = load_bank()
    sp = sys_prompt()
    print(f"system prompt: {len(sp)} chars (~{len(sp)//4} tokens)")
    print(f"blocks: {len(blocks)}  (a paper the prompt has never been tuned on)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = os.path.join(ROOT, "courses", "da-python", "evals", "runs")
    os.makedirs(runs, exist_ok=True)
    answers = []
    for i, b in enumerate(blocks, 1):
        t0 = time.time()
        a = ask([{"role": "system", "content": sp}, {"role": "user", "content": b}])
        answers.append(a)
        print(f"  block {i}: {int(time.time()-t0)}s, {len(a)} chars, "
              f"{len([x for x in blocks_of(a) if x.strip()])} code block(s)", flush=True)
    open(os.path.join(runs, f"{stamp}-titanic.md"), "w", encoding="utf-8").write(
        "\n\n---\n\n".join(answers))
    c, probe, out = grade(answers, "titanic")
    open(os.path.join(WD, "titanic_stdout.txt"), "w", encoding="utf-8").write(out)
    report("GENERALIZATION (unseen paper, 3 fresh chats)", c)


if __name__ == "__main__":
    main()
