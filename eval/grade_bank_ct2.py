r"""Objectively grade the REAL bank.csv Class Test 02 paper (ML + DL pipeline).

    python eval\grade_bank_ct2.py           # both modes
    python eval\grade_bank_ct2.py clean     # the paper as printed
    python eval\grade_bank_ct2.py typo      # as the student actually types it
    python eval\grade_bank_ct2.py regrade <runs\...-clean.md>

Assumes llama-server is listening on :11434 (launch-da.bat, or
python start.py da-python).

WHAT MAKES THIS GRADER DIFFERENT FROM THE OTHER TWO
---------------------------------------------------
The student pastes ONE QUESTION BLOCK per BRAND-NEW CHAT. So Q2 and Q3 reach
the model with NO conversation history at all: no dataset description, no
starter code, no memory of what Q1 produced. This grader reproduces that
exactly - every block is sent as a fresh [system, user] pair and the previous
answers are deliberately NOT in context.

That makes the whole paper hinge on one thing: whether the model recognises
"this continues a notebook I cannot see" and uses the CANONICAL VARIABLE NAMES
the earlier block created, instead of rebuilding the pipeline from scratch.
The three answers are then concatenated in notebook order - starter code first,
because the paper supplies it and the student has already run it - and executed
as one script. A rebuild, a renamed variable or a missing import therefore
shows up as a real traceback, not as a matter of opinion.

After it runs, a PROBE (appended by the grader, never written by the model)
inspects the LIVE objects - the scaler, the fitted MLP, the confusion matrices -
and compares them against values computed by eval/ref_bank_ct2.py. Code that
runs clean and reads correct can still be wrong; only the live state settles it.

Needs pandas/seaborn/sklearn on the grading box - dev machine only.
"""
import ast, json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dataset_scan

WD = os.environ.get("LABRAT_GRADE_DIR") or os.path.join(ROOT, "eval", "_gradework")
os.makedirs(WD, exist_ok=True)
BANK = os.path.join(ROOT, "courses", "da-python", "evals", "class_test_3_bank.md")
# Port is overridable: a dev box may already have something (Ollama, say)
# squatting on 11434.
AI_PORT = os.environ.get("LABRAT_AI_PORT", "11434")
AI = f"http://127.0.0.1:{AI_PORT}/v1/chat/completions"

# Ground truth from eval/ref_bank_ct2.py - never from the model.
KEY = {
    "n_train": 3616, "n_test": 905, "n_feat": 7,
    "cm": [[768, 33], [79, 25]],
    "cm_custom": [[783, 18], [90, 14]],
    "fp_default": 33, "fp_custom": 18,
    "top2": ["duration", "balance"],
    "mlp_loss": 0.2034,
}

# The paper hands the student this block and they run it before Q1(a).
# It is part of their notebook, so it is part of the script we execute - but
# the model must NOT re-emit it.
STARTER = """
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('bank.csv', sep=';')
features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
X = df[features].copy()
y = df['y'].map({'no': 0, 'yes': 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Data ready:", X_train.shape, X_test.shape)
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

_t("train_shape", lambda: list(X_train.shape))
_t("scaled_shape", lambda: list(_np.asarray(X_train_s).shape))
# A correctly fitted StandardScaler leaves the TRAIN mean at 0 and std at 1.
# Fitting on the full frame or on the test set moves both off.
_t("train_s_mean", lambda: round(float(_np.asarray(X_train_s).mean()), 6))
_t("train_s_std",  lambda: round(float(_np.asarray(X_train_s).std()), 4))
_t("test_s_mean",  lambda: round(float(_np.asarray(X_test_s).mean()), 4))
_t("mlp_hidden",   lambda: list(mlp.hidden_layer_sizes))
_t("mlp_act",      lambda: str(mlp.activation))
_t("mlp_solver",   lambda: str(mlp.solver))
_t("mlp_maxiter",  lambda: int(mlp.max_iter))
_t("mlp_seed",     lambda: int(mlp.random_state))
_t("mlp_loss",     lambda: round(float(mlp.loss_), 4))
_t("mlp_curve_len",lambda: len(mlp.loss_curve_))
_t("mlp_fitted_on_scaled", lambda: int(mlp.n_features_in_))
_t("rf_trees",     lambda: int(rf.n_estimators))
_t("logreg_acc",   lambda: round(float(logreg.score(X_test_s, y_test)), 4))
_t("rf_acc",       lambda: round(float(rf.score(X_test_s, y_test)), 4))

from sklearn.metrics import confusion_matrix as _cmf
_t("cm_live", lambda: _cmf(y_test, mlp.predict(X_test_s)).tolist())

# Whatever the model called its default matrix, find one that matches.
_t("cm_var", lambda: [v.tolist() for k, v in list(globals().items())
                      if k in ("cm", "cm_default", "cm_mlp")
                      and hasattr(v, "shape") and tuple(v.shape) == (2, 2)][0])
_t("cm_custom_var", lambda: [v.tolist() for k, v in list(globals().items())
                             if "custom" in k and hasattr(v, "shape")
                             and tuple(v.shape) == (2, 2)][0])
_t("importance_index", lambda: [str(i) for i in importance.index[:2]])
# The block may legitimately hold two pipelines under any names (pipe_rf,
# mlp_pipe, ...), so find them by type rather than by a name we guessed.
from sklearn.pipeline import Pipeline as _Pl
_pipes = [v for v in list(globals().values()) if isinstance(v, _Pl)]
_t("pipe_count", lambda: len(_pipes))
_t("pipe_steps", lambda: [[st[0] for st in p.steps] for p in _pipes])
# A Pipeline that contains a scaler must be FITTED ON RAW X. Handed X_train_s
# it still runs - silently double-scaled - and keeps no feature_names_in_,
# which a DataFrame-fitted pipeline always has.
#
# Only FITTED pipelines can be judged: the one passed to cross_val_score is
# never fitted itself (cross_val_score clones it), and that is correct usage.
def _fitted(p):
    return hasattr(p, "feature_names_in_") or "n_features_in_" in p.__dict__
_fit_pipes = []
for _p in _pipes:
    try:
        _p.n_features_in_          # raises unless the pipeline was fitted
        _fit_pipes.append(_p)
    except Exception:
        pass
_t("fitted_pipe_count", lambda: len(_fit_pipes))
_t("pipe_nfeat", lambda: [int(p.n_features_in_) for p in _fit_pipes])
_t("pipe_feature_names", lambda: [[str(c) for c in getattr(p, "feature_names_in_", [])]
                                  for p in _fit_pipes])

_json.dump(_o, open("probe.json", "w"))
print("PROBE_OK", len(_o), "keys")
'''


def load_bank():
    txt = open(BANK, encoding="utf-8").read()
    clean = re.findall(r"^## (Q\d)\n(.*?)(?=^## Q\d|\n---\n|\Z)", txt, re.S | re.M)
    typo = re.findall(r"^## (T\d)\n(.*?)(?=^## T\d|\n---\n|\Z)", txt, re.S | re.M)
    return [b.strip() for _, b in clean], [b.strip() for _, b in typo]


def sys_prompt():
    p = open(os.path.join(ROOT, "courses", "da-python", "prompt.md"), encoding="utf-8").read().strip()
    scan = dataset_scan.scan(dataset_scan.resolve("bank", ROOT))
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


# Raising the threshold above 0.5 can only REDUCE false positives, so prose
# claiming they went up is wrong whatever the split. Getting this check right
# is fiddly, because the CORRECT answer is usually phrased
# "reduces False Positives and increases False Negatives" - a naive proximity
# match flags that as wrong, and "Increasing the threshold" trips a naive
# clause match. So: the increase verb must attach DIRECTLY to false positives,
# with nothing between but a parenthetical or an auxiliary. Tested below.
_WRONG_FP = [
    r"false positives?\s*(?:\([^)]*\))?\s*(?:counts?\s*)?"
    r"(?:ha(?:ve|s|d)\s+|were\s+|was\s+|are\s+|is\s+)?"
    r"(?:increas|rose|grew|went up|higher|climb)",
    r"(?:increase|rise|growth|jump)\s+in\s+(?:the\s+)?(?:number\s+of\s+)?false positives?",
    r"more false positives?",
]


def fp_claim_is_wrong(prose):
    """Return the offending phrase, or None if the prose is fine."""
    for pat in _WRONG_FP:
        m = re.search(pat, prose, re.I | re.S)
        if m:
            return m.group(0)
    return None


def _selftest_fp():
    ok = [
        "Increasing the threshold (0.5 -> 0.65) reduces False Positives (FP) "
        "and increases False Negatives (FN).",
        "The count of False Positives fell from 33 to 18, while False "
        "Negatives rose.",
        "Raising T reduces false positives; recall drops.",
    ]
    bad = [
        "The number of False Positives increased after raising the threshold.",
        "False Positives (FP) rose sharply.",
        "There was an increase in false positives.",
        "This produces more false positives.",
        "False positives are higher at T=0.65.",
    ]
    for s in ok:
        assert fp_claim_is_wrong(s) is None, f"false alarm on: {s}"
    for s in bad:
        assert fp_claim_is_wrong(s), f"missed: {s}"
    return True


def blocks_of(text):
    return re.findall(r"```(?:python)?\n(.*?)```", text, re.S)


def code_of(text):
    """Concatenate only the fenced sections that are actually Python.

    A Notes line wrapped in a bare ``` fence is prose, and a student would
    never paste it into a code cell. Executing it turns one formatting slip
    into a SyntaxError that wipes out every probe check downstream - 13
    cascading failures that say nothing about the answer's correctness. The
    slip is still reported, by its own check ("one fenced code block").
    """
    out = []
    for b in blocks_of(text):
        try:
            ast.parse(b)
        except SyntaxError:
            continue
        out.append(b)
    return "\n".join(out)


def prose_fences(text):
    """Fenced sections that are NOT valid Python - i.e. fenced prose."""
    bad = []
    for b in blocks_of(text):
        if not b.strip():
            continue                      # empty fence: reported as `stray`
        try:
            ast.parse(b)
        except SyntaxError:
            bad.append(b.strip()[:60])
    return bad


def run_code(code, tag):
    src = os.path.join(WD, f"bank_{tag}.py")
    open(src, "w", encoding="utf-8").write(code + "\n" + PROBE)
    probe = os.path.join(WD, "probe.json")
    if os.path.exists(probe):
        os.remove(probe)
    dst = os.path.join(WD, "bank.csv")
    if not os.path.exists(dst):
        import shutil
        shutil.copy(os.path.join(ROOT, "data", "bank.csv"), dst)
    try:
        r = subprocess.run([sys.executable, src], capture_output=True, text=True, cwd=WD,
                           env=dict(os.environ, MPLBACKEND="Agg"), timeout=900)
    except subprocess.TimeoutExpired:
        class _R:
            returncode, stdout, stderr = 1, "", "TIMEOUT after 900s"
        r = _R()
    data = {}
    if os.path.exists(probe):
        try:
            data = json.load(open(probe))
        except Exception:
            pass
    return r, data


def grade(answers, tag):
    """answers: the three block answers, in paper order."""
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:130]))

    a1, a2, a3 = (answers + ["", "", ""])[:3]
    # One answer = one fenced block, and it contains only runnable Python.
    # A Notes line inside a bare ``` fence reads as a second code cell, and the
    # student pastes it and gets a SyntaxError.
    fenced_prose = [f for a in (a1, a2, a3) for f in prose_fences(a)]
    # An EMPTY fenced section means a stray closing fence, which is a different
    # defect from a fenced Notes line - report them separately so the fix is
    # obvious from the report.
    stray = [i for i, a in enumerate((a1, a2, a3), 1)
             if any(not b.strip() for b in blocks_of(a))]
    multi = [i for i, a in enumerate((a1, a2, a3), 1)
             if len([b for b in blocks_of(a) if b.strip()]) > 1]
    c1, c2, c3 = code_of(a1), code_of(a2), code_of(a3)
    allc = "\n".join([c1, c2, c3])

    # ---- 1. THE CONTINUATION CONTRACT ----------------------------------
    # This is the whole point of the paper's three-fresh-chats shape.
    for i, c in ((2, c2), (3, c3)):
        chk(f"Q{i} does not re-read the CSV", "read_csv" not in c,
            "reloading resets df and wipes every earlier cell")
        chk(f"Q{i} does not re-split", not re.search(r"train_test_split\s*\(", c),
            "a second split renames X_test and invalidates the trained models")
    chk("Q2 does not re-fit the scaler", "fit_transform" not in c2,
        "re-fitting the scaler changes X_test_s under the trained model")
    chk("Q2 does not retrain the MLP", not re.search(r"\bmlp\s*=\s*MLPClassifier", c2),
        "Q1(c) already trained it; Q2 only evaluates")
    chk("Q1 does not re-emit the starter code",
        "read_csv" not in c1 and not re.search(r"train_test_split\s*\(", c1),
        "the paper supplies and the student has already run the loader + split")
    chk("continuation blocks declare what they assume",
        sum(bool(re.search(r"#.*(continue|assume|uses:|from Q|existing)", c, re.I))
            for c in (c2, c3)) >= 1,
        "no '# continues the notebook - uses: ...' header in Q2 or Q3")

    # ---- 2. CANONICAL NAMES --------------------------------------------
    chk("uses X_train_s / X_test_s", "X_train_s" in c1 and "X_test_s" in c1,
        "the _s names are the contract Q2 and Q3 depend on")
    chk("names the models logreg / rf / mlp",
        all(re.search(rf"\b{n}\s*=", c1) for n in ("logreg", "rf", "mlp")),
        "Q2/Q3 arrive with no history and can only guess these names")

    # ---- 3. Q1: SCALE, TRAIN, REPORT -----------------------------------
    # X_train[num_cols] is just as correct as a bare X_train - match the NAME,
    # not the exact punctuation that follows it.
    chk("scaler fit on X_train only",
        re.search(r"fit_transform\(\s*X_train\b", c1)
        and re.search(r"\.transform\(\s*X_test\b", c1),
        "must be fit_transform(X_train) then transform(X_test)")
    chk("scaler never fit on test or full X",
        not re.search(r"fit_transform\(\s*X_test", allc)
        and not re.search(r"fit(_transform)?\(\s*X\s*[,)]", c1),
        "fitting on test or on the whole frame leaks test statistics")
    chk("LogisticRegression(max_iter=1000)",
        re.search(r"LogisticRegression\([^)]*max_iter\s*=\s*1000", c1), "")
    chk("RandomForestClassifier(n_estimators=100, random_state=42)",
        re.search(r"RandomForestClassifier\([^)]*n_estimators\s*=\s*100", c1)
        and re.search(r"RandomForestClassifier\([^)]*random_state\s*=\s*42", c1), "")
    chk("both Q1(b) accuracies printed",
        len(re.findall(r"accuracy_score|\.score\(", c1)) >= 2, "")
    chk("MLP hyper-parameters exactly as specified",
        re.search(r"hidden_layer_sizes\s*=\s*\(\s*32\s*,\s*16\s*\)", c1)
        and "'relu'" in c1 and "'adam'" in c1
        and re.search(r"max_iter\s*=\s*500", c1)
        and re.search(r"random_state\s*=\s*42", c1), "")
    chk("Q1(c) prints .loss_ (not loss_curve_)",
        re.search(r"mlp\.loss_(?!curve)", c1), "the question asks for the final loss")

    # ---- 4. Q2: EVALUATION ---------------------------------------------
    chk("confusion matrix computed", "confusion_matrix" in c2, "")
    chk("seaborn heatmap with fmt='d'",
        "heatmap" in c2 and re.search(r"fmt\s*=\s*['\"]d['\"]", c2),
        "without fmt='d' counts render as 1.2e+02")
    chk("heatmap carries BOTH tick labels",
        c2.count("No Deposit") >= 2 and c2.count("Subscribed") >= 2,
        "the question names xticklabels AND yticklabels")
    chk("classification_report with target_names",
        "classification_report" in c2 and "target_names" in c2, "")
    chk("loss curve plotted from loss_curve_",
        "loss_curve_" in c2 and re.search(r"plt\.plot\(", c2), "")
    chk("loss curve labelled",
        all(k in c2 for k in ("xlabel", "ylabel", "title")),
        "the question asks for clear axis labels and a title")

    # ---- 5. Q3: IMPORTANCES, PIPELINE, CV, THRESHOLD --------------------
    chk("feature_importances_ indexed by X_train.columns",
        "feature_importances_" in c3
        and re.search(r"index\s*=\s*X_train\.columns|index\s*=\s*features", c3),
        "a scaled numpy array has no .columns; hand-typed names drift")
    chk("top 2 taken from the data, not typed",
        re.search(r"\.index\[:\s*2\s*\]|head\(\s*2\s*\)|nlargest\(\s*2", c3),
        "naming the winners in prose is a guess")
    chk("Pipeline used", "Pipeline(" in c3, "")
    chk("cross-validation is 5-fold stratified",
        "cross_val_score" in c3
        and (re.search(r"StratifiedKFold\([^)]*(n_splits\s*=\s*)?5", c3)
             or re.search(r"cv\s*=\s*5", c3)), "")
    chk("CV mean AND std reported",
        re.search(r"\.mean\(\)", c3) and re.search(r"\.std\(\)", c3), "")
    chk("cross_val_score gets RAW X, not X_train_s",
        not re.search(r"cross_val_score\([^)]*X_train_s", c3),
        "a Pipeline holding a scaler must never receive pre-scaled data")
    chk("threshold 0.65 applied to predict_proba[:, 1]",
        re.search(r"predict_proba\([^)]*\)\s*\[\s*:\s*,\s*1\s*\]", c3)
        and "0.65" in c3, "")
    chk("custom predictions cast with .astype(int)", ".astype(int)" in c3, "")
    chk("second confusion matrix computed", c3.count("confusion_matrix") >= 1, "")
    chk("False Positive COUNT extracted, not just the matrices printed",
        re.search(r"\.ravel\(\)|cm\w*\[0\s*,\s*1\]|cm\w*\[0\]\[1\]", c3),
        "the question asks what happened to the COUNT - print fp -> fp2")
    chk("MLP pipeline fitted on RAW X_train",
        re.search(r"\w*pipe\w*\.fit\(\s*X_train\s*,", c3)
        and not re.search(r"\w*pipe\w*\.fit\(\s*X_train_s", c3),
        "the pipeline contains the scaler; feeding it X_train_s scales twice")
    chk("new_client reindexed to training columns",
        re.search(r"new_client\s*=\s*new_client\[|new_client\[\s*X_train\.columns",
                  c3) or re.search(r"\[\s*X_train\.columns\s*\]", c3),
        "sklearn matches feature names AND order")
    chk("new_client prediction + probability printed",
        re.search(r"predict\(\s*new_client", c3)
        and re.search(r"predict_proba\(\s*new_client", c3), "")

    # ---- 6. IT HAS TO RUN ----------------------------------------------
    r, probe = run_code(STARTER + "\n" + allc, tag)
    out = r.stdout
    chk("the whole notebook executes", r.returncode == 0,
        (r.stderr.strip().splitlines() or ["?"])[-1])
    chk("probe reached the end", "PROBE_OK" in out, "script died before the probe")

    def pv(k):
        return probe.get(k)

    chk("X_train_s is the scaled training matrix",
        pv("scaled_shape") == [KEY["n_train"], KEY["n_feat"]], pv("scaled_shape"))
    chk("scaler really was fitted on train only",
        isinstance(pv("train_s_mean"), float) and abs(pv("train_s_mean")) < 1e-6
        and abs((pv("train_s_std") or 0) - 1.0) < 1e-3,
        f"train mean {pv('train_s_mean')} std {pv('train_s_std')} (expect 0.0 / 1.0)")
    chk("test set was transformed, not re-fitted",
        isinstance(pv("test_s_mean"), float) and abs(pv("test_s_mean")) > 1e-6,
        f"test mean {pv('test_s_mean')} - exactly 0 means fit_transform on test")
    chk("live MLP has the specified architecture",
        pv("mlp_hidden") == [32, 16] and pv("mlp_act") == "relu"
        and pv("mlp_solver") == "adam" and pv("mlp_maxiter") == 500
        and pv("mlp_seed") == 42,
        f"{pv('mlp_hidden')} {pv('mlp_act')} {pv('mlp_solver')} "
        f"{pv('mlp_maxiter')} {pv('mlp_seed')}")
    chk("MLP trained on SCALED features",
        pv("mlp_fitted_on_scaled") == KEY["n_feat"]
        and isinstance(pv("mlp_loss"), float)
        and abs(pv("mlp_loss") - KEY["mlp_loss"]) < 0.05,
        f"loss {pv('mlp_loss')} vs reference {KEY['mlp_loss']} - a big gap means "
        f"it was fitted on unscaled data")
    chk("live confusion matrix matches the reference",
        pv("cm_live") == KEY["cm"], f"{pv('cm_live')} vs {KEY['cm']}")
    chk("the model's own cm variable is the right one",
        pv("cm_var") == KEY["cm"], pv("cm_var"))
    chk("custom-threshold matrix matches the reference",
        pv("cm_custom_var") == KEY["cm_custom"],
        f"{pv('cm_custom_var')} vs {KEY['cm_custom']}")
    chk("top 2 features are duration & balance",
        pv("importance_index") == KEY["top2"], pv("importance_index"))
    names = pv("pipe_feature_names")
    chk("every FITTED pipeline got the 7 RAW named features",
        pv("fitted_pipe_count") == 1 or pv("fitted_pipe_count") == 2,
        f"expected the Q3(c) pipeline to be fitted; found "
        f"{pv('fitted_pipe_count')} fitted of {pv('pipe_count')}")
    chk("no pipeline was fed pre-scaled data",
        isinstance(pv("pipe_nfeat"), list) and pv("pipe_nfeat")
        and all(n == KEY["n_feat"] for n in pv("pipe_nfeat"))
        and isinstance(names, list) and names
        and all(len(nl) == KEY["n_feat"] for nl in names),
        f"{pv('pipe_nfeat')} features, named={names} - a pipeline fed "
        f"X_train_s keeps no feature names at all")

    # ---- 7. THE PROSE ---------------------------------------------------
    # A 31/31 penguins run once shipped correct code under an INVENTED output
    # transcript with the species counts swapped. Executing alone misses that.
    prose = "\n".join(re.sub(r"```.*?```", "", a, flags=re.S) for a in answers)
    tells = [t for t in ("dtype:", "Name: count", "Output:", "[[", "accuracy  ",
                         "macro avg", "Accuracy: 0.", "precision    recall")
             if t in prose]
    chk("one fenced code block per answer", not multi,
        f"block(s) {multi} emitted more than one code section")
    chk("no stray closing fence", not stray,
        f"block(s) {stray} ended with an extra fence marker")
    chk("nothing but Python inside a fence", not fenced_prose,
        "fenced prose: " + " | ".join(fenced_prose))
    chk("no fabricated output transcript", not tells, "invented output: " + ", ".join(tells))
    # The threshold direction is arithmetic: raising it can only cut FPs.
    wrong_dir = fp_claim_is_wrong(prose)
    chk("prose gets the False-Positive direction right", not wrong_dir, wrong_dir or "")
    chk("Q3(b) actually answers the 'why'",
        re.search(r"budget|cost|wast|spend|resource|call", prose, re.I),
        "the question awards 2 marks for the economic reasoning")
    chk("no invented accuracy figure in prose",
        not re.search(r"accuracy (of|is|was|:)\s*0?\.\d", prose, re.I),
        "every number must come from the student's own screen")

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
    """Re-score a saved run log offline, without touching the model."""
    txt = open(path, encoding="utf-8").read()
    answers = [s for s in re.split(r"\n\n---\n\n", txt) if s.strip()]
    tag = "regrade-" + os.path.basename(path).replace(".md", "")
    c, probe, out = grade(answers, tag[:40])
    report(os.path.basename(path), c)


def run_mode(name, blocks, sp, stamp, runs):
    """Every block is a FRESH chat - the previous answers are NOT in context."""
    print(f"\n[{name}] three blocks, each in a brand-new empty chat ...", flush=True)
    answers = []
    for i, b in enumerate(blocks, 1):
        t0 = time.time()
        a = ask([{"role": "system", "content": sp}, {"role": "user", "content": b}])
        answers.append(a)
        print(f"  block {i}: {int(time.time()-t0)}s, {len(a)} chars, "
              f"{len(blocks_of(a))} code block(s)", flush=True)
    open(os.path.join(runs, f"{stamp}-bank-{name}.md"), "w", encoding="utf-8").write(
        "\n\n---\n\n".join(answers))
    c, probe, out = grade(answers, name)
    open(os.path.join(WD, f"{name}_stdout.txt"), "w", encoding="utf-8").write(out)
    return report(f"{name.upper()} (3 fresh chats)", c)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "regrade":
        regrade(sys.argv[2])
        return
    clean, typo = load_bank()
    sp = sys_prompt()
    print(f"system prompt: {len(sp)} chars (~{len(sp)//4} tokens, incl. dataset block)")
    print(f"blocks: clean={len(clean)} typo={len(typo)}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = os.path.join(ROOT, "courses", "da-python", "evals", "runs")
    os.makedirs(runs, exist_ok=True)
    totals = []
    if mode in ("both", "clean"):
        totals.append(("CLEAN (paper as printed)",) + run_mode("clean", clean, sp, stamp, runs))
    if mode in ("both", "typo"):
        totals.append(("TYPO (as actually typed)",) + run_mode("typo", typo, sp, stamp, runs))
    print("\n" + "=" * 50)
    for name, n, tot in totals:
        print(f"  {name.ljust(28)} {n}/{tot}")
    print("=" * 50)


if __name__ == "__main__":
    main()
