r"""Scan a CSV and produce a COMPACT schema block for the system prompt.

Why this exists
---------------
In the July 2026 Class Test the model wrote

    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

because the question said "mapping 'Male': 0 and 'Female': 1". The file
actually contains MALE and FEMALE, so .map() returned NaN for every row and
silently destroyed the column. It also hand-typed 'island_Boscoe' and
'island_Targersen' for one-hot columns whose real values are Biscoe, Dream and
Torgersen. Both are unfixable by prompt-writing: the model cannot know what is
in a file it has never seen.

So we read the file and tell it. Note the split:

  * SCANNING is Python and reads every row - it costs zero context.
  * INJECTING is tokens, and is capped hard (~250) regardless of file size.

Reading only the first few lines would be cheaper still but would miss exactly
what matters: in penguins.csv the first three data rows are all Torgersen (so
Biscoe/Dream never appear) and the first NaN is on row four.

Stdlib only - this runs on the exam PC, where nothing may be installed.
"""
import csv, io, os

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?"}
MAX_DISTINCT = 12      # list values only for genuinely categorical columns
MAX_ROWS     = 500_000 # guard against a pathological file; exam CSVs are tiny
VALUE_CHARS  = 90      # truncate a very long distinct-value list


def _is_missing(v):
    return v.strip().lower() in NA_TOKENS


def _num(v):
    """Return float(v) or None. Accepts 1, 1.0, 1e3, -2.5."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(x):
    """Compact number: 172.0 -> 172, 39.10 -> 39.1"""
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def scan(path, max_distinct=MAX_DISTINCT):
    """Read a CSV and return {ok, name, n_rows, columns[], text} .

    Never raises: on any failure returns {"ok": False, "error": "..."} so a bad
    path can never take the launcher down mid-exam.
    """
    try:
        return _scan(path, max_distinct)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "path": path}


def _scan(path, max_distinct):
    if not os.path.isfile(path):
        return {"ok": False, "error": "file not found", "path": path}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(64 * 1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        try:
            header = next(reader)
        except StopIteration:
            return {"ok": False, "error": "file is empty", "path": path}

        header = [h.strip() for h in header]
        n = len(header)
        missing = [0] * n
        distinct = [set() for _ in range(n)]
        over = [False] * n          # blew past max_distinct, stop collecting
        numeric = [True] * n        # all non-missing values parse as a number
        allint = [True] * n
        lo = [None] * n
        hi = [None] * n
        n_rows = 0
        truncated = False

        for row in reader:
            if n_rows >= MAX_ROWS:
                truncated = True
                break
            n_rows += 1
            for i in range(n):
                v = row[i] if i < len(row) else ""
                if _is_missing(v):
                    missing[i] += 1
                    continue
                v = v.strip()
                x = _num(v)
                if x is None:
                    numeric[i] = False
                    allint[i] = False
                else:
                    if lo[i] is None or x < lo[i]:
                        lo[i] = x
                    if hi[i] is None or x > hi[i]:
                        hi[i] = x
                    if allint[i] and x != int(x):
                        allint[i] = False
                if not over[i]:
                    distinct[i].add(v)
                    if len(distinct[i]) > max_distinct:
                        over[i] = True
                        distinct[i] = set()

    cols = []
    for i, name in enumerate(header):
        c = {"name": name, "missing": missing[i]}
        if numeric[i] and lo[i] is not None:
            c["kind"] = "int" if allint[i] else "float"
            c["min"], c["max"] = lo[i], hi[i]
        else:
            c["kind"] = "text"
        if not over[i] and distinct[i]:
            c["values"] = sorted(distinct[i])
        cols.append(c)

    return {
        "ok": True,
        "path": path,
        "name": os.path.basename(path),
        "n_rows": n_rows,
        "n_cols": n,
        "truncated": truncated,
        "columns": cols,
        "text": render(os.path.basename(path), n_rows, n, cols, truncated),
    }


def render(name, n_rows, n_cols, cols, truncated=False):
    """Format the block that gets appended to the system prompt."""
    w = max((len(c["name"]) for c in cols), default=10)
    w = min(w, 28)
    lines = [
        "=== DATASET IN USE (scanned from the real file) ===",
        "These are the ACTUAL column names and values. They override anything",
        "written in the question - questions are typed in a hurry and often",
        "misspell names or guess at category spelling/capitalisation.",
        "",
        f"FILE: {name}   {n_rows} rows x {n_cols} columns"
        + ("   [first %d rows scanned]" % n_rows if truncated else ""),
    ]
    for c in cols:
        kind = {"int": "number", "float": "number", "text": "text"}[c["kind"]]
        miss = f"{c['missing']} missing" if c["missing"] else "complete"
        detail = ""
        if "values" in c:
            vals = " | ".join(c["values"])
            if len(vals) > VALUE_CHARS:
                vals = vals[:VALUE_CHARS].rsplit(" |", 1)[0] + " | ..."
            detail = f"{len(c['values'])} distinct: {vals}"
        elif c["kind"] in ("int", "float"):
            detail = f"range {_fmt(c['min'])} .. {_fmt(c['max'])}"
        lines.append(f"  {c['name'].ljust(w)}  {kind:6s}  {miss:>12s}   {detail}")
    lines += [
        "",
        f"Load it with:  df = pd.read_csv('{name}')",
        "Use these names verbatim. If the question names a column that is not",
        "in this list, use the closest match here and say so in one line.",
        "Never invent a category value or a one-hot column name - read them",
        "off this block, or derive them from the data.",
    ]
    return "\n".join(lines)


def resolve(spec, root):
    """Turn a user-supplied dataset spec into a path.

    Accepts an absolute path, a path relative to the launcher folder, or a bare
    dataset name ('penguins', 'penguins.csv') resolved against data\\ - so the
    same control works for a teacher-supplied CSV and for a built-in dataset.
    """
    spec = (spec or "").strip().strip('"').strip("'")
    if not spec:
        return None
    if os.path.isabs(spec) and os.path.isfile(spec):
        return spec
    cands = [
        os.path.join(root, spec),
        os.path.join(root, "data", spec),
        os.path.join(root, "data", spec + ".csv"),
        spec,
    ]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def list_builtin(root):
    """Dataset names shipped in data\\ , for the UI dropdown."""
    d = os.path.join(root, "data")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d) if f.lower().endswith(".csv"))


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) < 2:
        print("usage: python dataset_scan.py <csv path or dataset name>")
        print("builtin:", ", ".join(list_builtin(root)))
        sys.exit(0)
    p = resolve(sys.argv[1], root)
    if not p:
        print("could not resolve:", sys.argv[1])
        sys.exit(1)
    r = scan(p)
    if not r["ok"]:
        print("ERROR:", r["error"])
        sys.exit(1)
    print(r["text"])
    print()
    print(f"[block is {len(r['text'])} chars ~ {len(r['text'])//4} tokens]")
