r"""
Lab Rat AI - Generic course launcher.

Usage:
    python start.py                  -> default course (dataviz-python)
    python start.py dataviz-r        -> any course folder under courses\
    python start.py --list           -> show available courses

A course is a folder under courses\<id>\ containing:
    course.json   - display name, disguise title, temperature, max_tokens,
                    optional "models" list to pin a per-course model order
    prompt.md     - the system prompt (the actual IP of this project)

Model registry lives in models.json (ctx + extra llama-server flags per model).
Everything runs offline from this folder: no pip, no admin, no internet.
"""
import subprocess, sys, os, time, signal, json, threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import dataset_scan

ROOT     = os.path.dirname(os.path.abspath(__file__))
SERVER   = os.path.join(ROOT, "llama-cpp", "llama-server.exe")
SERVER_LOG = os.path.join(ROOT, "llama-server.log")
MODELS   = os.path.join(ROOT, "models")
COURSES  = os.path.join(ROOT, "courses")
UI_FILE  = os.path.join(ROOT, "ui", "index.html")
AI_PORT  = 11434
UI_PORT  = 8080

DEFAULT_COURSE = "dataviz-python"

# ── Model registry ────────────────────────────────────────────────────────────
def load_registry():
    with open(os.path.join(ROOT, "models.json"), encoding="utf-8") as f:
        return json.load(f)

_REG = load_registry()
MODELS_PRIORITY = _REG["priority"]
# Kept for backward compat (start_powerbi.py uses base.CTX_FOR)
CTX_FOR = {name: m["ctx"] for name, m in _REG["models"].items()}
FLAGS_FOR = {name: m.get("flags", []) for name, m in _REG["models"].items()}

def default_threads():
    """Physical-core estimate: half the logical count, floor 4.

    Override with the LABRAT_THREADS env var. This matters on Intel 12th-gen
    and later, which are HYBRID: P-cores have hyper-threading, E-cores do not,
    so "logical // 2" is only an approximation of the physical core count.
    llama.cpp runs a barrier per layer, so every thread waits on the slowest
    one - putting threads on E-cores can cost more than it adds.

    Measured on the dev box (llama-bench, gemma-4-E4B): decode is
    memory-bandwidth-bound and barely moves with thread count (11.3 tok/s at 16
    threads vs 11.9 at 6), while prefill is compute-bound and nearly halves
    (252 -> 138 tok/s). So if a lab PC feels slow, try setting this to the
    number of P-cores and compare - a LOWER number is often faster.

        set LABRAT_THREADS=6     (cmd, before running launch-da.bat)
    """
    env = os.environ.get("LABRAT_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    n = os.cpu_count() or 8
    return max(4, n // 2)

THREADS = default_threads()

# ── Course config ─────────────────────────────────────────────────────────────
def list_courses():
    out = []
    if os.path.isdir(COURSES):
        for d in sorted(os.listdir(COURSES)):
            if not d.startswith("_") and os.path.isfile(os.path.join(COURSES, d, "course.json")):
                out.append(d)
    return out

def load_course(course_id):
    cdir = os.path.join(COURSES, course_id)
    with open(os.path.join(cdir, "course.json"), encoding="utf-8") as f:
        course = json.load(f)
    with open(os.path.join(cdir, "prompt.md"), encoding="utf-8") as f:
        course["sysPrompt"] = f.read().strip()
    return course

COURSE = None  # set in run(); UIHandler serves it at /api/config

# ── Sessions (per course) ─────────────────────────────────────────────────────
SESSIONS = os.path.join(ROOT, "sessions.json")  # overridden per course in run()
sessions_lock = threading.Lock()

def load_sessions():
    try:
        if os.path.exists(SESSIONS):
            with open(SESSIONS, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return []

def save_sessions(data):
    with open(SESSIONS, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def proxy(self, method):
        """Forward /v1/* and /health to llama-server, streaming the response."""
        upstream = f"http://127.0.0.1:{AI_PORT}{self.path}"
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        headers = {}
        ct = self.headers.get("Content-Type", "")
        if ct:
            headers["Content-Type"] = ct
        try:
            req = urllib.request.Request(upstream, data=body, headers=headers, method=method)
            # Generous timeout. The UI streams, so this bounds the gap BETWEEN
            # chunks rather than the whole generation - but a non-streaming
            # client (or a very long answer on a slow CPU) can otherwise 502
            # mid-question, which would be indistinguishable from a crash to a
            # student sitting an exam.
            resp = urllib.request.urlopen(req, timeout=1800)
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            try:
                self.send_json(502, {"error": "upstream error"})
            except:
                pass

    def do_GET(self):
        p = urlparse(self.path).path  # strip query string for routing
        if p in ("/", "/index.html"):
            try:
                with open(UI_FILE, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(body))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        elif p == "/api/config":
            self.send_json(200, COURSE or {})

        elif p == "/api/sessions":
            with sessions_lock:
                data = load_sessions()
            self.send_json(200, data)

        elif p == "/api/datasets":
            # Names of the CSVs vendored in data\ , for the UI dropdown.
            self.send_json(200, {"builtin": dataset_scan.list_builtin(ROOT)})

        elif p == "/api/scan":
            # Scan a CSV and return the schema block the UI appends to the
            # system prompt. Accepts an absolute path, a path relative to this
            # folder, or a bare built-in name - so one control covers both a
            # teacher-supplied file and a vendored dataset.
            spec = (parse_qs(urlparse(self.path).query).get("spec") or [""])[0]
            path = dataset_scan.resolve(spec, ROOT)
            if not path:
                self.send_json(200, {"ok": False,
                                     "error": f"could not find '{spec}'"})
            else:
                self.send_json(200, dataset_scan.scan(path))

        elif p.startswith("/v1/") or p == "/health":
            self.proxy("GET")

        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/v1/") or self.path == "/health":
            self.proxy("POST")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if self.path == "/api/sessions":
            try:
                session = json.loads(body)
                with sessions_lock:
                    data = load_sessions()
                    # Update existing session or append new one
                    idx = next((i for i, s in enumerate(data) if s.get("id") == session.get("id")), -1)
                    if idx >= 0:
                        data[idx] = session
                    else:
                        data.insert(0, session)
                    # Keep max 50 sessions
                    data = data[:50]
                    save_sessions(data)
                self.send_json(200, {"ok": True})
            except Exception as e:
                self.send_json(400, {"error": str(e)})
        else:
            self.send_json(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.startswith("/api/sessions/"):
            sid = self.path[len("/api/sessions/"):]
            with sessions_lock:
                data = load_sessions()
                data = [s for s in data if s.get("id") != sid]
                save_sessions(data)
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "not found"})


def kill_port(port):
    try:
        out = subprocess.check_output(['netstat', '-aon'], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if f':{port} ' in line and 'LISTENING' in line:
                pid = line.split()[-1]
                subprocess.run(['taskkill', '/F', '/PID', pid],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

def find_model(preferred=None):
    """preferred: optional per-course filename list tried before the global priority."""
    for name in (preferred or []) + MODELS_PRIORITY:
        path = os.path.join(MODELS, name)
        if os.path.exists(path):
            return name, path
    return None, None

def start_llama_server(model_name, model_path, ctx=None):
    """Boot llama-server with registry flags for this model. Returns the process."""
    ctx = ctx or CTX_FOR.get(model_name, 8192)
    cmd = [SERVER, "--model", model_path, "--threads", str(THREADS),
           "--ctx-size", str(ctx), "--batch-size", "512", "--ubatch-size", "128",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           # ONE slot. llama-server defaults to auto (4 here) and rotates
           # requests across slots; each new slot re-prefills the whole system
           # prompt from cold, which is ~12k tokens for da-python and costs
           # minutes on a CPU. One user needs one slot: the prefix cache is
           # then always warm after the first question, and the single
           # conversation gets the full context instead of a share of it.
           "--parallel", "1",
           "--port", str(AI_PORT), "--host", "127.0.0.1", "--no-mmap"]
    cmd += FLAGS_FOR.get(model_name, [])
    # Keep the server's own output. It used to go to DEVNULL, which made every
    # startup failure look identical: a bare "Timeout" with no cause. The one
    # that actually happened was "couldn't bind HTTP server socket" because
    # another program already owned the port.
    # Fall back to DEVNULL if the folder is not writable (running straight off
    # a read-only pendrive). Losing the log must never cost us the launcher.
    try:
        log = open(SERVER_LOG, "w", encoding="utf-8", errors="replace")
    except OSError:
        log = subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

def wait_healthy(timeout=180):
    """Poll llama-server /health. Returns seconds waited, or None on timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{AI_PORT}/health", timeout=2)
            return int(time.time() - start)
        except:
            print(f"\r  Loading... {int(time.time()-start)}s", end="", flush=True)
            time.sleep(3)
    return None


def run():
    global COURSE, SESSIONS

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        print("Available courses:")
        for c in list_courses():
            print(f"  {c}")
        return

    course_id = args[0] if args else DEFAULT_COURSE
    # A stripped deployment (e.g. the USB copy, which ships only da-python) may
    # not contain DEFAULT_COURSE. If the default is absent but exactly one
    # course is installed, use it rather than erroring on a bare "start.py".
    if not args and not os.path.isdir(os.path.join(COURSES, course_id)):
        installed = list_courses()
        if len(installed) == 1:
            course_id = installed[0]
    try:
        COURSE = load_course(course_id)
    except FileNotFoundError:
        print(f"  ERROR: course '{course_id}' not found under courses\\")
        print(f"  Available: {', '.join(list_courses()) or '(none)'}")
        input("  Press Enter to exit...")
        sys.exit(1)

    SESSIONS = os.path.join(ROOT, f"sessions-{course_id}.json")

    print()
    print("  =====================================================")
    print(f"   Lab Rat AI  |  {COURSE.get('name', course_id)}  |  CPU mode")
    print("  =====================================================")
    print()

    if not os.path.exists(SERVER):
        print("  ERROR: llama-server.exe not found in llama-cpp\\")
        input("  Press Enter to exit...")
        sys.exit(1)

    model_name, model_path = find_model(COURSE.get("models"))
    if not model_name:
        print("  ERROR: No model found in models\\")
        print("  Run: python download-model.py")
        input("  Press Enter to exit...")
        sys.exit(1)

    ctx = CTX_FOR.get(model_name, 8192)
    COURSE["model"] = model_name
    kill_port(AI_PORT)
    kill_port(UI_PORT)

    print(f"  Course:  {course_id}")
    print(f"  Model:   {model_name}")
    print(f"  Context: {ctx} tokens  |  Threads: {THREADS}")
    print()
    print("  Starting AI server... (first load: 30-90 seconds)")
    print()

    procs = [start_llama_server(model_name, model_path, ctx)]

    waited = wait_healthy()
    if waited is None:
        print("\n\n  Timeout: AI server did not start.")
        # Show the reason instead of making the student guess.
        try:
            tail = [l for l in open(SERVER_LOG, encoding="utf-8",
                                    errors="replace").read().splitlines() if l.strip()][-6:]
            if tail:
                print("  Last lines from the AI server:\n")
                for l in tail:
                    print("    " + l[:150])
                print(f"\n  Full log: {SERVER_LOG}")
        except Exception:
            pass
        for p in procs: p.terminate()
        input("  Press Enter to exit...")
        sys.exit(1)

    print(f"\r  AI server ready! ({waited}s)          ")

    ui_server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), UIHandler)
    ui_thread = threading.Thread(target=ui_server.serve_forever, daemon=True)
    ui_thread.start()

    url = f"http://localhost:{UI_PORT}/?v={int(time.time())}"
    print()
    print("  =====================================================")
    print(f"   Chat UI:  {url}")
    print()
    print("   Open the URL above in Chrome/Edge.")
    print("   Ctrl+C to stop.")
    print("  =====================================================")
    print()

    def shutdown(sig=None, frame=None):
        print("\n  Shutting down...")
        ui_server.shutdown()
        for p in procs:
            try: p.terminate()
            except: pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(5)

if __name__ == "__main__":
    run()
