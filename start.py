"""
Lab Rat AI - Launcher
Usage: python start.py
"""
import subprocess, sys, os, time, signal, json, threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT     = os.path.dirname(os.path.abspath(__file__))
SERVER   = os.path.join(ROOT, "llama-cpp", "llama-server.exe")
MODELS   = os.path.join(ROOT, "models")
UI_FILE  = os.path.join(ROOT, "ui", "index.html")
SESSIONS = os.path.join(ROOT, "sessions.json")
AI_PORT  = 11434
UI_PORT  = 8080

MODELS_PRIORITY = [
    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf",
    "Qwen3-14B-Q4_K_M.gguf",
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    "Qwen2.5-7B-Instruct-Q2_K.gguf",
]
CTX_FOR = {
    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf":  16384,
    "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf":  8192,
    "Qwen3.5-9B-Q4_K_M.gguf":                  8192,
    "Qwen3-14B-Q4_K_M.gguf":                   6144,
    "Qwen3.5-4B-Q4_K_M.gguf":                 16384,
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf":        4096,
}

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
            resp = urllib.request.urlopen(req, timeout=300)
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

        elif p == "/api/sessions":
            with sessions_lock:
                data = load_sessions()
            self.send_json(200, data)

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

def find_model():
    for name in MODELS_PRIORITY:
        path = os.path.join(MODELS, name)
        if os.path.exists(path):
            return name, path
    return None, None


def run():
    print()
    print("  =====================================================")
    print("   Lab Rat AI  |  CPU mode")
    print("  =====================================================")
    print()

    if not os.path.exists(SERVER):
        print("  ERROR: llama-server.exe not found in llama-cpp\\")
        input("  Press Enter to exit...")
        sys.exit(1)

    model_name, model_path = find_model()
    if not model_name:
        print("  ERROR: No model found in models\\")
        print("  Run: python download-model.py")
        input("  Press Enter to exit...")
        sys.exit(1)

    ctx = CTX_FOR.get(model_name, 8192)
    kill_port(AI_PORT)
    kill_port(UI_PORT)

    print(f"  Model:   {model_name}")
    print(f"  Context: {ctx} tokens  |  Threads: 8")
    print()
    print("  Starting AI server... (first load: 30-90 seconds)")
    print()

    procs = []

    # ── Start llama-server ──────────────────────────────────────────────────────
    ai_proc = subprocess.Popen(
        [SERVER, "--model", model_path, "--threads", "8",
         "--ctx-size", str(ctx), "--batch-size", "512", "--ubatch-size", "128",
         "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "--port", str(AI_PORT), "--host", "127.0.0.1", "--no-mmap"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(ai_proc)

    # ── Poll health ─────────────────────────────────────────────────────────────
    start = time.time()
    deadline = start + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{AI_PORT}/health", timeout=2)
            break
        except:
            print(f"\r  Loading... {int(time.time()-start)}s", end="", flush=True)
            time.sleep(3)
    else:
        print("\n\n  Timeout: AI server did not start.")
        for p in procs: p.terminate()
        input("  Press Enter to exit...")
        sys.exit(1)

    print(f"\r  AI server ready! ({int(time.time()-start)}s)          ")

    # ── Start UI HTTP server ────────────────────────────────────────────────────
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
