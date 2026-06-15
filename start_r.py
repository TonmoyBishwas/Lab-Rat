"""
Lab Rat AI - R programming helper launcher.
Usage: python start_r.py

Same llama-server + proxy stack as start.py, but serves ui/r_lang.html
with an R / tidyverse / ggplot2 system prompt instead of the default
Python data-viz prompt.
"""
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

import start as base  # reuse find_model, kill_port, UIHandler, CTX_FOR, AI_PORT, UI_PORT

ROOT    = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(ROOT, "ui", "r_lang.html")
AI_PORT = base.AI_PORT
UI_PORT = base.UI_PORT


class RHandler(base.UIHandler):
    def do_GET(self):
        p = urlparse(self.path).path
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
            return
        super().do_GET()


def run():
    print()
    print("  =====================================================")
    print("   Lab Rat AI  |  R programming helper  |  CPU mode")
    print("  =====================================================")
    print()

    if not os.path.exists(base.SERVER):
        print("  ERROR: llama-server.exe not found in llama-cpp\\")
        input("  Press Enter to exit...")
        sys.exit(1)

    model_name, model_path = base.find_model()
    if not model_name:
        print("  ERROR: No model found in models\\")
        print("  Run: python download-model.py")
        input("  Press Enter to exit...")
        sys.exit(1)

    ctx = base.CTX_FOR.get(model_name, 8192)
    base.kill_port(AI_PORT)
    base.kill_port(UI_PORT)

    print(f"  Model:   {model_name}")
    print(f"  Context: {ctx} tokens  |  Threads: 8")
    print()
    print("  Starting AI server... (first load: 30-90 seconds)")
    print()

    procs = []
    ai_proc = subprocess.Popen(
        [base.SERVER, "--model", model_path, "--threads", "8",
         "--ctx-size", str(ctx), "--batch-size", "512", "--ubatch-size", "128",
         "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "--port", str(AI_PORT), "--host", "127.0.0.1", "--no-mmap"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(ai_proc)

    start = time.time()
    deadline = start + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{AI_PORT}/health", timeout=2)
            break
        except Exception:
            print(f"\r  Loading... {int(time.time() - start)}s", end="", flush=True)
            time.sleep(3)
    else:
        print("\n\n  Timeout: AI server did not start.")
        for p in procs:
            p.terminate()
        input("  Press Enter to exit...")
        sys.exit(1)

    print(f"\r  AI server ready! ({int(time.time() - start)}s)          ")

    ui_server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), RHandler)
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
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()
