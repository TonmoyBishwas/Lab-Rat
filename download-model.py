"""
Lab Rat AI - Model Downloader
Run with: python download-model.py
"""
import urllib.request
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

MODELS = [
    {
        "name": "Qwen2.5-Coder-7B Instruct Q4_K_M  (~4.7 GB)  RECOMMENDED  |  no-think, fast, code-specialized",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "launcher": "python start.py",
    },
    {
        "name": "Qwen2.5-Coder-14B Instruct Q4_K_M (~8.9 GB)  BETTER QUALITY  |  no-think, slower, strongest code",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "launcher": "python start.py",
    },
    {
        "name": "Qwen3.5-9B Q4_K_M (~5.7 GB)  THINKING MODEL  |  slower, reasons through ambiguity",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
        "launcher": "python start.py",
    },
]

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        done = int(pct / 2)
        bar = "#" * done + "-" * (50 - done)
        gb_done = downloaded / 1e9
        gb_total = total_size / 1e9
        print(f"\r  [{bar}] {pct:.1f}%  {gb_done:.2f}/{gb_total:.2f} GB", end="", flush=True)
    else:
        mb = downloaded / 1e6
        print(f"\r  Downloaded {mb:.1f} MB...", end="", flush=True)

def main():
    print()
    print("  =====================================================")
    print("   Lab Rat AI  |  Model Downloader")
    print("  =====================================================")
    print()
    for i, m in enumerate(MODELS, 1):
        print(f"  [{i}] {m['name']}")
        print()

    while True:
        choice = input("  Enter choice (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            model = MODELS[int(choice) - 1]
            break
        print("  Please enter 1, 2, or 3.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, model["filename"])

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"\n  File already exists ({size_mb:.0f} MB): {model['filename']}")
        overwrite = input("  Delete and re-download? (y/n): ").strip().lower()
        if overwrite == "y":
            os.remove(dest)
            print("  Deleted.")
        else:
            print(f"\n  Keeping existing file. Run {model['launcher']} to start.")
            input("\n  Press Enter to exit...")
            return

    print(f"\n  Downloading: {model['filename']}")
    print(f"  To: {dest}")
    print()

    try:
        req = urllib.request.Request(model["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 1024  # 1 MB chunks

            with open(dest, "wb") as f:
                block_num = 0
                while True:
                    chunk = response.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    block_num += 1
                    show_progress(block_num, block, total)

        print()

        # Verify GGUF magic
        with open(dest, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            print(f"\n  ERROR: File is not a valid GGUF model (got: {magic}).")
            print("  The URL may have returned an error page.")
            os.remove(dest)
            input("\n  Press Enter to exit...")
            return

        size_gb = os.path.getsize(dest) / 1e9
        print()
        print("  =====================================================")
        print(f"   Download complete!  ({size_gb:.2f} GB)")
        print(f"   File: {dest}")
        print()
        print(f"   Run {model['launcher']} to start the AI.")
        print("  =====================================================")

    except Exception as e:
        print(f"\n\n  ERROR: {e}")
        if os.path.exists(dest):
            os.remove(dest)

    input("\n  Press Enter to exit...")

if __name__ == "__main__":
    main()
