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
        "name": "Gemma 4 E4B-it Q4_K_M  (~5.0 GB)  SAFE DEFAULT  |  Apr 2026, dense, eval-validated 25/25 on DV Lab banks",
        "filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
        "launcher": "python start.py",
    },
    {
        "name": "Mellum 2 12B-A2.5B Instruct MXFP4_MOE (~7.0 GB)  FASTEST  |  Jun 2026 JetBrains MoE, 2.5B active. Measured 20.4 tok/s vs Gemma's 11.3. NEEDS llama.cpp >= b9482. Weak instruction-following - follows exam decoys. See courses/da-python/evals/findings.md.",
        "filename": "Mellum2-12B-A2.5B-Instruct-MXFP4_MOE.gguf",
        "url": "https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-MXFP4_MOE/resolve/main/Mellum2-12B-A2.5B-Instruct-MXFP4_MOE.gguf",
        "launcher": "python start.py",
    },
    {
        "name": "Granite 4.1 8B Q4_K_M (~5.4 GB)  NOT RECOMMENDED  |  IBM Apr 2026, best published HumanEval of anything that fits, but measured only 4.9 tok/s warm on a fast desktop - too slow for the exam laptop. Also failed the ordinal-encoding decoy.",
        "filename": "granite-4.1-8b-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/granite-4.1-8b-GGUF/resolve/main/granite-4.1-8b-Q4_K_M.gguf",
        "launcher": "python start.py",
    },
    {
        "name": "Qwen3.5-9B Q4_K_M (~5.7 GB)  REASONING CANDIDATE  |  Mar 2026, thinking disabled via llama-server flag, slower (dense 9B)",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
        "launcher": "python start.py",
    },
    {
        "name": "Qwen2.5-Coder-7B Instruct Q4_K_M (~4.7 GB)  LEGACY FALLBACK  |  Nov 2024 code specialist",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
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

    valid = [str(i) for i in range(1, len(MODELS) + 1)]
    while True:
        choice = input(f"  Enter choice ({'/'.join(valid)}): ").strip()
        if choice in valid:
            model = MODELS[int(choice) - 1]
            break
        print(f"  Please enter one of: {', '.join(valid)}.")

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
