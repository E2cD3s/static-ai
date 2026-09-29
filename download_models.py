"""Downloads local TTS models.  python download_models.py [kokoro] [piper]   (default: kokoro)
Whisper models download automatically on first run into models/whisper."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = {
    "kokoro": [
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
         "models/kokoro/kokoro-v1.0.onnx"),
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
         "models/kokoro/voices-v1.0.bin"),
    ],
    "piper": [
        ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
         "models/piper/en_US-lessac-medium.onnx"),
        ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
         "models/piper/en_US-lessac-medium.onnx.json"),
    ],
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok   {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  get  {dest.name}: {pct:3d}% of {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    tmp.replace(dest)
    print()


def main() -> None:
    wanted = sys.argv[1:] or ["kokoro"]
    for name in wanted:
        if name not in MODELS:
            sys.exit(f"Unknown model set '{name}'. Options: {', '.join(MODELS)}")
        print(f"[{name}]")
        for url, rel in MODELS[name]:
            download(url, ROOT / rel)


if __name__ == "__main__":
    main()
