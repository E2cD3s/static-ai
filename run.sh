#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# faster-whisper (CTranslate2) loads cuBLAS/cuDNN by soname; point it at the pip-installed nvidia libs.
NV_LIBS=$(python -c "import site, glob, os; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))")
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

exec python main.py "$@"
