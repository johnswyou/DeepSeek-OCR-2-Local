#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  "${VENV_DIR}/bin/pip" install --upgrade packaging psutil ninja
  "${VENV_DIR}/bin/pip" install flash-attn==2.7.3 --no-build-isolation
fi

cat <<'EOF'
Setup complete.

Activate the environment with:
  source .venv/bin/activate

Sanity check:
  python run_ocr.py --check

Run OCR on an image:
  python run_ocr.py /path/to/image.png

Run OCR on a PDF:
  python run_ocr.py /path/to/file.pdf

Notes:
  - First model load will download about 6.8 GB from Hugging Face.
  - On T4-class GPUs this launcher defaults to fp16 + eager attention.
  - Optional FlashAttention install:
      INSTALL_FLASH_ATTN=1 ./setup.sh
      (installs FlashAttention build prerequisites before compiling flash-attn)
      python run_ocr.py --attention flash_attention_2 /path/to/image.png
EOF
