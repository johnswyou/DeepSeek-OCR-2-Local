# DeepSeek-OCR-2 local runner

This repository gives you a self-contained local setup for running `deepseek-ai/DeepSeek-OCR-2` on a Linux machine with a pre-Ampere NVIDIA GPU.

**NOTE**: I have only tested this repo on a machine running Ubuntu 22.04.5 LTS with an NVIDIA Tesla T4 GPU

Features of this repo:

- defaults to `float16` instead of the upstream `bfloat16`
- defaults to `eager` attention instead of requiring `flash-attn`
- supports both single images and PDFs
- pins the validated Hugging Face revision for reproducible remote code

## Quick start

```bash
cd /home/ubuntu/personal/Documents/projects/deepseek-ocr2-local
./setup.sh
source .venv/bin/activate
python run_ocr.py --check
```

First model load downloads about `6.8 GB` from Hugging Face.

## Usage

Run OCR on an image:

```bash
python run_ocr.py /path/to/image.png
```

Run OCR on a PDF:

```bash
python run_ocr.py /path/to/document.pdf
```

Use the simpler no-layout prompt:

```bash
python run_ocr.py --prompt-mode free-ocr /path/to/image.png
```

Override the output directory:

```bash
python run_ocr.py /path/to/image.png --output-dir /tmp/deepseek-ocr-output
```

## Output files

Image runs create:

- `result.md` and `result.mmd` with the OCR markdown
- `result.raw.txt` with the raw decoded text
- `result_with_boxes.jpg` when grounding boxes are emitted
- `images/` for cropped image regions

PDF runs create:

- `combined_result.md` with all pages combined
- `pages/page_0001/`, `pages/page_0002/`, etc. with per-page artifacts

## Optional FlashAttention

If you have an Ampere-or-newer GPU and want to try the official FlashAttention path:

```bash
INSTALL_FLASH_ATTN=1 ./setup.sh
python run_ocr.py --attention flash_attention_2 /path/to/image.png
```

FlashAttention requires an Ampere-or-newer GPU at runtime. On pre-Ampere GPUs such as the Tesla T4, `python run_ocr.py --attention flash_attention_2 ...` will fail with `RuntimeError: FlashAttention only supports Ampere GPUs or newer.`

**NOTE**: I have not tested the FlashAttention path.

`eager` is the working default.
