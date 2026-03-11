#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
MODEL_REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
DOC_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
FREE_OCR_PROMPT = "<image>\nFree OCR. "
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR-2 locally with T4-friendly defaults."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Image or PDF to process. Omit with --check to only validate the setup.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for OCR outputs. Defaults to ./outputs/<input-stem>/",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Hugging Face model id (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="Pinned Hugging Face revision for reproducible remote code and weights.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("document", "free-ocr"),
        default="document",
        help="Built-in prompt to use when --prompt is not provided.",
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt. If it does not contain <image>, it will be prefixed automatically.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda"),
        default="auto",
        help="Execution device. CPU is not supported by the upstream model code.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype. Auto picks bfloat16 on Ampere+ GPUs and float16 on older GPUs.",
    )
    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
        help="Attention backend. Auto prefers FlashAttention on newer GPUs when installed, otherwise eager.",
    )
    parser.add_argument("--base-size", type=int, default=1024, help="Global image size.")
    parser.add_argument("--image-size", type=int, default=768, help="Patch image size.")
    parser.add_argument(
        "--disable-crop-mode",
        action="store_true",
        help="Disable the upstream dynamic crop path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="Render DPI for PDF pages.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        help="Optional limit on how many PDF pages to process.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streamed token output during generation.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the tokenizer and model, print resolved settings, and exit if no input is given.",
    )
    return parser


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA was requested but no NVIDIA GPU is available.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    fail(
        "DeepSeek-OCR-2's upstream custom code assumes CUDA. "
        "No supported NVIDIA GPU was detected on this machine."
    )


def get_cuda_capability() -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def supports_bfloat16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = get_cuda_capability()
    if major < 8:
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        return bool(torch.cuda.is_bf16_supported())
    return True


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        if device.type != "cuda" or not supports_bfloat16():
            fail("bfloat16 was requested, but the active GPU does not support it.")
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32

    if device.type == "cuda" and supports_bfloat16():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def flash_attn_installed() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_attention(attention_arg: str, device: torch.device) -> str:
    if attention_arg != "auto":
        if attention_arg == "flash_attention_2" and not flash_attn_installed():
            fail(
                "flash_attention_2 was requested but flash-attn is not installed. "
                "Re-run setup with INSTALL_FLASH_ATTN=1 ./setup.sh or choose --attention sdpa."
            )
        return attention_arg

    if device.type != "cuda":
        return "eager"

    major, _minor = get_cuda_capability()
    if major >= 8 and flash_attn_installed():
        return "flash_attention_2"
    return "eager"


def output_dir_for(args: argparse.Namespace, input_path: Path | None) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if input_path is None:
        return Path.cwd() / "outputs" / "check"
    return Path.cwd() / "outputs" / input_path.stem


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        if "<image>" in args.prompt:
            return args.prompt
        return f"<image>\n{args.prompt.strip()} "
    if args.prompt_mode == "free-ocr":
        return FREE_OCR_PROMPT
    return DOC_PROMPT


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def import_remote_helpers(model: Any) -> Any:
    module = importlib.import_module(type(model).__module__)
    return module


def load_model_and_tokenizer(
    model_name: str,
    revision: str,
    device: torch.device,
    dtype: torch.dtype,
    attention: str,
    attention_was_auto: bool,
) -> tuple[Any, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=revision,
    )

    attempts = [attention]
    if attention_was_auto and attention == "flash_attention_2":
        attempts.extend(["sdpa", "eager"])
    elif attention_was_auto and attention == "sdpa":
        attempts.append("eager")

    last_error: Exception | None = None
    for attn_impl in attempts:
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                revision=revision,
                use_safetensors=True,
                torch_dtype=dtype,
                _attn_implementation=attn_impl,
            )
            model = model.eval().to(device=device, dtype=dtype)
            return model, tokenizer, attn_impl
        except Exception as exc:  # explicit fallback path
            last_error = exc
            print(
                f"Model load failed with attention={attn_impl}: {exc}",
                file=sys.stderr,
            )
            if not attention_was_auto:
                raise

    assert last_error is not None
    raise last_error


def save_markdown_outputs(output_dir: Path, cleaned_text: str, raw_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.raw.txt").write_text(raw_text + "\n", encoding="utf-8")
    (output_dir / "result.md").write_text(cleaned_text + "\n", encoding="utf-8")
    (output_dir / "result.mmd").write_text(cleaned_text + "\n", encoding="utf-8")


def clean_generated_text(text: str) -> str:
    stop_str = "<｜end▁of▁sentence｜>"
    if text.endswith(stop_str):
        text = text[: -len(stop_str)]
    return text.strip()


def rewrite_embedded_image_paths(markdown_text: str, page_dir_name: str) -> str:
    return markdown_text.replace("![](images/", f"![]({page_dir_name}/images/")


def postprocess_and_save(
    helpers: Any,
    image_draw: Image.Image,
    output_dir: Path,
    decoded_text: str,
) -> str:
    cleaned_text = clean_generated_text(decoded_text)
    processed_text = cleaned_text

    matches_ref, matches_images, matches_other = helpers.re_match(cleaned_text)
    if matches_ref:
        result_image = helpers.process_image_with_refs(image_draw, matches_ref, str(output_dir))
        result_image.save(output_dir / "result_with_boxes.jpg")

    for idx, image_match in enumerate(matches_images):
        processed_text = processed_text.replace(image_match, f"![](images/{idx}.jpg)\n")

    for other_match in matches_other:
        processed_text = (
            processed_text.replace(other_match, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
        )

    save_markdown_outputs(output_dir, processed_text, cleaned_text)
    return processed_text


def run_inference(
    model: Any,
    tokenizer: Any,
    helpers: Any,
    prompt: str,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    max_new_tokens: int,
    stream: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [str(image_path)],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    formatted_prompt = helpers.format_messages(
        conversations=conversation, sft_format="plain", system_prompt=""
    )

    patch_size = 16
    downsample_ratio = 4
    images = helpers.load_pil_images(conversation)
    if not images:
        fail(f"Failed to load image: {image_path}")

    image_draw = images[0].copy()
    width, height = image_draw.size
    ratio = 1 - ((max(width, height) - min(width, height)) / max(width, height))

    image_transform = helpers.BasicImageTransform(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        normalize=True,
    )

    image_token = "<image>"
    image_token_id = 128815
    text_splits = formatted_prompt.split(image_token)

    images_list: list[torch.Tensor] = []
    images_crop_list: list[torch.Tensor] = []
    images_seq_mask: list[bool] = []
    tokenized_str: list[int] = []
    images_spatial_crop: list[list[int]] = []

    for text_sep, image in zip(text_splits, images):
        tokenized_sep = helpers.text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:
            if image.size[0] <= 768 and image.size[1] <= 768:
                crop_ratio = [1, 1]
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = helpers.dynamic_preprocess(image)

            global_view = helpers.ImageOps.pad(
                image,
                (base_size, base_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )

            images_list.append(image_transform(global_view).to(dtype))
            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_image in images_crop_raw:
                    images_crop_list.append(image_transform(crop_image).to(dtype))

            num_queries = (image_size // patch_size + downsample_ratio - 1) // downsample_ratio
            num_queries_base = (base_size // patch_size + downsample_ratio - 1) // downsample_ratio

            tokenized_image = ([image_token_id] * num_queries_base) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([image_token_id] * (num_queries * width_crop_num)) * (
                    num_queries * height_crop_num
                )

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
        else:
            if image_size <= 768:
                image = image.resize((image_size, image_size))

            global_view = helpers.ImageOps.pad(
                image,
                (image_size, image_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )
            images_list.append(image_transform(global_view).to(dtype))
            images_spatial_crop.append([1, 1])

            num_queries = (image_size // patch_size + downsample_ratio - 1) // downsample_ratio
            tokenized_image = ([image_token_id] * num_queries) * num_queries
            tokenized_image += [image_token_id]
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = helpers.text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    images_seq_mask_tensor = torch.tensor(images_seq_mask, dtype=torch.bool).unsqueeze(0).to(device)

    if not images_list:
        fail("No image tensors were created for inference.")

    images_ori = torch.stack(images_list, dim=0).to(device=device, dtype=dtype)
    images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long).to(device)
    if images_crop_list:
        images_crop = torch.stack(images_crop_list, dim=0).to(device=device, dtype=dtype)
    else:
        images_crop = torch.zeros((1, 3, base_size, base_size), dtype=dtype, device=device)

    streamer = None
    if stream:
        streamer = helpers.NoEOSTextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None) or getattr(model.config, "eos_token_id", None) or 1
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    with autocast_context:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[(images_crop, images_ori)],
                images_seq_mask=images_seq_mask_tensor,
                images_spatial_crop=images_spatial_crop_tensor,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=20,
                use_cache=True,
                streamer=streamer,
            )

    decoded_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=False)
    processed_text = postprocess_and_save(helpers, image_draw, output_dir, decoded_text)

    print("\nSaved outputs to:", output_dir)
    return processed_text


def render_pdf_pages(pdf_path: Path, render_dir: Path, dpi: int, page_limit: int | None) -> list[Path]:
    render_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            if page_limit is not None and index >= page_limit:
                break
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            page_path = render_dir / f"page_{index + 1:04d}.png"
            pixmap.save(page_path)
            pages.append(page_path)
    return pages


def process_pdf(
    model: Any,
    tokenizer: Any,
    helpers: Any,
    prompt: str,
    pdf_path: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    max_new_tokens: int,
    stream: bool,
    pdf_dpi: int,
    page_limit: int | None,
) -> None:
    pages_root = output_dir / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="deepseek-ocr2-pages-") as temp_dir:
        page_images = render_pdf_pages(pdf_path, Path(temp_dir), pdf_dpi, page_limit)
        if not page_images:
            fail(f"No PDF pages were rendered from {pdf_path}")

        combined_chunks: list[str] = []
        for page_index, page_image in enumerate(page_images, start=1):
            page_dir_name = f"page_{page_index:04d}"
            page_output_dir = pages_root / page_dir_name
            print(f"\n=== Processing page {page_index}/{len(page_images)} ===")
            page_text = run_inference(
                model=model,
                tokenizer=tokenizer,
                helpers=helpers,
                prompt=prompt,
                image_path=page_image,
                output_dir=page_output_dir,
                device=device,
                dtype=dtype,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                max_new_tokens=max_new_tokens,
                stream=stream,
            )
            page_text = rewrite_embedded_image_paths(page_text, f"pages/{page_dir_name}")
            combined_chunks.append(f"# Page {page_index}\n\n{page_text}\n")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    combined_markdown = "\n".join(combined_chunks).strip() + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined_result.md").write_text(combined_markdown, encoding="utf-8")
    print("\nSaved combined PDF markdown to:", output_dir / "combined_result.md")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve() if args.input_path else None
    if input_path is None and not args.check:
        parser.error("input_path is required unless --check is used.")
    if input_path is not None and not input_path.exists():
        fail(f"Input path does not exist: {input_path}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    requested_attention = resolve_attention(args.attention, device)
    output_dir = output_dir_for(args, input_path)
    prompt = resolve_prompt(args)

    if device.type == "cuda":
        major, minor = get_cuda_capability()
        device_summary = (
            f"{torch.cuda.get_device_name(0)} "
            f"(compute capability {major}.{minor})"
        )
    else:
        device_summary = str(device)

    print("Model:", args.model_name)
    print("Revision:", args.revision)
    print("Device:", device_summary)
    print("DType:", dtype_name(dtype))
    print("Requested attention:", requested_attention)
    print("Output directory:", output_dir)

    model, tokenizer, loaded_attention = load_model_and_tokenizer(
        model_name=args.model_name,
        revision=args.revision,
        device=device,
        dtype=dtype,
        attention=requested_attention,
        attention_was_auto=(args.attention == "auto"),
    )
    helpers = import_remote_helpers(model)

    print("Loaded attention:", loaded_attention)
    print("Tokenizer and model loaded successfully.")

    if args.check and input_path is None:
        return 0

    assert input_path is not None
    is_pdf = input_path.suffix.lower() == ".pdf"
    is_image = input_path.suffix.lower() in IMAGE_SUFFIXES
    if not (is_pdf or is_image):
        fail("Only image files and PDFs are supported.")

    if is_pdf:
        process_pdf(
            model=model,
            tokenizer=tokenizer,
            helpers=helpers,
            prompt=prompt,
            pdf_path=input_path,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=not args.disable_crop_mode,
            max_new_tokens=args.max_new_tokens,
            stream=not args.no_stream,
            pdf_dpi=args.pdf_dpi,
            page_limit=args.page_limit,
        )
    else:
        run_inference(
            model=model,
            tokenizer=tokenizer,
            helpers=helpers,
            prompt=prompt,
            image_path=input_path,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=not args.disable_crop_mode,
            max_new_tokens=args.max_new_tokens,
            stream=not args.no_stream,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
