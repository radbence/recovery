#!/usr/bin/env python3
"""Classify PDFs into `epstein` or `user` groups (copy-only).

Rule:
- If a case ID is detected in the bottom-right region on ANY page -> `epstein`
- Otherwise -> `user`

Detection pipeline per page:
1) (Optional) PDF text extraction from bottom-right ROI (fast path, if PyMuPDF installed)
2) OCR with local Ollama model (default: glm-ocr) on bottom-right ROI image

Examples:
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --dry-run
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --model glm-ocr:q8_0
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from path_config import DEFAULT_CONFIG_PATH, load_config_section


# You can extend this list with known case-id variants.
CASE_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bEFTA\d{8}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}:\d{2}-cv-\d{3,7}\b", re.IGNORECASE),
    re.compile(r"\bCase\s+\d{1,2}:\d{2}-cv-\d{3,7}\b", re.IGNORECASE),
    re.compile(r"\bCase\s*(No\.?|#)?\s*\d{1,2}:\d{2}-cv-\d{3,7}\b", re.IGNORECASE),
    re.compile(r"\bNo\.?\s*\d{1,2}:\d{2}-cv-\d{3,7}\b", re.IGNORECASE),
]


@dataclass
class DetectionResult:
    is_epstein: bool
    match: str
    method: str
    page: int


@dataclass
class Config:
    input_dir: Path
    output_root_dir_name: str
    epstein_dir_name: str
    user_dir_name: str
    report_csv: Path
    model: str
    dpi: int
    roi_x: float
    roi_y: float
    roi_w: float
    roi_h: float
    recursive: bool
    dry_run: bool
    use_ocr: bool
    ollama_url: str
    timeout_sec: float


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem, suffix = path.stem, path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def iter_pdf_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    items = input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf")
    for path in items:
        if path.is_file():
            yield path


def find_case_id(text: str) -> str:
    for pattern in CASE_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return ""


def build_ollama_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/{endpoint.lstrip('/')}"


def ensure_ollama_running(base_url: str, timeout_sec: float) -> None:
    url = build_ollama_url(base_url, "api/tags")
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama is not ready: HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Start it first (e.g. open Ollama app or run `ollama serve`)."
        ) from exc


def ask_ollama_ocr(base_url: str, model: str, image_bytes: bytes, timeout_sec: float) -> str:
    prompt = (
        "Read only the visible text in this image. "
        "Return plain text exactly as seen, preserving IDs and punctuation."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
    }

    url = build_ollama_url(base_url, "api/generate")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        data = json.loads(response.read().decode("utf-8"))

    return str(data.get("response", ""))


def detect_with_pymupdf(pdf_path: Path, cfg: Config) -> DetectionResult:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF is not installed") from exc

    with fitz.open(str(pdf_path)) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_rect = page.rect

            # Bottom-right normalized ROI.
            x0 = page_rect.x0 + page_rect.width * cfg.roi_x
            y0 = page_rect.y0 + page_rect.height * cfg.roi_y
            x1 = x0 + page_rect.width * cfg.roi_w
            y1 = y0 + page_rect.height * cfg.roi_h
            roi = fitz.Rect(x0, y0, x1, y1)

            # Keep ROI valid even if custom args overflow.
            roi = roi & page_rect
            if roi.is_empty:
                continue

            # 1) Fast text-layer extraction.
            # Prefer word-based extraction for better control over region text order.
            words = page.get_text("words")
            roi_words = []
            for w in words:
                wx0, wy0, wx1, wy1, token = w[0], w[1], w[2], w[3], w[4]
                if wx1 < roi.x0 or wx0 > roi.x1 or wy1 < roi.y0 or wy0 > roi.y1:
                    continue
                roi_words.append((wy0, wx0, token))

            roi_words.sort(key=lambda t: (t[0], t[1]))
            text = " ".join(token for _, _, token in roi_words).strip()
            if not text:
                text = page.get_text("text", clip=roi)

            text_match = find_case_id(text)
            if text_match:
                return DetectionResult(True, text_match, "text-roi", page_index + 1)

            # 2) OCR fallback on rendered ROI.
            if not cfg.use_ocr:
                continue

            scale = cfg.dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, clip=roi, alpha=False)
            image_bytes = pix.tobytes("png")

            ocr_text = ask_ollama_ocr(cfg.ollama_url, cfg.model, image_bytes, cfg.timeout_sec)
            ocr_match = find_case_id(ocr_text)
            if ocr_match:
                return DetectionResult(True, ocr_match, "glm-ocr", page_index + 1)

    return DetectionResult(False, "", "none", -1)


def detect_case_id(pdf_path: Path, cfg: Config) -> DetectionResult:
    """Detect case-id on ALL pages. Returns on first positive match."""
    return detect_with_pymupdf(pdf_path, cfg)


def classify_and_copy(pdf_path: Path, cfg: Config) -> tuple[str, str, str, int, str]:
    result = detect_case_id(pdf_path, cfg)
    group = cfg.epstein_dir_name if result.is_epstein else cfg.user_dir_name

    target_dir = cfg.input_dir / cfg.output_root_dir_name / group
    target_path = unique_destination(target_dir / pdf_path.name)
    relative_target = target_path.relative_to(cfg.input_dir)

    if cfg.dry_run:
        print(f"[DRY RUN] {pdf_path.name} -> {relative_target}")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(pdf_path), str(target_path))
        print(f"Copied: {pdf_path.name} -> {relative_target}")

    return (
        pdf_path.name,
        group,
        result.method,
        result.page,
        result.match,
    )


def write_report(rows: list[tuple[str, str, str, int, str]], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group", "method", "matched_page", "matched_value"])
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify PDFs into epstein/user using bottom-right case-id detection")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to shared JSON config")
    parser.add_argument("directory", nargs="?", default=".", help="Folder containing PDFs (default: current folder)")
    parser.add_argument("--recursive", action="store_true", help="Scan PDFs in subfolders too")
    parser.add_argument("--dry-run", action="store_true", help="Preview copy actions only")

    parser.add_argument("--output-root", default="pdf", help="Root output folder for grouped PDFs")
    parser.add_argument("--epstein-dir", default="epstein", help="Output folder name for epstein PDFs")
    parser.add_argument("--user-dir", default="user", help="Output folder name for user PDFs")
    parser.add_argument("--report", default="classification_report.csv", help="CSV report path (relative to input dir if not absolute)")

    parser.add_argument("--model", default="glm-ocr:q8_0", help="Ollama vision model (default: glm-ocr:q8_0)")
    parser.add_argument("--use-ocr", action="store_true", help="Enable Ollama OCR fallback (off by default)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--timeout-sec", type=float, default=25.0, help="Network timeout for Ollama requests")
    parser.add_argument("--dpi", type=int, default=220, help="Render DPI for ROI OCR")

    parser.add_argument("--roi-x", type=float, default=0.52, help="Bottom-right ROI start x (0-1)")
    parser.add_argument("--roi-y", type=float, default=0.70, help="Bottom-right ROI start y (0-1, page origin is top)")
    parser.add_argument("--roi-w", type=float, default=0.48, help="Bottom-right ROI width ratio (0-1)")
    parser.add_argument("--roi-h", type=float, default=0.30, help="Bottom-right ROI height ratio (0-1)")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config_section(Path(args.config), "classify_pdfs_user_epstein")

    directory = args.directory
    if directory == "." and isinstance(config.get("directory"), str):
        directory = str(config["directory"])

    recursive = args.recursive
    if not recursive and isinstance(config.get("recursive"), bool):
        recursive = bool(config["recursive"])

    dry_run = args.dry_run
    if not dry_run and isinstance(config.get("dry_run"), bool):
        dry_run = bool(config["dry_run"])

    output_root = args.output_root
    if output_root == "pdf" and isinstance(config.get("output_root"), str):
        output_root = str(config["output_root"])

    user_dir = args.user_dir
    if user_dir == "user" and isinstance(config.get("user_dir"), str):
        user_dir = str(config["user_dir"])

    epstein_dir = args.epstein_dir
    if epstein_dir == "epstein" and isinstance(config.get("epstein_dir"), str):
        epstein_dir = str(config["epstein_dir"])

    report_arg = args.report
    if report_arg == "classification_report.csv" and isinstance(config.get("report"), str):
        report_arg = str(config["report"])

    input_dir = Path(directory).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Invalid directory: {input_dir}")

    report_csv = Path(report_arg)
    if not report_csv.is_absolute():
        report_csv = input_dir / report_csv

    cfg = Config(
        input_dir=input_dir,
        output_root_dir_name=output_root,
        epstein_dir_name=epstein_dir,
        user_dir_name=user_dir,
        report_csv=report_csv,
        model=args.model,
        dpi=args.dpi,
        roi_x=args.roi_x,
        roi_y=args.roi_y,
        roi_w=args.roi_w,
        roi_h=args.roi_h,
        recursive=recursive,
        dry_run=dry_run,
        use_ocr=args.use_ocr,
        ollama_url=args.ollama_url,
        timeout_sec=args.timeout_sec,
    )

    if cfg.use_ocr:
        ensure_ollama_running(cfg.ollama_url, cfg.timeout_sec)

    pdf_files = sorted(iter_pdf_files(cfg.input_dir, cfg.recursive))
    if not pdf_files:
        print("No PDF files found.")
        return

    rows: list[tuple[str, str, str, int, str]] = []
    for pdf_path in pdf_files:
        # Prevent re-processing already grouped files if recursive mode is used.
        if cfg.recursive and any(
            part in {cfg.output_root_dir_name, cfg.epstein_dir_name, cfg.user_dir_name} for part in pdf_path.parts
        ):
            continue

        try:
            row = classify_and_copy(pdf_path, cfg)
        except Exception as exc:
            print(f"ERROR: {pdf_path.name}: {exc}")
            row = (pdf_path.name, "error", "error", -1, str(exc))
        rows.append(row)

    write_report(rows, cfg.report_csv)
    print(f"\nDone. Processed {len(rows)} PDF file(s).")
    print(f"Report: {cfg.report_csv}")


if __name__ == "__main__":
    main()
