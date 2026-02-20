#!/usr/bin/env python3
"""Classify PDFs into `epstein` or `user` groups (copy-only).

Rule:
- If a case ID is detected in the bottom-right region on ANY page -> `epstein`
- Otherwise -> `user`

Detection uses PDF text-layer extraction from the bottom-right ROI via PyMuPDF.
Optimized for large-scale processing (100 GB+) with multiprocessing support.

Examples:
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --dry-run
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --recursive --workers 8
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --first-page-only
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Single combined regex for all case-id variants (one-pass matching).
# - EFTA-like IDs: 2-6 uppercase letters followed by 6-12 digits (e.g. EFTA00037815)
# - Court case numbers: e.g. 1:15-cv-07433, Case No. 1:15-cv-07433
CASE_ID_RE: re.Pattern[str] = re.compile(
    r"\b[A-Z]{2,6}\d{6,12}\b"
    r"|\bCase\s*(No\.?|#)?\s*\d{1,2}:\d{2}-cv-\d{3,7}\b"
    r"|\bNo\.?\s*\d{1,2}:\d{2}-cv-\d{3,7}\b"
    r"|\b\d{1,2}:\d{2}-cv-\d{3,7}\b",
    re.IGNORECASE,
)


@dataclass
class DetectionResult:
    is_epstein: bool
    match: str
    page: int


@dataclass
class Config:
    input_dir: Path
    output_root_dir_name: str
    epstein_dir_name: str
    user_dir_name: str
    report_csv: Path
    roi_x: float
    roi_y: float
    roi_w: float
    roi_h: float
    recursive: bool
    dry_run: bool
    first_page_only: bool
    workers: int
    gpu: bool


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
    m = CASE_ID_RE.search(text)
    return m.group(0) if m else ""


_ocr_reader = None


def _get_ocr_reader(gpu: bool):
    """Lazily initialize an EasyOCR reader (one per process)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "easyocr is required for --gpu mode. "
                "Install with: pip install -r requirements_gpu.txt"
            )
        _ocr_reader = easyocr.Reader(["en"], gpu=gpu)
    return _ocr_reader


def _ocr_roi(page, roi_rect, gpu: bool) -> str:
    """Render a page ROI to a bitmap and run OCR on the GPU."""
    import fitz  # noqa: F811
    import numpy as np

    mat = fitz.Matrix(2, 2)  # 2× zoom for better OCR accuracy
    pix = page.get_pixmap(matrix=mat, clip=roi_rect)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n,
    )
    if pix.n == 4:  # RGBA → RGB
        img = img[:, :, :3]

    reader = _get_ocr_reader(gpu)
    results = reader.readtext(img, detail=0)
    return " ".join(results)


def detect_case_id(pdf_path: Path, cfg: Config) -> DetectionResult:
    """Detect case-id via text-layer extraction. Stops at first match."""
    import fitz  # PyMuPDF  (lazy import; cached after first call)

    with fitz.open(str(pdf_path)) as doc:
        page_limit = 1 if cfg.first_page_only else len(doc)
        for page_index in range(page_limit):
            page = doc[page_index]
            rect = page.rect

            # Bottom-right ROI.
            x0 = rect.x0 + rect.width * cfg.roi_x
            y0 = rect.y0 + rect.height * cfg.roi_y
            x1 = x0 + rect.width * cfg.roi_w
            y1 = y0 + rect.height * cfg.roi_h
            roi = fitz.Rect(x0, y0, x1, y1) & rect
            if roi.is_empty:
                continue

            text = page.get_text("text", clip=roi)
            case_id = find_case_id(text)
            if case_id:
                return DetectionResult(True, case_id, page_index + 1)

            # GPU OCR fallback when text layer yields no match.
            if cfg.gpu:
                ocr_text = _ocr_roi(page, roi, gpu=cfg.gpu)
                case_id = find_case_id(ocr_text)
                if case_id:
                    return DetectionResult(True, case_id, page_index + 1)

    return DetectionResult(False, "", -1)


def _process_one(args: tuple[Path, Config]) -> tuple[str, str, int, str]:
    """Worker function for multiprocessing: detect + copy a single PDF."""
    pdf_path, cfg = args
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

    return pdf_path.name, group, result.page, result.match


def write_report(rows: list[tuple[str, str, int, str]], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group", "matched_page", "matched_value"])
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify PDFs into epstein/user using bottom-right case-id detection",
    )
    parser.add_argument("directory", nargs="?", default=".", help="Folder containing PDFs (default: current folder)")
    parser.add_argument("--recursive", action="store_true", help="Scan PDFs in subfolders too")
    parser.add_argument("--dry-run", action="store_true", help="Preview copy actions only")
    parser.add_argument("--output-root", default="pdf", help="Root output folder for grouped PDFs (default: pdf)")
    parser.add_argument("--epstein-dir", default="epstein", help="Output folder name for epstein PDFs (default: epstein)")
    parser.add_argument("--user-dir", default="user", help="Output folder name for user PDFs (default: user)")
    parser.add_argument("--report", default="classification_report.csv", help="CSV report path (default: classification_report.csv)")
    parser.add_argument("--roi-x", type=float, default=0.52, help="Bottom-right ROI start x (0-1)")
    parser.add_argument("--roi-y", type=float, default=0.70, help="Bottom-right ROI start y (0-1)")
    parser.add_argument("--roi-w", type=float, default=0.48, help="Bottom-right ROI width ratio (0-1)")
    parser.add_argument("--roi-h", type=float, default=0.30, help="Bottom-right ROI height ratio (0-1)")
    parser.add_argument("--first-page-only", action="store_true", help="Only check the first page of each PDF (faster)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU-accelerated OCR fallback (requires CUDA GPU and easyocr)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1, use 0 for all CPUs)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.directory).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Invalid directory: {input_dir}")

    report_csv = Path(args.report)
    if not report_csv.is_absolute():
        report_csv = input_dir / report_csv

    workers = max(args.workers, 0)
    if workers == 0:
        workers = os.cpu_count() or 1

    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU mode enabled: {torch.cuda.get_device_name(0)}")
            else:
                print("Warning: CUDA not available. OCR will run on CPU.")
        except ImportError:
            raise SystemExit(
                "GPU mode requires PyTorch and easyocr. "
                "Install with: pip install -r requirements_gpu.txt"
            )
        if workers > 1:
            print("Note: GPU mode uses 1 worker (the GPU handles parallelism).")
            workers = 1

    cfg = Config(
        input_dir=input_dir,
        output_root_dir_name=args.output_root,
        epstein_dir_name=args.epstein_dir,
        user_dir_name=args.user_dir,
        report_csv=report_csv,
        roi_x=args.roi_x,
        roi_y=args.roi_y,
        roi_w=args.roi_w,
        roi_h=args.roi_h,
        recursive=args.recursive,
        dry_run=args.dry_run,
        first_page_only=args.first_page_only,
        workers=workers,
        gpu=args.gpu,
    )

    skip_parts = {cfg.output_root_dir_name, cfg.epstein_dir_name, cfg.user_dir_name}
    pdf_files = sorted(
        p for p in iter_pdf_files(cfg.input_dir, cfg.recursive)
        if not cfg.recursive or not skip_parts.intersection(p.parts)
    )
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Processing {len(pdf_files)} PDF(s) with {workers} worker(s)...")

    rows: list[tuple[str, str, int, str]] = []
    if workers == 1:
        for pdf_path in pdf_files:
            try:
                rows.append(_process_one((pdf_path, cfg)))
            except Exception as exc:
                print(f"ERROR: {pdf_path.name}: {exc}")
                rows.append((pdf_path.name, "error", -1, str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, (pdf_path, cfg)): pdf_path
                for pdf_path in pdf_files
            }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    print(f"ERROR: {pdf_path.name}: {exc}")
                    rows.append((pdf_path.name, "error", -1, str(exc)))

    write_report(rows, cfg.report_csv)
    print(f"\nDone. Processed {len(rows)} PDF file(s).")
    print(f"Report: {cfg.report_csv}")


if __name__ == "__main__":
    main()
