#!/usr/bin/env python3
"""Classify PDFs into `epstein` or `user` groups (copy-only).

Rule:
- If a case ID is detected in the bottom-right region on ANY page -> `epstein`
- Otherwise -> `user`

Detection uses PDF text-layer extraction from the bottom-right ROI via PyMuPDF.

Examples:
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --dry-run
    python3 classify_pdfs_user_epstein.py /path/to/pdfs --recursive
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Case-id patterns for Epstein-related documents.
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


def detect_case_id(pdf_path: Path, cfg: Config) -> DetectionResult:
    """Detect case-id via text-layer extraction on all pages."""
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
            roi = fitz.Rect(x0, y0, x1, y1) & page_rect
            if roi.is_empty:
                continue

            # Word-based extraction for better control over region text order.
            # Each word tuple: (x0, y0, x1, y1, token, block, line, word).
            roi_words = sorted(
                (
                    (wy0, wx0, token)
                    for wx0, wy0, wx1, wy1, token, *_ in page.get_text("words")
                    if not (wx1 < roi.x0 or wx0 > roi.x1 or wy1 < roi.y0 or wy0 > roi.y1)
                ),
                key=lambda t: (t[0], t[1]),
            )
            text = " ".join(token for _, _, token in roi_words).strip()
            if not text:
                text = page.get_text("text", clip=roi)

            case_id = find_case_id(text)
            if case_id:
                return DetectionResult(True, case_id, page_index + 1)

    return DetectionResult(False, "", -1)


def classify_and_copy(pdf_path: Path, cfg: Config) -> tuple[str, str, int, str]:
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.directory).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Invalid directory: {input_dir}")

    report_csv = Path(args.report)
    if not report_csv.is_absolute():
        report_csv = input_dir / report_csv

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
    )

    pdf_files = sorted(iter_pdf_files(cfg.input_dir, cfg.recursive))
    if not pdf_files:
        print("No PDF files found.")
        return

    rows: list[tuple[str, str, int, str]] = []
    for pdf_path in pdf_files:
        # Skip already-grouped files in recursive mode.
        if cfg.recursive and any(
            part in {cfg.output_root_dir_name, cfg.epstein_dir_name, cfg.user_dir_name}
            for part in pdf_path.parts
        ):
            continue

        try:
            row = classify_and_copy(pdf_path, cfg)
        except Exception as exc:
            print(f"ERROR: {pdf_path.name}: {exc}")
            row = (pdf_path.name, "error", -1, str(exc))
        rows.append(row)

    write_report(rows, cfg.report_csv)
    print(f"\nDone. Processed {len(rows)} PDF file(s).")
    print(f"Report: {cfg.report_csv}")


if __name__ == "__main__":
    main()
