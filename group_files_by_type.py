#!/usr/bin/env python3
"""Group files into folders by file type.

Example:
    python3 group_files_by_type.py /path/to/folder
    python3 group_files_by_type.py /path/to/folder --recursive --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

CATEGORY_MAP: dict[str, set[str]] = {
    "images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".svg"},
    "pdfs": {".pdf"},
    "words": {".doc", ".docx", ".odt", ".rtf"},
    "texts": {".txt", ".md", ".csv", ".log"},
    "spreadsheets": {".xls", ".xlsx", ".ods"},
    "presentations": {".ppt", ".pptx", ".odp"},
    "archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"},
    "videos": {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm"},
    "audio": {".mp3", ".wav", ".aac", ".ogg", ".flac", ".m4a"},
    "code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".swift", ".go", ".php", ".rb", ".rs"},
}

# Build a reverse lookup for O(1) extension -> category.
_EXT_TO_CATEGORY: dict[str, str] = {
    ext: cat for cat, exts in CATEGORY_MAP.items() for ext in exts
}


def category_for_extension(extension: str) -> str:
    return _EXT_TO_CATEGORY.get(extension.lower(), "others")


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


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def organize(
    directory: Path,
    recursive: bool = False,
    dry_run: bool = False,
    output_root: str = "",
) -> None:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    items = directory.rglob("*") if recursive else directory.iterdir()
    output_base = (directory / output_root).resolve() if output_root else directory

    # Collect files upfront so we know the total for progress display.
    files = [
        item for item in items
        if item.is_file() and item.resolve() != Path(__file__).resolve()
    ]
    total = len(files)
    if total == 0:
        print("No files found.")
        return

    print(f"Processing {total} file(s)...")

    counts: Counter[str] = Counter()
    start_time = time.monotonic()

    for done, item in enumerate(files, 1):
        category = category_for_extension(item.suffix)
        target_dir = output_base / category
        target_path = unique_destination(target_dir / item.name)

        if dry_run:
            print(f"[DRY RUN] {item} -> {target_path}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target_path))
            print(f"Copied: {item.name} -> {category}/{target_path.name}")
        counts[category] += 1

        elapsed = time.monotonic() - start_time
        remaining = total - done
        if elapsed > 0:
            eta = _format_eta(elapsed / done * remaining)
        else:
            eta = "calculating..."
        cat_summary = " | ".join(f"{cat}: {n}" for cat, n in sorted(counts.items()))
        print(f"  [{done}/{total}] {cat_summary} | remaining: {remaining} | ETA: {eta}")

    elapsed = time.monotonic() - start_time
    cat_summary = " | ".join(f"{cat}: {n}" for cat, n in sorted(counts.items()))
    print(f"\nDone. Processed {total} file(s) in {_format_eta(elapsed)}.")
    print(f"  {cat_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group files by file type")
    parser.add_argument("directory", nargs="?", default=".", help="Folder to organize (default: current folder)")
    parser.add_argument("--recursive", action="store_true", help="Also organize files in subfolders")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without moving files")
    parser.add_argument("--output-root", default="", help="Optional output root folder (e.g. grouped -> grouped/images)")

    args = parser.parse_args()

    organize(
        Path(args.directory).expanduser().resolve(),
        recursive=args.recursive,
        dry_run=args.dry_run,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
