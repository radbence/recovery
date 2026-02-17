#!/usr/bin/env python3
"""Group files into folders by file type.

Example:
    python3 group_files_by_type.py /path/to/folder
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CATEGORY_MAP = {
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


def category_for_extension(extension: str) -> str:
    extension = extension.lower()
    for category, extensions in CATEGORY_MAP.items():
        if extension in extensions:
            return category
    return "others"


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


def organize(directory: Path, recursive: bool = False, dry_run: bool = False) -> None:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    files = directory.rglob("*") if recursive else directory.iterdir()

    moved = 0
    for item in files:
        if not item.is_file():
            continue

        # Avoid moving this script while it is running.
        if item.resolve() == Path(__file__).resolve():
            continue

        category = category_for_extension(item.suffix)
        target_dir = directory / category
        target_path = unique_destination(target_dir / item.name)

        if dry_run:
            print(f"[DRY RUN] {item} -> {target_path}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target_path))
            print(f"Copied: {item.name} -> {category}/{target_path.name}")
        moved += 1

    print(f"\nDone. Processed {moved} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group files by file type")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Folder to organize (default: current folder)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also organize files in subfolders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without moving files",
    )

    args = parser.parse_args()
    organize(Path(args.directory).expanduser().resolve(), recursive=args.recursive, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
