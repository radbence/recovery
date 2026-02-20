# Recovery — PDF Classification & File Organization Toolkit

A set of Python utilities for classifying PDFs and organizing files by type. Built for large-scale batch processing with multiprocessing support.

## Tools

### `classify_pdfs_user_epstein.py`

Classifies PDFs into **epstein** or **user** groups by detecting case IDs (e.g. `EFTA00037815`, `1:15-cv-07433`) in the bottom-right region of each page. Matching uses the PDF text layer via PyMuPDF — no OCR required.

**Features:**
- Configurable bottom-right ROI (Region of Interest) for case-ID detection
- Multiprocessing with configurable worker count
- Optional GPU-accelerated OCR fallback for scanned PDFs (CUDA via EasyOCR)
- Dry-run mode to preview actions without copying files
- CSV report generation (`classification_report.csv`)
- Recursive directory scanning
- First-page-only mode for faster processing

### `group_files_by_type.py`

Organizes files into category folders (images, PDFs, documents, spreadsheets, archives, videos, audio, code, and more) based on file extension.

**Features:**
- 10+ built-in file categories with common extensions
- Recursive folder processing
- Dry-run preview mode
- Automatic handling of filename conflicts

## Requirements

- Python 3.10+
- [PyMuPDF](https://pymupdf.readthedocs.io/) ≥ 1.24.0 (only for the PDF classifier)
- **GPU mode (optional):** NVIDIA GPU with CUDA support, [PyTorch](https://pytorch.org/) ≥ 2.6.0, and [EasyOCR](https://github.com/JaidedAI/EasyOCR) ≥ 1.7.0

## Installation

```bash
# CPU-only
pip install -r requirements_pdf_classifier.txt

# GPU-accelerated OCR (requires NVIDIA CUDA toolkit)
pip install -r requirements_gpu.txt
```

## Usage

### Classify PDFs

```bash
# Preview classification without copying
python3 classify_pdfs_user_epstein.py /path/to/pdfs --dry-run

# Classify recursively with 8 workers
python3 classify_pdfs_user_epstein.py /path/to/pdfs --recursive --workers 8

# Only check the first page of each PDF (faster)
python3 classify_pdfs_user_epstein.py /path/to/pdfs --first-page-only

# GPU-accelerated OCR fallback (useful for scanned PDFs without a text layer)
python3 classify_pdfs_user_epstein.py /path/to/pdfs --gpu
```

Classified PDFs are copied into `<input_dir>/pdf/epstein/` and `<input_dir>/pdf/user/`. A CSV report is written to `<input_dir>/classification_report.csv`.

Run `python3 classify_pdfs_user_epstein.py --help` for all options.

### Organize Files by Type

```bash
# Organize files in a folder
python3 group_files_by_type.py /path/to/folder

# Recursive with dry-run preview
python3 group_files_by_type.py /path/to/folder --recursive --dry-run

# Copy organized files into a separate output folder
python3 group_files_by_type.py /path/to/folder --output-root grouped
```

Run `python3 group_files_by_type.py --help` for all options.
