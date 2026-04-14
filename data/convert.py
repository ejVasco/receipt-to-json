"""
convert.py
single file for running pdf -> png -> ocr (txt)

(reminder to run from root)
usage:
    python -m data.convert
    python -m data.convert --pdf   # only convert PDFs to PNGs
    python -m data.convert --ocr   # only run ocr on PNGs
    python -m data.convert --force # force reconvert even if already converted file found
"""

import argparse
import shutil
from pathlib import Path

import data.pdf2png
import data.png2ocr


def run() -> None:
    pass


if __name__ == "__main__":
    run()
