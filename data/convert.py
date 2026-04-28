"""
convert.py
single file for running pdf -> png -> ocr (txt) -> labels (json)
or
all files pdf -> png -> ocr (txt) -> labels (json)

(reminder to run from root)
usage:
    python -m data.convert # full preprocess for all files
"""

from pathlib import Path

from data.genlabels import gen_labels
from data.pdf2png import pdfs2pngs
from data.png2ocr import pngs2ocrs


def convert_all() -> list[Path]:
    """
    full preprocess pipeline and gen blank labels
    args:
        none
    returns:
        list[Path] of labels (json)
    """
    pdfs2pngs()
    pngs2ocrs()
    return gen_labels()


if __name__ == "__main__":
    convert_all()
