"""
pdf2png.py

Converts PDFs in [project root]/data/raw_pdf
to PNG images in [project root]/data/png/ .
Each page becomes a separate file
(this works only if each receipt is a 1 page PDF) .
Skips PDFs which are already converted .
"""

from pathlib import Path

import fitz  # pymupdf

# paths are from project root
RAW_PDF_DIR = Path("data/raw_pdf")
PNG_DIR = Path("data/png")


def _already_converted(pdf_path: Path) -> bool:
    """
    Return True if output PNG already exist for this PDf
    """
    return (PNG_DIR / f"{pdf_path.stem}.png").exists()


def convert_pdf(pdf_path: Path, dpi: int = 200) -> Path:
    """
    Convert a single PDF to a PNG
    Returns output path
    """
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PNG_DIR / f"{pdf_path.stem}.png"

    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(out_path)
    doc.close()

    print(f"   [done] {out_path.name}")
    return out_path


def convert_all(dpi: int = 200) -> list[Path]:
    """
    Converts all not yet converted PDFs in [proj root]/data/raw_pdf/
    Returns all list of output PNG paths
    """
    pdfs = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw_pdf/")
        return []

    pending = [p for p in pdfs if not _already_converted(p)]
    skipped = len(pdfs) - len(pending)
    print(f" Skipping {skipped} already converted PDF(s).")
    print(f" Count PDF(s) to convert: {len(pending)}")

    all_outputs = []
    for pdf in pending:
        print(f"  Processing: {pdf.name}")
        all_outputs.append(convert_pdf(pdf, dpi=dpi))

    return all_outputs


if __name__ == "__main__":
    convert_all()
