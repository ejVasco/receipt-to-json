"""
png2ocr.py

Converts png in [proj root]/{PNG_DIR} .
to txt in [proj root]/{OCR_DIR} .
Uses easyocr for this process .
Skips already converted png .
"""

from pathlib import Path

import easyocr

from config import OCR_DIR, PNG_DIR

reader = easyocr.Reader(["en"])


def _already_converted(png_path: Path) -> bool:
    """
    return true if output txt already exist
    """
    return (OCR_DIR / f"{png_path.stem}.txt").exists()


def png2ocr(png_path: Path) -> Path:
    """
    run ocr on a single png
    returns output .txt
    """
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OCR_DIR / f"{png_path.stem}.txt"

    result = reader.readtext(str(png_path), detail=0)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(item) for item in result))

    return out_path


def pngs2ocrs() -> list[Path]:
    """
    run ocr on all png's in [proj root]/{PNG_DIR}
    return all ouptut TXT paths
    """
    pngs = sorted(PNG_DIR.glob("*.png"))
    if not pngs:
        print(f"No PNGs found in {PNG_DIR}")
        return []

    pending = [p for p in pngs if not _already_converted(p)]
    skipped = len(pngs) - len(pending)
    print(f" Skipping {skipped} already converted PNG(s).")
    print(f" Count PNG(s) to convert: {len(pending)}")

    all_outputs = []
    for png in pending:
        print(f"  Processing: {png.name}")
        all_outputs.append(png2ocr(png))

    return all_outputs


if __name__ == "__main__":
    pngs2ocrs()
