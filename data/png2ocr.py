"""
png2ocr.py

Converts png in [proj root]/data/png/
to txt in [proj root]/data/ocr .
Uses Paddle OCR for this process .
Skips already converted png .
"""

from pathlib import Path

import easyocr

PNG_DIR = Path("data/png")
OCR_DIR = Path("data/ocr")

reader = easyocr.Reader(["en"])


def _already_converted(png_path: Path) -> bool:
    """
    return true if output txt already exist
    """
    return (OCR_DIR / f"{png_path.stem}.txt").exists()


def convert_png(png_path: Path) -> Path:
    """
    run ocr on a single png
    returns output .txt
    """
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OCR_DIR / f"{png_path.stem}.txt"

    result = reader.readtext(str(png_path), detail=0)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result))

    return out_path


def convert_all() -> list[Path]:
    """
    run ocr on all png's in [proj root]/data/png/
    return all ouptut .txt paths
    """
    pngs = sorted(PNG_DIR.glob("*.png"))
    if not pngs:
        print("No PNGs found in data/png/")
        return []

    pending = [p for p in pngs if not _already_converted(p)]
    skipped = len(pngs) - len(pending)
    print(f" Skipping {skipped} already converted PNG(s).")
    print(f" Count PNG(s) to convert: {len(pending)}")

    all_outputs = []
    for png in pending:
        print(f"  Processing: {png.name}")
        all_outputs.append(convert_png(png))

    return all_outputs


if __name__ == "__main__":
    convert_all()
