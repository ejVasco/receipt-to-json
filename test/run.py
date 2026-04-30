"""
test/run.py

runs ful inference on pdfs in test/input/
and writes structured json to test/output

pipeline per file follows the same as found in data/

usage (project root):
    python -m test.run
"""

import json
import re
import sys
from pathlib import Path

import easyocr
import fitz  # pymupdf
import torch

import config
from model.dataset import LABELS, extract_features
from model.model import ReceiptMLP

# --------- preprocess helpers, but written specifically for test folders since i forgot to make those scripts reuasable
#


def _pdf2png(pdf_path: Path, dpi: int = 200) -> Path:
    """convert first page of pdf to png in config.TEST_PNG_DIR"""
    config.TEST_PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.TEST_PNG_DIR / f"{pdf_path.stem}.png"
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    pix.save(out_path)
    doc.close()
    return out_path


def _png2ocr(png_path: Path, reader: easyocr.Reader) -> tuple[Path, list[str]]:
    """use easy ocr on a png, write txt to config.TEST_OCR_DIR and return lines"""
    config.TEST_OCR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.TEST_OCR_DIR / f"{png_path.stem}.txt"
    lines = [str(item) for item in reader.readtext(str(png_path), detail=0)]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path, lines


# ------- infereenc
#
_PRICE_RE = re.compile(r"\d+[.,]\d{2}")


def _predict_lines(lines: list[str], model: ReceiptMLP) -> list[str]:
    """return predicted label string for every ocr line"""
    model.eval()
    total = len(lines)
    preds: list[str] = []
    with torch.no_grad():
        for idx, line in enumerate(lines):
            feats = extract_features(line, idx, total)
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            pred_idx = int(model(x).argmax(dim=1).item())
            preds.append(LABELS[pred_idx])
        return preds


def _extract_price(text: str) -> float | None:
    """pull first decimal num out of a string or return none"""
    m = _PRICE_RE.search(text)
    return float(m.group().replace(",", ".")) if m else None


# -------- json asssembly -------------\
#
def _build_json(lines: list[str], preds: list[str]) -> dict:
    """
    collapse per line predictions into structured receipt dict

    items grouped by pairing each item_name with item_price and item_modifier lines that immediately followt it
    """

    result: dict = {
        "merchant": "",
        "address": "",
        "date": "",
        "time": "",
        "items": [],
        "totals": {
            "subtotal": None,
            "discount": None,
            "tax": None,
            "tip": None,
            "total": None,
        },
    }

    current_item: dict | None = None

    def _flush_item():
        nonlocal current_item
        if current_item is not None:
            result["items"].append(current_item)
            current_item = None

    for line, label in zip(lines, preds):
        s = line.strip()
        if not s:
            continue

        if label == "MERCHANT":
            if not result["merchant"]:
                result["merchant"] = s

        elif label == "ADDRESS":
            result["address"] = (result["address"] + " " + s).strip()

        elif label == "DATE":
            if not result["date"]:
                result["date"] = s

        elif label == "TIME":
            if not result["time"]:
                result["time"] = s

        elif label == "ITEM_NAME":
            _flush_item()
            current_item = {"quantity": None, "name": s, "price": None, "modifiers": []}

        elif label == "ITEM_PRICE":
            price = _extract_price(s)
            if current_item is not None:
                current_item["price"] = price
            else:
                # orphaned price line - create namelenss item so nothing lost
                current_item = {
                    "quantity": None,
                    "name": "",
                    "price": price,
                    "modifiers": [],
                }

        elif label == "ITEM_MODIFIER":
            if current_item is None:
                current_item = {
                    "quantity": None,
                    "name": "",
                    "price": None,
                    "modifiers": [],
                }
            current_item["modifiers"].append({"name": s, "price": None})

        elif label == "SUBTOTAL":
            result["totals"]["subtotal"] = _extract_price(s)

        elif label == "TAX":
            result["totals"]["tax"] = _extract_price(s)

        elif label == "TOTAL":
            result["totals"]["total"] = _extract_price(s)

        # other -> skip

    _flush_item()
    return result


# - ------------ main ---------------
#
def run() -> None:
    pdfs = sorted(config.RAW_INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.RAW_INPUT_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) to process")

    print("loading model...")
    model = ReceiptMLP()
    model.load_state_dict(
        torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval

    print("Initializing OCR reader...")
    reader = easyocr.Reader(["en"], verbose=False)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdfs:
        print(f"\n[{pdf_path.name}]")

        png_path = _pdf2png(pdf_path)
        print(f"  png -> {png_path}")

        _, raw_lines = _png2ocr(png_path, reader)
        lines = [l for l in raw_lines if l.strip()]
        print(f"  ocr -> {len(lines)} lines")

        preds = _predict_lines(lines, model)

        label_counts = {}
        for p in preds:
            label_counts[p] = label_counts.get(p, 0) + 1
        print(f"  pred -> {label_counts}")

        output = _build_json(lines, preds)

        out_path = config.OUTPUT_DIR / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"  json -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    run()
