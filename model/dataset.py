"""
dataset.py
converts ocr txt files and label json files into labeled lines for MLP training

pipeline:
    1. split ocr text into lines, remove blank lines
    2. label lines (via fuzzy matching against JSON ground truth
    3. extract fixed length feature vector per line
    4. return as pytorch tensors

labels:
    0 other
    1 merchant
    2. address
    3 date
    4 time
    5 item name
    6 item price
    7 item modifier
    8 subtotal
    9 tax
    10 total

theres more in the json template, but in this dataset its been essentially unused
"""

import difflib
import json
import re
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

# ------- label map ----------------------------
LABELS = [
    "OTHER",
    "MERCHANT",
    "ADDRESS",
    "DATE",
    "TIME",
    "ITEM_NAME",
    "ITEM_PRICE",
    "ITEM_MODIFIER",
    "SUBTOTAL",
    "TAX",
    "TOTAL",
]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# ----------- regex patterns ---------------------
_PRICE_RE = re.compile(r"\$?\s*\d{1,4}[.,]\d{2}\b")
_DATE_RE = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b")
_TIME_RE = re.compile(r"\b\d{1,2}[;:]\d{2}\s*(?:AM|PM|am|pm)?\b")


# --- fuzzy similarity --------------------
def _sim(a: str, b: str) -> float:
    """normalized similarity ratio between 2 string,s case insensitive"""
    return difflib.SequenceMatcher(
        None,
        re.sub(r"\s+", " ", a).strip().lower(),
        re.sub(r"\s+", " ", b).strip().lower(),
    ).ratio()


def _best_sim(line: str, candidates: list[str]) -> float:
    """return highest similarity between line and candidate string"""
    if not candidates:
        return 0.0
    return max(_sim(line, c) for c in candidates)


# ------- auto labeler ----------
def label_line(line: str, label_json: dict, position: float) -> str:
    """
    assign label stringto a single ocr line given gnound truth json

    strategy (first match wins) (ordered by specificity)
        1. regex patterns for date/time
        2. fuzzy amtch for totals (subtotal / tax / total)
        3. fuzzy match item prices and item names / modifiers
        4. fuzzy match for merchant and address
        5. fall back to other
    """
    stripped = line.strip()
    if not stripped:
        return "OTHER"

    totals = label_json.get("totals", {}) or {}
    items = label_json.get("items", []) or []

    # 1. date/time via regex
    if _DATE_RE.search(stripped):
        return "DATE"
    if _TIME_RE.search(stripped):
        return "TIME"

    # 2. totals
    def _price_str(v) -> Optional[str]:
        return f"{float(v):.2f}" if v is not None else None

    subtotal_s = _price_str(totals.get("subtotal"))
    tax_s = _price_str(totals.get("tax"))
    total_s = _price_str(totals.get("total"))

    # check for price total
    if _PRICE_RE.search(stripped):
        if total_s and total_s in re.sub(r"[^\d.]", "", stripped):
            return "TOTAL"
        if tax_s and tax_s in re.sub(r"[^\d.]", "", stripped):
            return "TAX"
        if subtotal_s and subtotal_s in re.sub(r"[^\d.]", "", stripped):
            return "SUBTOTAL"

    # 3. items
    item_names = [it["name"] for it in items if it.get("name")]
    item_prices = [
        _price_str(it.get("price")) for it in items if it.get("price") is not None
    ]
    modifiers = [
        mod["name"]
        for it in items
        for mod in (it.get("modifiers") or [])
        if mod.get("name")
    ]

    if item_prices and _PRICE_RE.search(stripped):
        stripped_digits = re.sub(r"[^\d.]", "", stripped)
        if any(p in stripped_digits for p in item_prices if p):
            return "ITEM_PRICE"

    if modifiers and _best_sim(stripped, modifiers) >= 0.55:
        return "ITEM_MODIFIER"

    if item_names and _best_sim(stripped, item_names) >= 0.45:
        return "ITEM_NAME"

    # 4. merchant / addresss
    merchant = label_json.get("merchant", "") or ""
    address = label_json.get("address", "") or ""

    # address lines usually after mecrhant in the upper part of receipt
    address_parts = (
        [p.strip() for p in re.split(r"[,\n]", address) if p.strip()] if address else []
    )

    if merchant and _sim(stripped, merchant) >= 0.45:
        return "MERCHANT"

    if address_parts and _best_sim(stripped, address_parts) >= 0.45:
        return "ADDRESS"

        # fallback, to guess find merchant name in top 20% of reiciept
    if (
        position < 0.20
        and re.match(r"^[A-Za-z\s\-&']+$", stripped)
        and len(stripped) > 3
    ):
        return "MERCHANT"

    return "OTHER"


# --- feature extraction -----
FEATURE_DIM = 15


def extract_features(line: str, line_idx: int, total_lines: int) -> list[float]:
    """
    return fixed length feature vector for one ocr line.

    feature (15):
        0  relative_position   - line index / total lines
        1  is_early            - 1 if position in first 20%
        2  is_late             - 1 if in last 20%
        3  char_count          - len(line), clamped/normalized to [0,1] over 80 characters
        4  word_count          - num of whitespace separated tokens, norm over 10
        5  digit_ratio         - fraction of chars that are digits
        6  alpha_ratio         - fraction of chars that are letters
        7  upper_ratio         - fraction of alpha chars that are uppercase
        8  has_dollar          - 1 if $ is present
        9  has_price_pattern   - 1 if price regex matches
        10 has_date_pattern    - 1 if date regex matches
        11 has_time_pattern    - 1 if time regex matches
        12 has_modifier_prefix - 1 if line starts with +/- (typical modifier hints)
        13 punct_ratio         - fraction of chars that are punctuation
        14 ends_with_price     - 1 if line ends with digits that look like a price
    """
    s = line.strip()
    n = max(len(s), 1)
    pos = line_idx / max(total_lines - 1, 1)

    digits = sum(c.isdigit() for c in s)
    alphas = sum(c.isalpha() for c in s)
    uppers = sum(c.isupper() for c in s)
    punct = sum(c in ".,l/:;$" for c in s)
    words = s.split()

    has_modifier = 1.0 if re.match(r"^\s*[+\-]", s) else 0.0
    ends_price = 1.0 if re.search(r"\d{1,4}[.,]\d{2}\s*$", s) else 0.0

    return [
        pos,  # 0
        1.0 if pos < 0.20 else 0.0,  # 1
        1.0 if pos > 0.80 else 0.0,  # 2
        min(len(s) / 80.0, 1.0),  # 3
        min(len(words) / 10.0, 1.0),  # 4
        digits / n,  # 5
        alphas / n,  # 6
        uppers / max(alphas, 1),  # 7
        1.0 if "$" in s else 0.0,  # 8
        1.0 if _PRICE_RE.search(s) else 0.0,  # 9
        1.0 if _DATE_RE.search(s) else 0.0,  # 10
        1.0 if _TIME_RE.search(s) else 0.0,  # 11
        has_modifier,  # 12
        punct / n,  # 13
        ends_price,  # 14
    ]


# ----- dataset ------------
class ReceiptDataset(Dataset):
    """
    each item in dataset is one ocr line represented as:
        x: float tensor of shape (feature_dim,)
        y: longtensor scalar - sclass index
    args:
        ocr_dir    : path to directory of txt OCR files
        labels_dir : path to dir of matching json lable files
        stems      : optional list of file stems to include (for LOOCV splits), if none then all pairs are used
    """

    def __init__(
        self,
        ocr_dir: Path,
        labels_dir: Path,
        stems: Optional[list[str]] = None,
    ):
        self.samples: list[tuple[list[float], int]] = []

        ocr_dir = Path(ocr_dir)
        label_path = Path(labels_dir)

        all_stems = (
            stems
            if stems is not None
            else [
                p.stem
                for p in ocr_dir.glob("*.txt")
                if (labels_dir / (p.stem + ".json")).exists()
            ]
        )

        for stem in all_stems:
            ocr_path = ocr_dir / f"{stem}.txt"
            label_path = labels_dir / f"{stem}.json"

            if not ocr_path.exists() or not label_path.exists():
                continue

            ocr_text = ocr_path.read_text(encoding="utf-8", errors="replace")
            label_json = json.loads(label_path.read_text(encoding="utf-8"))

            lines = [lbl for lbl in ocr_text.splitlines() if lbl.strip()]
            total = len(lines)

            for idx, line in enumerate(lines):
                pos = idx / max(total - 1, 1)
                label = label_line(line, label_json, pos)
                label_idx = LABEL2IDX[label]
                features = extract_features(line, idx, total)
                self.samples.append((features, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        features, label = self.samples[idx]
        x = torch.as_tensor(features, dtype=torch.float32)  # type: ignore[attr-defined]
        y = torch.as_tensor(label, dtype=torch.long)  # type: ignore[attr-defined]
        return x, y


# -- helpers for LOOCVsplits
def get_all_stems(ocr_dir: Path, labels_dir: Path) -> list[str]:
    """return sorted list of stems that have both OCR and label files"""
    ocr_dir = Path(ocr_dir)
    labels_dir = Path(labels_dir)
    return sorted(
        p.stem
        for p in ocr_dir.glob("*.txt")
        if (labels_dir / (p.stem + ".json")).exists()
    )


def loocv_splits(stems: list[str]):
    """
    yield (train_stems, val_stems) for leave one out cross validation
    val_stemsis always a single-element list
    """
    for i, held_out in enumerate(stems):
        train = stems[:i] + stems[i + 1 :]
        yield train, [held_out]
