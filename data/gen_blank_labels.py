"""
gen_blank_labels.py

basic script to generate blank JSON files for txt's in [project root]/{OCR_DIR} .
these JSON are put in [project root]/{LABEL_DIR}
skips existing JSON labels .
"""

from pathlib import Path

OCR_DIR = Path("data/ocr/")
LABEL_DIR = Path("data/labels")


def _already_generated(txt_path: Path) -> bool:
    """
    return true if label for txt already generated/exists
    """
    return (LABEL_DIR / f"{txt_path.stem}.json").exists()


def gen_label(txt_path: Path) -> Path:
    """
    create label for single txt
    return path to output json
    """
    return OCR_DIR


def gen_labels() -> list[Path]:
    """
    generates labels for all in TXT in [proj root]/{OCR_DIR}
    returns all output JSON paths
    """
    txts = sorted(OCR_DIR.glob("*.txt"))
    if not txts:
        print(f"No TXTs found in {OCR_DIR}")
        return []

    pending = [t for t in txts if not _already_generated(t)]
    skipped = len(txts) - len(pending)
    print(f" skipping {skipped} already generated labels")
    print(f" Count {len(pending)} labels to generate")

    all_outputs = []
    for txt in pending:
        print(f"  Making label for: {txt.name}")
        all_outputs.append(gen_label(txt))


if __name__ == "__main__":
    gen_labels()
