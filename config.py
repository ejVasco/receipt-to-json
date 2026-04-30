from pathlib import Path

# All PATHS are from project root dir


# these are for training
LABELS_DIR = Path("data/labels")
OCR_DIR = Path("data/ocr")
PNG_DIR = Path("data/png")
PDF_DIR = Path("data/raw_pdf")

# for testing/utilizing the model
MODEL_PATH = Path("model/receipt_mlp.pt")
RAW_INPUT_DIR = Path("test/input")  # pdf
OUTPUT_DIR = Path("test/output")  # json
TEST_PNG_DIR = Path("test/png")
TEST_OCR_DIR = Path("test/ocr")
