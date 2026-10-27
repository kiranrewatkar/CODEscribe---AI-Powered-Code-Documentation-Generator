"""
research/confusion_matrix_eval/confusion_matrix_runner.py

Confusion Matrix Evaluation for the CODEscribe Documentation Audit Engine.

This script independently re-runs audit.py on freshly designed source/documentation
pairs with INDEPENDENTLY SPECIFIED ground-truth labels (not derived from the
workbook runner output, ROUGE/BLEU, or model predictions).

Task Framing
------------
We evaluate the audit engine's binary PARAMETER MENTION DETECTION:
  - Positive (P): a source-code parameter IS genuinely documented in the Parameters section.
  - Negative (N): a source-code parameter IS NOT genuinely documented in the Parameters section.
  - Audit Prediction True  → audit says MENTIONED (audit agrees it IS documented)
  - Audit Prediction False → audit says NOT_MENTIONED (audit says it is NOT documented)

This gives us 4 cells:
  TP: Audit says MENTIONED and ground truth says it IS documented.
  TN: Audit says NOT_MENTIONED and ground truth says it IS NOT documented.
  FP: Audit says MENTIONED but ground truth says it IS NOT documented.
  FN: Audit says NOT_MENTIONED but ground truth says it IS documented.

Disclaimer
----------
This evaluation assesses ONLY the audit engine's heuristic parameter-mention detection
accuracy relative to human-specified ground-truth labels. It does NOT measure:
- CodeT5 generation accuracy or ROUGE/BLEU scores.
- Runtime behavior or semantic correctness of documentation.
- Full documentation quality (returns, exceptions, etc.).
The ground-truth labels are manually specified based on visible Markdown text inspection.

Usage
-----
  $env:PYTHONPATH = "D:\\MCA\\CODEscribe"
  python research/confusion_matrix_eval/confusion_matrix_runner.py
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# Ensure app is importable when run from anywhere.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app.audit import audit_documentation  # noqa: E402


# ---------------------------------------------------------------------------
# Ground-Truth Labeled Cases
# ---------------------------------------------------------------------------
# Each entry defines:
#   source        - Python source code of the function under test
#   markdown      - The generated/candidate documentation Markdown
#   function      - The function name to audit
#   labeled_params - list of (param_name, gt_documented: bool)
#     gt_documented=True  → human inspection confirms this param IS in the Parameters section
#     gt_documented=False → human inspection confirms this param IS NOT in the Parameters section

LABELED_CASES = [
    # -----------------------------------------------------------------------
    # Case A: Fully documented function — all params in Parameters section
    # Expected: all TP (audit says MENTIONED, GT says True)
    # -----------------------------------------------------------------------
    {
        "case_id": "A",
        "name": "Fully documented — both params clearly in Parameters section",
        "source": """\
def add_numbers(x: int, y: int) -> int:
    return x + y
""",
        "markdown": """\
## add_numbers

Adds two integers.

```python
def add_numbers(x: int, y: int) -> int:
```

### Parameters
- **x** (int): The first integer.
- **y** (int): The second integer.

### Returns
int: The sum of x and y.
""",
        "function": "add_numbers",
        "labeled_params": [("x", True), ("y", True)],
    },
    # -----------------------------------------------------------------------
    # Case B: Partial documentation — only first param documented
    # Expected: x → TP, y → TN (audit NOT_MENTIONED, GT False)
    # -----------------------------------------------------------------------
    {
        "case_id": "B",
        "name": "Partially documented — first param present, second omitted",
        "source": """\
def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("Cannot divide by zero")
    return numerator / denominator
""",
        "markdown": """\
## divide

Divides numerator by denominator.

```python
def divide(numerator: float, denominator: float) -> float:
```

### Parameters
- **numerator** (float): The dividend value.

### Returns
float: Result of division.

### Raises
- **ValueError**: If denominator is zero.
""",
        "function": "divide",
        "labeled_params": [("numerator", True), ("denominator", False)],
    },
    # -----------------------------------------------------------------------
    # Case C: Completely undocumented parameters (no Parameters section)
    # Expected: all TN (audit NOT_MENTIONED, GT False)
    # -----------------------------------------------------------------------
    {
        "case_id": "C",
        "name": "No Parameters section at all — all params undocumented",
        "source": """\
def connect(host: str, port: int, timeout: float = 30.0) -> bool:
    return True
""",
        "markdown": """\
## connect

Attempts to connect to the specified server.

```python
def connect(host: str, port: int, timeout: float = 30.0) -> bool:
```

### Returns
bool: True if connection succeeded.
""",
        "function": "connect",
        "labeled_params": [("host", False), ("port", False), ("timeout", False)],
    },
    # -----------------------------------------------------------------------
    # Case D: Param mentioned only in the signature code block, NOT in Parameters section.
    # Human ground truth: NOT documented in the Parameters section → False.
    # Expected: all TN (audit NOT_MENTIONED, GT False) — correct rejection of code-block text.
    # -----------------------------------------------------------------------
    {
        "case_id": "D",
        "name": "Param only in code block — not in Parameters section (should not count)",
        "source": """\
def transform(data: list, scale: float) -> list:
    return [x * scale for x in data]
""",
        "markdown": """\
## transform

Transforms data by scale.

```python
def transform(data: list, scale: float) -> list:
```

No parameters section provided.

### Returns
list: Scaled output.
""",
        "function": "transform",
        "labeled_params": [("data", False), ("scale", False)],
    },
    # -----------------------------------------------------------------------
    # Case E: Param word appears in prose (not a Parameters section entry).
    # 'mode' is mentioned in ordinary prose: "depends on mode of operation".
    # Ground truth: NOT documented as a parameter entry → False.
    # Expected: TN (audit correctly rejects prose mention as NOT_MENTIONED).
    # -----------------------------------------------------------------------
    {
        "case_id": "E",
        "name": "Param name in prose only — not a Parameters section entry",
        "source": """\
def process(data: list, mode: str) -> list:
    return data
""",
        "markdown": """\
## process

Processes data. The behavior depends on mode of operation chosen by the caller.

```python
def process(data: list, mode: str) -> list:
```

### Parameters
- **data** (list): Input list.

### Returns
list: Processed output.
""",
        "function": "process",
        "labeled_params": [("data", True), ("mode", False)],
    },
    # -----------------------------------------------------------------------
    # Case F: Async function with all three params documented.
    # Expected: all TP.
    # -----------------------------------------------------------------------
    {
        "case_id": "F",
        "name": "Async function — all params documented",
        "source": """\
async def fetch(url: str, retries: int, timeout: float) -> bytes:
    return b""
""",
        "markdown": """\
## fetch

Fetches a remote resource asynchronously.

```python
async def fetch(url: str, retries: int, timeout: float) -> bytes:
```

### Parameters
- **url** (str): The target URL.
- **retries** (int): Number of retry attempts.
- **timeout** (float): Request timeout in seconds.

### Returns
bytes: Raw response body.
""",
        "function": "fetch",
        "labeled_params": [("url", True), ("retries", True), ("timeout", True)],
    },
    # -----------------------------------------------------------------------
    # Case G: One param documented, two omitted.
    # Expected: first → TP, second and third → TN.
    # -----------------------------------------------------------------------
    {
        "case_id": "G",
        "name": "Three params, only first documented in Parameters section",
        "source": """\
def compress(data: bytes, level: int, method: str) -> bytes:
    return data
""",
        "markdown": """\
## compress

Compresses the input data.

```python
def compress(data: bytes, level: int, method: str) -> bytes:
```

### Parameters
- **data** (bytes): Raw data to compress.

### Returns
bytes: Compressed output.
""",
        "function": "compress",
        "labeled_params": [("data", True), ("level", False), ("method", False)],
    },
    # -----------------------------------------------------------------------
    # Case H: No-parameter (void-argument) function.
    # No parameters to evaluate — this case contributes 0 labeled instances
    # but confirms the engine handles parameter-free functions gracefully.
    # Excluded from matrix counts explicitly.
    # -----------------------------------------------------------------------
    {
        "case_id": "H",
        "name": "No-parameter function — excluded from parameter matrix",
        "source": """\
def get_version() -> str:
    return "1.0.0"
""",
        "markdown": """\
## get_version

Returns the current version string.

### Returns
str: The semantic version number.
""",
        "function": "get_version",
        "labeled_params": [],  # no params → excluded from CM counts
    },
]


def run_evaluation():
    """
    Run audit on each labeled case, collect per-parameter predictions,
    compare against ground-truth labels, and compute confusion matrix metrics.
    """
    all_results = []
    matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    per_param_records = []

    for case in LABELED_CASES:
        report = audit_documentation(
            source_code=case["source"],
            markdown_doc=case["markdown"],
            target_function=case["function"],
        )

        if not report.success:
            print(f"[WARN] Case {case['case_id']} audit failed: {report.summary.overall_status}")
            continue

        # Build a lookup: param_name → AuditStatus from findings
        param_findings = {}
        for finding in report.findings:
            if finding.category == "parameter":
                param_findings[finding.subject] = finding.status.value if hasattr(finding.status, "value") else str(finding.status)

        case_records = []
        for (param_name, gt_documented) in case["labeled_params"]:
            predicted_status = param_findings.get(param_name, "NOT_MENTIONED")
            predicted_mentioned = (predicted_status == "MENTIONED")

            if gt_documented and predicted_mentioned:
                cell = "TP"
            elif (not gt_documented) and (not predicted_mentioned):
                cell = "TN"
            elif (not gt_documented) and predicted_mentioned:
                cell = "FP"
            else:  # gt_documented and not predicted_mentioned
                cell = "FN"

            matrix[cell] += 1
            record = {
                "case_id": case["case_id"],
                "case_name": case["name"],
                "param": param_name,
                "gt_documented": gt_documented,
                "audit_status": predicted_status,
                "predicted_mentioned": predicted_mentioned,
                "cell": cell,
            }
            case_records.append(record)
            per_param_records.append(record)

        all_results.append({
            "case_id": case["case_id"],
            "case_name": case["name"],
            "overall_status": report.summary.overall_status,
            "coverage_ratio": report.summary.parameters_coverage_ratio,
            "per_param": case_records,
        })

    # Compute metrics
    TP, TN, FP, FN = matrix["TP"], matrix["TN"], matrix["FP"], matrix["FN"]
    total = TP + TN + FP + FN
    precision = TP / (TP + FP) if (TP + FP) > 0 else None
    recall = TP / (TP + FN) if (TP + FN) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
    accuracy = (TP + TN) / total if total > 0 else None

    return {
        "per_param_records": per_param_records,
        "case_results": all_results,
        "matrix": matrix,
        "total_param_instances": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def write_png_confusion_matrix(matrix: dict, metrics: dict, output_path: str) -> None:
    """
    Write a simple PNG confusion matrix image using only Python stdlib
    (struct + zlib — no matplotlib/PIL required).

    Renders a 2x2 confusion matrix grid with TP/FP/FN/TN cell values,
    axis labels, and metrics text using pure bytes.
    """
    import struct
    import zlib

    # ---- Image dimensions & colour palette ----
    W, H = 560, 480
    # Background: white
    BG = (248, 248, 248)
    # Cell backgrounds
    TP_COL = (200, 235, 200)   # green tint
    TN_COL = (200, 235, 200)   # green tint
    FP_COL = (245, 210, 210)   # red tint
    FN_COL = (245, 210, 210)   # red tint
    GRID = (100, 100, 100)
    TEXT = (20, 20, 20)
    TITLE_COL = (40, 40, 40)

    # Simple 5x7 pixel bitmapped font for digits, letters and common chars
    # Each char is a list of 7 rows × 5 cols bitfield (row = int, MSB left)
    FONT = {
        " ": [0b00000]*7,
        "0": [0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110],
        "1": [0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110],
        "2": [0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111],
        "3": [0b11111,0b00010,0b00100,0b00010,0b00001,0b10001,0b01110],
        "4": [0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010],
        "5": [0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110],
        "6": [0b01110,0b10000,0b10000,0b11110,0b10001,0b10001,0b01110],
        "7": [0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000],
        "8": [0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110],
        "9": [0b01110,0b10001,0b10001,0b01111,0b00001,0b10001,0b01110],
        "T": [0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100],
        "P": [0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000],
        "F": [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000],
        "N": [0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001],
        "A": [0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
        "C": [0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110],
        "D": [0b11110,0b10001,0b10001,0b10001,0b10001,0b10001,0b11110],
        "E": [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111],
        "G": [0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01111],
        "I": [0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110],
        "L": [0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111],
        "M": [0b10001,0b11011,0b10101,0b10001,0b10001,0b10001,0b10001],
        "O": [0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        "R": [0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001],
        "S": [0b01110,0b10001,0b10000,0b01110,0b00001,0b10001,0b01110],
        "U": [0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        "X": [0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001],
        "c": [0b00000,0b00000,0b01110,0b10000,0b10000,0b10001,0b01110],
        "d": [0b00001,0b00001,0b01101,0b10011,0b10001,0b10001,0b01111],
        "e": [0b00000,0b00000,0b01110,0b10001,0b11111,0b10000,0b01110],
        "i": [0b00100,0b00000,0b00100,0b00100,0b00100,0b00100,0b00100],
        "l": [0b01100,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110],
        "n": [0b00000,0b00000,0b11110,0b10001,0b10001,0b10001,0b10001],
        "o": [0b00000,0b00000,0b01110,0b10001,0b10001,0b10001,0b01110],
        "p": [0b00000,0b00000,0b11110,0b10001,0b11110,0b10000,0b10000],
        "r": [0b00000,0b00000,0b10110,0b11001,0b10000,0b10000,0b10000],
        "s": [0b00000,0b00000,0b01111,0b10000,0b01110,0b00001,0b11110],
        "t": [0b00100,0b00100,0b11111,0b00100,0b00100,0b00101,0b00010],
        "u": [0b00000,0b00000,0b10001,0b10001,0b10001,0b10011,0b01101],
        ":": [0b00000,0b00100,0b00000,0b00000,0b00100,0b00000,0b00000],
        ".": [0b00000,0b00000,0b00000,0b00000,0b00000,0b00100,0b00000],
        "/": [0b00001,0b00001,0b00010,0b00100,0b01000,0b10000,0b10000],
        "%": [0b11000,0b11001,0b00010,0b00100,0b01000,0b10011,0b00011],
        "-": [0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000],
        "=": [0b00000,0b00000,0b11111,0b00000,0b11111,0b00000,0b00000],
        "(": [0b00010,0b00100,0b01000,0b01000,0b01000,0b00100,0b00010],
        ")": [0b01000,0b00100,0b00010,0b00010,0b00010,0b00100,0b01000],
        "1": [0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110],
    }

    # ---- Pixel buffer ----
    pixels = [[list(BG) for _ in range(W)] for _ in range(H)]

    def fill_rect(x0, y0, x1, y1, color):
        for y in range(max(0, y0), min(H, y1)):
            for x in range(max(0, x0), min(W, x1)):
                pixels[y][x] = list(color)

    def draw_rect_outline(x0, y0, x1, y1, color, thickness=2):
        for t in range(thickness):
            for x in range(x0, x1):
                if 0 <= y0+t < H: pixels[y0+t][x] = list(color)
                if 0 <= y1-1-t < H: pixels[y1-1-t][x] = list(color)
            for y in range(y0, y1):
                if 0 <= x0+t < W: pixels[y][x0+t] = list(color)
                if 0 <= x1-1-t < W: pixels[y][x1-1-t] = list(color)

    def draw_char(char, cx, cy, color, scale=2):
        bitmap = FONT.get(char.upper() if char.upper() in FONT else char, FONT.get(char, FONT[" "]))
        for row_idx, row in enumerate(bitmap):
            for col_idx in range(5):
                if row & (1 << (4 - col_idx)):
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cx + col_idx * scale + sx
                            py = cy + row_idx * scale + sy
                            if 0 <= px < W and 0 <= py < H:
                                pixels[py][px] = list(color)

    def draw_text(text, x, y, color=(20,20,20), scale=2):
        cx = x
        for ch in text:
            draw_char(ch, cx, y, color, scale)
            cx += 5 * scale + 1

    def text_width(text, scale=2):
        return len(text) * (5 * scale + 1)

    def draw_text_centered(text, cx_center, y, color=(20,20,20), scale=2):
        w = text_width(text, scale)
        draw_text(text, cx_center - w // 2, y, color, scale)

    # ---- Layout ----
    MARGIN = 30
    TITLE_H = 60
    LABEL_W = 90
    CELL_W = (W - MARGIN * 2 - LABEL_W) // 2
    CELL_H = 110
    GRID_TOP = MARGIN + TITLE_H + 40
    METRICS_TOP = GRID_TOP + 2 * CELL_H + 30

    TP_VAL = matrix["TP"]
    TN_VAL = matrix["TN"]
    FP_VAL = matrix["FP"]
    FN_VAL = matrix["FN"]

    # Title
    draw_text_centered("AUDIT ENGINE CONFUSION MATRIX", W // 2, MARGIN, TITLE_COL, scale=2)
    draw_text_centered("(Parameter Mention Detection)", W // 2, MARGIN + 20, (80,80,80), scale=1)

    # Column header labels
    col_labels_x = MARGIN + LABEL_W
    draw_text_centered("Pred: MENTIONED", col_labels_x + CELL_W // 2, GRID_TOP - 20, GRID, scale=1)
    draw_text_centered("Pred: NOT MENT.", col_labels_x + CELL_W + CELL_W // 2, GRID_TOP - 20, GRID, scale=1)

    # Row header labels
    row1_cy = GRID_TOP + CELL_H // 2 - 7
    row2_cy = GRID_TOP + CELL_H + CELL_H // 2 - 7
    draw_text_centered("GT: True", MARGIN + LABEL_W // 2, row1_cy, GRID, scale=1)
    draw_text_centered("GT: False", MARGIN + LABEL_W // 2, row2_cy, GRID, scale=1)

    # Cells: [GT=True, Pred=MENT] = TP, [GT=True, Pred=NOT] = FN
    #        [GT=False, Pred=MENT]= FP, [GT=False, Pred=NOT]= TN
    cells = [
        (0, 0, TP_VAL, "TP", TP_COL),
        (0, 1, FN_VAL, "FN", FN_COL),
        (1, 0, FP_VAL, "FP", FP_COL),
        (1, 1, TN_VAL, "TN", TN_COL),
    ]
    for row, col, val, label, bg in cells:
        x0 = MARGIN + LABEL_W + col * CELL_W
        y0 = GRID_TOP + row * CELL_H
        fill_rect(x0, y0, x0 + CELL_W, y0 + CELL_H, bg)
        draw_rect_outline(x0, y0, x0 + CELL_W, y0 + CELL_H, GRID, 2)
        # Cell label (TP/FP/FN/TN)
        draw_text_centered(label, x0 + CELL_W // 2, y0 + 14, (80, 80, 80), scale=1)
        # Big number
        draw_text_centered(str(val), x0 + CELL_W // 2, y0 + 30, TEXT, scale=4)

    # Metrics text
    metrics_lines = [
        f"Total instances (params) = {TP_VAL + TN_VAL + FP_VAL + FN_VAL}",
        f"TP={TP_VAL}  TN={TN_VAL}  FP={FP_VAL}  FN={FN_VAL}",
        f"Precision = {metrics['precision']:.3f}" if metrics["precision"] is not None else "Precision = N/A",
        f"Recall    = {metrics['recall']:.3f}" if metrics["recall"] is not None else "Recall = N/A",
        f"F1 Score  = {metrics['f1']:.3f}" if metrics["f1"] is not None else "F1 = N/A",
        f"Accuracy  = {metrics['accuracy']:.3f}" if metrics["accuracy"] is not None else "Accuracy = N/A",
    ]
    for i, line in enumerate(metrics_lines):
        draw_text(line, MARGIN + LABEL_W, METRICS_TOP + i * 16, TEXT, scale=1)

    # Axis labels
    draw_text_centered("Predicted Label", W // 2, GRID_TOP - 35, (50, 50, 120), scale=1)
    draw_text("True Label", MARGIN, GRID_TOP + CELL_H - 10, (50, 50, 120), scale=1)

    # ---- Write PNG ----
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # PNG header
    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit depth 8, colour type 2 (RGB), compression 0, filter 0, interlace 0
    ihdr_data = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # IDAT: raw scanlines with filter byte 0 prepended per row
    raw_rows = []
    for row in pixels:
        row_bytes = b"\x00"  # filter none
        for px in row:
            row_bytes += bytes(px)
        raw_rows.append(row_bytes)
    raw_data = b"".join(raw_rows)
    compressed = zlib.compress(raw_data, 9)
    idat = png_chunk(b"IDAT", compressed)

    iend = png_chunk(b"IEND", b"")

    with open(output_path, "wb") as f:
        f.write(header + ihdr + idat + iend)


def main():
    print("=" * 65)
    print("CODEscribe Audit Engine — Parameter Mention Confusion Matrix")
    print("=" * 65)
    print()

    result = run_evaluation()

    matrix = result["matrix"]
    TP, TN, FP, FN = matrix["TP"], matrix["TN"], matrix["FP"], matrix["FN"]
    total = result["total_param_instances"]
    precision = result["precision"]
    recall = result["recall"]
    f1 = result["f1"]
    accuracy = result["accuracy"]

    # Print per-param table
    print(f"{'Case':>5}  {'Param':<14}  {'GT':>10}  {'Audit':>14}  {'Cell':>4}")
    print("-" * 58)
    for rec in result["per_param_records"]:
        gt_str = "documented" if rec["gt_documented"] else "not documented"
        print(f"{rec['case_id']:>5}  {rec['param']:<14}  {gt_str:>14}  {rec['audit_status']:>14}  {rec['cell']:>4}")

    print()
    print("Confusion Matrix (Parameter Mention Detection):")
    print("                  Pred MENTIONED   Pred NOT_MENTIONED")
    print(f"  GT documented:       {TP:>4}               {FN:>4}")
    print(f"  GT not documented:   {FP:>4}               {TN:>4}")
    print()
    print(f"  Total labeled param instances: {total}")
    print(f"  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print()
    if precision is not None:
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  Accuracy  : {accuracy:.4f}")
    else:
        print("  Precision, Recall, F1: insufficient TP+FP or TP+FN counts.")

    print()
    print("LIMITATIONS:")
    print("  - This evaluates parameter-mention detection ONLY, not full audit coverage.")
    print("  - Ground-truth labels were independently specified by the evaluator from")
    print("    visible Markdown text; they do NOT derive from workbook runner outputs,")
    print("    ROUGE/BLEU scores, or CodeT5 generation predictions.")
    print("  - The audit engine uses heuristic lexical matching (not semantic NLP).")
    print("  - Sample size is small (labeled parameter instances from 7 designed cases).")
    print("  - This is NOT a measure of CodeT5 model accuracy.")

    # Save JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "confusion_matrix_results.json")
    save_payload = {
        "evaluation_target": "app/audit.py — parameter mention detection (heuristic lexical)",
        "positive_definition": "Parameter IS genuinely documented in the Markdown Parameters section (GT=True, Audit=MENTIONED → TP)",
        "negative_definition": "Parameter IS NOT in the Markdown Parameters section (GT=False, Audit=NOT_MENTIONED → TN)",
        "labeled_param_instances": total,
        "matrix": matrix,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "per_param_records": result["per_param_records"],
        "disclaimer": (
            "This evaluation assesses ONLY the audit engine's heuristic parameter-mention "
            "detection accuracy. It does NOT measure CodeT5 generation quality, ROUGE/BLEU, "
            "or semantic documentation correctness."
        ),
    }
    with open(json_path, "w") as f:
        json.dump(save_payload, f, indent=2)
    print(f"\n  JSON saved to: {json_path}")

    # Save PNG
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    write_png_confusion_matrix(
        matrix,
        {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy},
        png_path,
    )
    print(f"  PNG saved to:  {png_path}")

    return save_payload


if __name__ == "__main__":
    main()
