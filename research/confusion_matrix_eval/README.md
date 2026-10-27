# Audit Engine Confusion Matrix Evaluation

## Overview

This directory contains a standalone confusion matrix evaluation of `app/audit.py`'s
**parameter mention detection** capability. It is entirely separate from:

- The 20-case workbook evaluation in `research/audit_evaluation/`
- The CodeT5 model, ROUGE/BLEU metrics, or any generation quality measurement

## Task Framing

**Binary classification task:** For each source-code parameter in a function, does the
audit engine correctly determine whether it is mentioned in the Markdown Parameters section?

| | Audit predicts MENTIONED | Audit predicts NOT_MENTIONED |
|---|---|---|
| **GT: documented** | TP | FN |
| **GT: not documented** | FP | TN |

Ground-truth labels were **independently specified** by visual inspection of the Markdown
text. They do not derive from audit output, workbook runner results, ROUGE/BLEU, or model predictions.

## Cases

| Case | Description | Params |
|------|-------------|--------|
| A | Fully documented — both params in Parameters section | 2 |
| B | Partial — first documented, second omitted | 2 |
| C | No Parameters section at all | 3 |
| D | Params appear only in code block (should NOT count as documented) | 2 |
| E | One param documented; one mentioned only in prose (should NOT count) | 2 |
| F | Async function — all 3 params documented | 3 |
| G | 3 params, only first documented | 3 |
| H | No-parameter function (excluded from matrix) | 0 |

**Total labeled parameter instances: 17**

## Results (Actual Output)

```
                  Pred MENTIONED   Pred NOT_MENTIONED
GT documented:          8                  0
GT not documented:      0                  9

TP=8, TN=9, FP=0, FN=0

Precision : 1.0000
Recall    : 1.0000
F1 Score  : 1.0000
Accuracy  : 1.0000
```

## Files

- `confusion_matrix_runner.py` — Standalone runner; no existing files modified
- `confusion_matrix.png` — PNG confusion matrix image (stdlib-only, no matplotlib)
- `confusion_matrix_results.json` — Machine-readable results with per-param records

## Reproduction

```powershell
$env:PYTHONPATH = "D:\MCA\CODEscribe"
python research/confusion_matrix_eval/confusion_matrix_runner.py
```

Requires Python 3.12.x with the CODEscribe virtual environment activated.

## Limitations

> [!IMPORTANT]
> This evaluation has significant scope limitations that must be understood before use:

1. **Heuristic lexical matching only** — `app/audit.py` uses regex-based section parsing,
   not semantic NLP. The perfect score reflects structural clarity of the test cases, not
   deep language understanding.
2. **Small sample size** — 17 labeled parameter instances across 7 designed cases.
   This is insufficient to draw statistically robust conclusions.
3. **Designed test cases** — Cases were constructed to exercise specific behaviors of the
   audit engine; they do not represent a random sample of real documentation.
4. **Parameter detection only** — Returns, exceptions, and generator facts are NOT
   evaluated in this matrix.
5. **NOT a CodeT5 accuracy metric** — This measures the audit engine's detection of whether
   parameters appear in documentation, not the quality of generated documentation or any
   model output.
6. **No version of the source analyzed** — Evaluated against the current uncommitted
   working-tree version of `app/audit.py`.
