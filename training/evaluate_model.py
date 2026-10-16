#!/usr/bin/env python3
"""
training/evaluate_model.py

Held-out test set evaluation for CODEscribe:
Evaluates a fine-tuned CodeT5 model on the held-out test partition
without training, weight updates, or checkpoint selection leakage.

Key Architecture:
- Test-data isolation: Strictly loads only the specified test partition.
- Leakage-safe: Applies the same AST docstring stripping and task prefix as training.
- Local checkpoint verification: Requires all model and tokenizer files locally;
  never downloads or substitutes models silently.
- Inference-only: Disables gradients (torch.no_grad()) and operates in model.eval() mode.
- Comprehensive metrics: Computes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU via evaluate library.
- Traceable artifacts: Generates test_evaluation.json report and predictions.csv.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add workspace root to sys.path
_workspace_root = Path(__file__).resolve().parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from training.data_pipeline import prepare_leakage_safe_dataframe
from training.train import compute_metrics_text, set_reproducible_seed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CODEscribe.Evaluate")


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for held-out test evaluation."""
    parser = argparse.ArgumentParser(
        description="CODEscribe Held-Out Test Evaluation Pipeline"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/pilot_run_accelerate16/model",
        help="Path to the saved local fine-tuned model directory (default: outputs/pilot_run_accelerate16/model).",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="data/processed/test.csv",
        help="Path to the held-out test CSV partition (default: data/processed/test.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation_pilot_accelerate16",
        help="Directory to save evaluation report and predictions (default: outputs/evaluation_pilot_accelerate16).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional upper bound on the number of eligible test samples to evaluate (default: None, evaluate all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference generation (default: 16).",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=512,
        help="Maximum input sequence length in tokens (default: 512).",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum generated sequence length in tokens (default: 128).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1 for deterministic greedy decoding).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cuda', 'cpu', or None for auto-detection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(args)


def verify_local_model_path(model_path: Path) -> None:
    """
    Verify that the specified local model directory exists and contains
    the essential model weights and configuration files.

    Raises:
        FileNotFoundError: If the directory or mandatory checkpoint files are missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Specified model path does not exist: {model_path.resolve()}\n"
            "Evaluation requires an existing local fine-tuned checkpoint. "
            "Silent downloading or model substitution is prohibited."
        )

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Specified model path is not a directory: {model_path.resolve()}"
        )

    # Check for configuration file
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Missing required model configuration file: {config_file.resolve()}"
        )

    # Check for model weights file (safetensors or pytorch_model.bin)
    safetensors_file = model_path / "model.safetensors"
    bin_file = model_path / "pytorch_model.bin"
    if not safetensors_file.exists() and not bin_file.exists():
        raise FileNotFoundError(
            f"No valid weights file found in {model_path.resolve()}.\n"
            f"Expected 'model.safetensors' or 'pytorch_model.bin'."
        )

    logger.info("Verified local checkpoint integrity at: %s", model_path.resolve())


def load_model_and_tokenizer(
    model_path_str: str,
    device_override: Optional[str] = None,
) -> Tuple[Any, Any, str]:
    """
    Load fine-tuned model and tokenizer strictly from local files.

    Returns:
        Tuple of (model, tokenizer, device_name).
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(model_path_str)
    verify_local_model_path(model_path)

    # Determine device
    if device_override:
        device = device_override
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer from local directory: %s", model_path.resolve())
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    logger.info("Loading model from local directory: %s", model_path.resolve())
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()

    # Note on tied weights:
    # CodeT5 uses tied embeddings where shared.weight, encoder.embed_tokens.weight,
    # decoder.embed_tokens.weight, and lm_head.weight refer to the exact same tensor memory.
    # The 'missing keys' notice during from_pretrained reflects this architectural design.
    if hasattr(model, "shared") and hasattr(model, "lm_head"):
        is_tied = model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
        logger.info("Tied weights verification: shared and lm_head memory identical = %s", is_tied)

    return model, tokenizer, device


def generate_predictions_batch(
    model: Any,
    tokenizer: Any,
    input_texts: List[str],
    device: str,
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_beams: int = 1,
) -> List[str]:
    """
    Generate documentation strings for a batch of docstring-free source code inputs.
    Inference mode with disabled gradients.
    """
    import torch

    inputs = tokenizer(
        input_texts,
        max_length=max_source_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = pad_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_target_length,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def evaluate_model(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute full held-out test partition evaluation:
    1. Loads test partition strictly.
    2. Applies leakage-safe docstring removal and task prefix.
    3. Evaluates eligible examples under deterministic generation settings.
    4. Computes ROUGE and BLEU metrics.
    5. Persists evaluation report and per-example predictions.
    """
    set_reproducible_seed(args.seed)
    start_time = time.time()

    # 1. Verify test data exists and isolate from train/val
    test_path = Path(args.test_data_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test partition missing at: {test_path.resolve()}")

    logger.info("Loading held-out test partition from: %s", test_path.resolve())
    df_test_raw = pd.read_csv(test_path)
    total_test_rows = len(df_test_raw)

    # 2. Apply leakage-safe transformation
    logger.info("Applying AST docstring stripping and task prefix to test partition...")
    clean_test_df, audit_meta = prepare_leakage_safe_dataframe(
        df_test_raw,
        source_col="func_code_string",
        target_col="func_documentation_string",
        task_prefix="document python code: ",
    )
    eligible_test_rows = len(clean_test_df)
    rejected_rows = audit_meta["total_rejected"]
    rejection_reasons = audit_meta["rejected_reasons"]

    logger.info(
        "Test dataset inspection: %d total rows, %d eligible rows, %d rejected (%s).",
        total_test_rows,
        eligible_test_rows,
        rejected_rows,
        rejection_reasons,
    )

    # 3. Apply sample limit if requested
    if args.sample_limit is not None and args.sample_limit > 0:
        if args.sample_limit < eligible_test_rows:
            logger.info("Applying sample limit: evaluating first %d eligible samples.", args.sample_limit)
            eval_df = clean_test_df.iloc[:args.sample_limit].copy()
        else:
            logger.info("Sample limit (%d) exceeds eligible count (%d); evaluating all eligible.",
                        args.sample_limit, eligible_test_rows)
            eval_df = clean_test_df.copy()
    else:
        eval_df = clean_test_df.copy()

    evaluated_rows = len(eval_df)

    # 4. Load Model & Tokenizer locally
    model, tokenizer, device_used = load_model_and_tokenizer(
        model_path_str=args.model_path,
        device_override=args.device,
    )
    logger.info("Using device: %s for inference batch size: %d", device_used, args.batch_size)

    # 5. Batch Generation
    predictions: List[str] = []
    references: List[str] = eval_df["target_text"].tolist()
    input_texts: List[str] = eval_df["input_text"].tolist()

    logger.info("Starting generation on %d samples in batches of %d...", evaluated_rows, args.batch_size)
    batch_start = time.time()
    for i in range(0, evaluated_rows, args.batch_size):
        batch_inputs = input_texts[i : i + args.batch_size]
        batch_preds = generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            input_texts=batch_inputs,
            device=device_used,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            num_beams=args.num_beams,
        )
        predictions.extend(batch_preds)
        if (i // args.batch_size + 1) % 10 == 0 or (i + len(batch_inputs)) == evaluated_rows:
            elapsed = time.time() - batch_start
            logger.info("Generated %d / %d samples (%.2f s elapsed)...",
                        min(i + len(batch_inputs), evaluated_rows), evaluated_rows, elapsed)

    # 6. Compute Metrics
    logger.info("Computing evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU)...")
    metric_results = compute_metrics_text(predictions=predictions, references=references)
    metric_engine = "evaluate (Hugging Face)" if "bleu_note" not in metric_results else "pure-python fallback"

    # 7. Build Predictions DataFrame
    # Construct a complete record for all test rows
    evaluated_indices = set(eval_df["original_index"])
    records: List[Dict[str, Any]] = []

    # Map evaluated rows
    pred_map = {row["original_index"]: pred for row, pred in zip(eval_df.to_dict("records"), predictions)}

    for idx, row in df_test_raw.iterrows():
        orig_idx = int(idx)
        repo = row.get("repository_name", "N/A")
        func = row.get("func_name", "N/A")
        ref_doc = str(row.get("func_documentation_string", "")).strip()

        if orig_idx in evaluated_indices:
            records.append({
                "test_row_index": orig_idx,
                "repository_name": repo,
                "func_name": func,
                "reference_documentation": ref_doc,
                "generated_documentation": pred_map.get(orig_idx, ""),
                "status": "EVALUATED",
            })
        elif orig_idx in clean_test_df["original_index"].values:
            records.append({
                "test_row_index": orig_idx,
                "repository_name": repo,
                "func_name": func,
                "reference_documentation": ref_doc,
                "generated_documentation": "",
                "status": "SKIPPED_SAMPLE_LIMIT",
            })
        else:
            records.append({
                "test_row_index": orig_idx,
                "repository_name": repo,
                "func_name": func,
                "reference_documentation": ref_doc,
                "generated_documentation": "",
                "status": "REJECTED_SYNTAX_ERROR",
            })

    predictions_df = pd.DataFrame(records)

    # 8. Persist Artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "predictions.csv"
    predictions_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Saved per-example predictions to: %s", csv_path.resolve())

    total_duration = round(time.time() - start_time, 2)
    samples_per_sec = round(evaluated_rows / total_duration, 2) if total_duration > 0 else 0.0

    report: Dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_script": "training/evaluate_model.py",
            "model_path": str(Path(args.model_path).resolve()).replace("\\", "/"),
            "test_data_path": str(Path(args.test_data_path).resolve()).replace("\\", "/"),
            "device": device_used,
            "seed": args.seed,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "dataset_counts": {
            "total_test_rows": total_test_rows,
            "eligible_test_rows": eligible_test_rows,
            "evaluated_rows": evaluated_rows,
            "rejected_rows": rejected_rows,
            "rejection_reasons": rejection_reasons,
            "skipped_sample_limit_rows": eligible_test_rows - evaluated_rows,
        },
        "generation_parameters": {
            "batch_size": args.batch_size,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "num_beams": args.num_beams,
            "do_sample": False,
            "task_prefix": "document python code: ",
        },
        "metrics": {
            "rouge1": metric_results.get("rouge1", 0.0),
            "rouge2": metric_results.get("rouge2", 0.0),
            "rougeL": metric_results.get("rougeL", 0.0),
            "bleu": metric_results.get("bleu", 0.0),
            "metric_scale": "percentage (0 to 100)",
            "metric_engine": metric_engine,
        },
        "runtime": {
            "duration_seconds": total_duration,
            "samples_per_second": samples_per_sec,
        },
        "architecture_verification": {
            "tied_weights_note": (
                "CodeT5 ties encoder.embed_tokens, decoder.embed_tokens, and lm_head "
                "to shared.weight. The missing-keys warning on load is standard T5 behavior."
            ),
        },
    }

    json_path = output_dir / "test_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to: %s", json_path.resolve())

    return report


def main() -> int:
    args = parse_arguments()
    try:
        report = evaluate_model(args)
        print("\n" + "=" * 70)
        print("  CODESCRIBE HELD-OUT TEST EVALUATION COMPLETED")
        print("=" * 70)
        print(f"  Model Evaluated        : {args.model_path}")
        print(f"  Test Partition         : {args.test_data_path}")
        print(f"  Total Test Rows        : {report['dataset_counts']['total_test_rows']}")
        print(f"  Eligible Test Rows     : {report['dataset_counts']['eligible_test_rows']}")
        print(f"  Evaluated Rows         : {report['dataset_counts']['evaluated_rows']}")
        print(f"  Rejected Rows          : {report['dataset_counts']['rejected_rows']}")
        print(f"  ROUGE-1                : {report['metrics']['rouge1']:.2f}")
        print(f"  ROUGE-2                : {report['metrics']['rouge2']:.2f}")
        print(f"  ROUGE-L                : {report['metrics']['rougeL']:.2f}")
        print(f"  BLEU                   : {report['metrics']['bleu']:.2f} ({report['metrics']['metric_scale']})")
        print(f"  Metric Engine          : {report['metrics']['metric_engine']}")
        print(f"  Runtime                : {report['runtime']['duration_seconds']}s ({report['runtime']['samples_per_second']} samples/s)")
        print(f"  Report JSON            : {args.output_dir}/test_evaluation.json")
        print(f"  Predictions CSV        : {args.output_dir}/predictions.csv")
        print("=" * 70 + "\n")
        return 0
    except Exception as exc:
        logger.error("Held-out test evaluation failed: %s", exc, exc_info=True)
        print(f"\n[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
