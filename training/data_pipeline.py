"""
training/data_pipeline.py

Data ingestion, leakage-free feature preparation, tokenization,
and dynamic collation pipeline for fine-tuning CodeT5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from training.remove_docstrings import remove_docstrings_from_source
from training.training_config import TrainingExperimentConfig

logger = logging.getLogger("CODEscribe.DataPipeline")


def prepare_leakage_safe_dataframe(
    df: pd.DataFrame,
    source_col: str = "func_code_string",
    target_col: str = "func_documentation_string",
    task_prefix: str = "document python code: ",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transform raw code-documentation pairs into leakage-safe model inputs:
    - Strips inline docstrings from source code using AST spans.
    - Applies multi-tier leakage verification.
    - Appends the standard task prefix.
    - Rejects malformed or leaking examples with documented justifications.

    Args:
        df: Input DataFrame containing source and target columns.
        source_col: Column name containing raw source code.
        target_col: Column name containing target documentation.
        task_prefix: Prompt prefix prepended to docstring-free code.

    Returns:
        Tuple of (clean_df, audit_metrics_dict).
    """
    total_records = len(df)
    clean_records: List[Dict[str, Any]] = []
    rejected_counts: Dict[str, int] = {}
    sample_rejections: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        code = str(row[source_col])
        doc = str(row[target_col]).strip()

        clean_code, status, meta = remove_docstrings_from_source(code, target_doc=doc)

        if status in ("SUCCESS", "NO_DOCSTRING_PRESENT") and clean_code is not None:
            clean_records.append({
                "original_index": int(idx),
                "repository_name": row.get("repository_name", "N/A"),
                "func_name": row.get("func_name", "N/A"),
                "input_text": f"{task_prefix}{clean_code}",
                "target_text": doc,
                "clean_code_chars": len(clean_code),
                "target_chars": len(doc),
            })
        else:
            rejected_counts[status] = rejected_counts.get(status, 0) + 1
            if len(sample_rejections) < 10:
                sample_rejections.append({
                    "original_index": int(idx),
                    "func_name": row.get("func_name", "N/A"),
                    "status": status,
                    "meta": meta,
                })

    clean_df = pd.DataFrame(clean_records)
    total_retained = len(clean_df)
    total_rejected = sum(rejected_counts.values())

    audit_meta: Dict[str, Any] = {
        "total_records_examined": total_records,
        "successfully_transformed": total_retained,
        "retained_pct": round((total_retained / total_records) * 100, 2) if total_records > 0 else 0.0,
        "total_rejected": total_rejected,
        "rejected_reasons": rejected_counts,
        "sample_rejections": sample_rejections,
    }

    logger.info(
        "Leakage-safe transformation complete: %d / %d records retained (%.2f%%). Rejections: %s",
        total_retained,
        total_records,
        audit_meta["retained_pct"],
        rejected_counts,
    )

    return clean_df, audit_meta


class CodeDocumentationDataset:
    """
    Dataset wrapper holding preprocessed input_text and target_text pairs.
    Compatible with PyTorch Dataset when torch is present.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if "input_text" not in data.columns or "target_text" not in data.columns:
            raise ValueError("DataFrame must contain 'input_text' and 'target_text' columns.")
        self.inputs = data["input_text"].tolist()
        self.targets = data["target_text"].tolist()
        self.metadata = data.drop(columns=["input_text", "target_text"], errors="ignore").to_dict("records")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_text": self.inputs[idx],
            "target_text": self.targets[idx],
            **self.metadata[idx],
        }


class DataCollatorForCodeT5:
    """
    Dynamic batch collator for sequence-to-sequence CodeT5 models:
    - Encodes input source code and target documentation.
    - Dynamically pads to batch maximum (or truncates to max_length).
    - Masks target pad tokens with -100 so CrossEntropyLoss ignores them.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_source_length: int = 512,
        max_target_length: int = 128,
        padding: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        input_texts = [ex["input_text"] for ex in batch]
        target_texts = [ex["target_text"] for ex in batch]

        model_inputs = self.tokenizer(
            input_texts,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            text_target=target_texts,
            max_length=self.max_target_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        labels_tensor = labels["input_ids"]
        # Replace padding token id with -100 to ignore loss on padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels_tensor = labels_tensor.masked_fill(
                labels_tensor == self.tokenizer.pad_token_id, -100
            )

        model_inputs["labels"] = labels_tensor
        return model_inputs
