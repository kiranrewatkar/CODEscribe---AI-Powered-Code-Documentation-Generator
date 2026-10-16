#!/usr/bin/env python3
"""
training/prepare_dataset.py

Production-grade dataset validation, cleaning, deduplication, and
leakage-resistant repository-level splitting pipeline for CODEscribe.

Produces:
- data/processed/train.csv
- data/processed/validation.csv
- data/processed/test.csv
- data/processed/dataset_manifest.json
- outputs/preprocessing_report.txt
- outputs/preprocessing_report.json

Usage:
    python training/prepare_dataset.py
    python training/prepare_dataset.py --raw-csv data/raw/codesearchnet_10000_python_examples.csv --seed 42
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Ensure workspace root and package are on sys.path regardless of launch method
_current_file_dir = Path(__file__).resolve().parent
_workspace_root = _current_file_dir.parent
for _p in [str(_workspace_root), str(_current_file_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from training.config import PreprocessingConfig
except ModuleNotFoundError:
    from config import PreprocessingConfig


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    """Configure structured logging for dataset preparation."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger("CODEscribe.PrepareDataset")


# ---------------------------------------------------------------------------
# File Utilities & Integrity Checks
# ---------------------------------------------------------------------------
def compute_file_sha256(file_path: Path) -> str:
    """
    Compute the SHA-256 cryptographic hash of a file for audit provenance.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Target file does not exist: {file_path.resolve()}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    logger.debug("Computed SHA-256 for %s: %s", file_path.name, digest)
    return digest


# ---------------------------------------------------------------------------
# Stage 1: Schema Validation
# ---------------------------------------------------------------------------
def validate_schema(
    df: pd.DataFrame,
    required_columns: Tuple[str, ...],
    source_col: str,
    target_col: str,
    group_col: str,
) -> None:
    """
    Validate that the DataFrame contains required columns with expected types.

    Args:
        df: Input DataFrame.
        required_columns: Iterable of required column names.
        source_col: Source code column name.
        target_col: Target documentation column name.
        group_col: Repository grouping column name.

    Raises:
        ValueError: If essential columns are missing or DataFrame is structurally invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")

    if len(df) == 0:
        raise ValueError("Input dataset is completely empty (0 rows).")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed. Required columns missing from dataset: {missing}. "
            f"Existing columns: {df.columns.tolist()}"
        )

    for col in [source_col, target_col, group_col]:
        if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
            raise ValueError(
                f"Column '{col}' must contain string/object text data, but found dtype '{df[col].dtype}'."
            )

    logger.info("Schema validation successful. Verified columns: %s", list(required_columns))


# ---------------------------------------------------------------------------
# Stage 2: Conservative Normalization
# ---------------------------------------------------------------------------
def normalize_text(text: Any, normalize_crlf: bool = True, strip_outer: bool = True) -> str:
    """
    Conservatively normalize text fields.
    Preserves meaningful internal Python indentation and doc formatting.

    Args:
        text: Input string or object.
        normalize_crlf: If True, convert CRLF and CR to LF (\\n).
        strip_outer: If True, strip only leading/trailing whitespace.

    Returns:
        Normalized string.
    """
    if text is None or pd.isna(text):
        return ""

    s = str(text)

    if normalize_crlf:
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    if strip_outer:
        s = s.strip()

    return s


def normalize_dataframe(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    group_col: str,
    normalize_crlf: bool = True,
    strip_outer: bool = True,
) -> pd.DataFrame:
    """
    Apply conservative normalization across DataFrame columns.
    Creates a new DataFrame without mutating the original in-place.
    """
    logger.info("Normalizing fields: line ending normalization (CRLF -> LF) and outer stripping...")
    norm_df = df.copy()
    norm_df[source_col] = norm_df[source_col].apply(
        lambda x: normalize_text(x, normalize_crlf=normalize_crlf, strip_outer=strip_outer)
    )
    norm_df[target_col] = norm_df[target_col].apply(
        lambda x: normalize_text(x, normalize_crlf=normalize_crlf, strip_outer=strip_outer)
    )
    norm_df[group_col] = norm_df[group_col].apply(
        lambda x: normalize_text(x, normalize_crlf=normalize_crlf, strip_outer=strip_outer)
    )
    return norm_df


# ---------------------------------------------------------------------------
# Stage 3: Example Validation & Exclusion Accounting
# ---------------------------------------------------------------------------
def filter_and_validate_records(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    group_col: str,
    min_code_chars: int = 1,
    min_doc_chars: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, int], List[Dict[str, Any]]]:
    """
    Filter records according to documented rules.
    Tracks exact counts and justifications for every excluded record.

    Exclusion Rules:
    - REASON_EMPTY_OR_BLANK_SOURCE_CODE: Source code is empty or whitespace-only.
    - REASON_EMPTY_OR_BLANK_DOCUMENTATION: Documentation is empty or whitespace-only.
    - REASON_SOURCE_CODE_BELOW_MIN_LENGTH: Source code length < min_code_chars.
    - REASON_DOCUMENTATION_BELOW_MIN_LENGTH: Documentation length < min_doc_chars.
    - REASON_EMPTY_REPOSITORY_METADATA: Repository name is missing or blank.

    Returns:
        Tuple of (retained_df, exclusion_counts_dict, sample_exclusions_log).
    """
    logger.info("Validating examples against documented retention rules...")
    retained_indices: List[int] = []
    exclusion_counts: Dict[str, int] = {
        "REASON_EMPTY_OR_BLANK_SOURCE_CODE": 0,
        "REASON_EMPTY_OR_BLANK_DOCUMENTATION": 0,
        "REASON_SOURCE_CODE_BELOW_MIN_LENGTH": 0,
        "REASON_DOCUMENTATION_BELOW_MIN_LENGTH": 0,
        "REASON_EMPTY_REPOSITORY_METADATA": 0,
    }
    sample_exclusions: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        val_code = row[source_col]
        val_doc = row[target_col]
        val_repo = row[group_col]

        code = "" if (val_code is None or pd.isna(val_code)) else str(val_code).strip()
        doc = "" if (val_doc is None or pd.isna(val_doc)) else str(val_doc).strip()
        repo = "" if (val_repo is None or pd.isna(val_repo)) else str(val_repo).strip()

        rejection_reason: Optional[str] = None

        if len(code) == 0:
            rejection_reason = "REASON_EMPTY_OR_BLANK_SOURCE_CODE"
        elif len(code) < min_code_chars:
            rejection_reason = "REASON_SOURCE_CODE_BELOW_MIN_LENGTH"
        elif len(doc) == 0:
            rejection_reason = "REASON_EMPTY_OR_BLANK_DOCUMENTATION"
        elif len(doc) < min_doc_chars:
            rejection_reason = "REASON_DOCUMENTATION_BELOW_MIN_LENGTH"
        elif len(repo) == 0:
            rejection_reason = "REASON_EMPTY_REPOSITORY_METADATA"

        if rejection_reason is not None:
            exclusion_counts[rejection_reason] += 1
            if len(sample_exclusions) < 10:
                sample_exclusions.append({
                    "original_index": int(idx),
                    "reason": rejection_reason,
                    "code_len": len(code),
                    "doc_len": len(doc),
                    "repo": repo,
                })
        else:
            retained_indices.append(idx)

    retained_df = df.loc[retained_indices].copy().reset_index(drop=True)
    total_excluded = sum(exclusion_counts.values())
    logger.info(
        "Validation complete. Retained: %d records, Excluded: %d records.",
        len(retained_df),
        total_excluded,
    )
    return retained_df, exclusion_counts, sample_exclusions


# ---------------------------------------------------------------------------
# Stage 4: Deduplication & Conflict Resolution
# ---------------------------------------------------------------------------
def deduplicate_dataset(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    deduplicate_identical_code: bool = True,
    deduplicate_identical_pairs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identify and resolve duplicates across code, documentation, and pairs.
    Prevents duplicate source code from crossing partition boundaries.

    Deduplication Policy:
    1. Exact duplicate code-documentation pairs (identical code, identical doc):
       Retain first occurrence, prune redundant copies.
    2. Identical code with conflicting documentation (identical code, different doc):
       Investigated and logged. If deduplicate_identical_code=True, retains the first
       instance to eliminate code crossing across splits.
    3. Documented near-duplicate risk:
       Identical code tokens / boilerplate are noted without aggressive fuzzy deletion.

    Returns:
        Tuple of (deduplicated_df, deduplication_audit_metadata).
    """
    logger.info("Executing deduplication analysis and enforcement...")
    initial_count = len(df)

    # 1. Exact duplicate pairs
    dup_pairs_mask = df.duplicated(subset=[source_col, target_col], keep="first")
    exact_duplicate_pairs_count = int(dup_pairs_mask.sum())

    # 2. Conflicting documentation for identical code
    code_grouped = df.groupby(source_col)[target_col].nunique()
    conflicting_code_series = code_grouped[code_grouped > 1]
    conflicting_code_count = int(len(conflicting_code_series))

    # 3. Overall duplicate code count
    dup_code_mask = df.duplicated(subset=[source_col], keep="first")
    duplicate_code_count = int(dup_code_mask.sum())

    # Apply deduplication according to policy
    dedup_df = df.copy()
    if deduplicate_identical_code:
        # Deduplicating on code guarantees no code appears more than once anywhere
        dedup_df = dedup_df.drop_duplicates(subset=[source_col], keep="first").reset_index(drop=True)
    elif deduplicate_identical_pairs:
        dedup_df = dedup_df.drop_duplicates(subset=[source_col, target_col], keep="first").reset_index(drop=True)

    pruned_records_count = initial_count - len(dedup_df)

    audit_meta: Dict[str, Any] = {
        "initial_records_before_dedup": initial_count,
        "exact_duplicate_pairs_detected": exact_duplicate_pairs_count,
        "duplicate_code_strings_detected": duplicate_code_count,
        "identical_code_with_conflicting_doc_count": conflicting_code_count,
        "records_pruned_by_deduplication": pruned_records_count,
        "records_retained_after_dedup": len(dedup_df),
        "deduplication_policy": (
            "Deduplicate on exact normalized source code ('func_code_string'). "
            "Retain first occurrence to strictly guarantee zero duplicate code across partitions. "
            "Aggressive fuzzy / near-duplicate deletion is intentionally avoided to preserve legitimate "
            "domain-specific function variations."
        ),
    }

    logger.info(
        "Deduplication complete. Pruned %d records; %d unique records remain.",
        pruned_records_count,
        len(dedup_df),
    )
    return dedup_df, audit_meta


# ---------------------------------------------------------------------------
# Stage 5: Leakage-Resistant Group Splitting
# ---------------------------------------------------------------------------
def split_by_group(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    min_groups_required: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partition dataset into train, validation, and test splits grouped by repository.
    Guarantees strict repository isolation: no repository in one partition can ever appear in another.

    Uses a deterministic greedy knapsack allocation based on group counts:
    - Target counts computed from total rows and ratios.
    - Repositories shuffled deterministically with a fixed seed.
    - Test partition filled first up to target_test.
    - Validation partition filled next up to target_val.
    - Remaining repositories allocated to training partition.

    Args:
        df: Input DataFrame.
        group_col: Column name identifying group (e.g. repository_name).
        train_ratio: Proportion of data targeted for training.
        val_ratio: Proportion of data targeted for validation.
        test_ratio: Proportion of data targeted for testing.
        seed: Random seed for deterministic reproducibility.
        min_groups_required: Minimum independent groups required to perform group split.

    Returns:
        Tuple of (train_df, val_df, test_df, split_metadata_dict).

    Raises:
        ValueError: If repository metadata is too sparse or ratios are invalid.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' missing from DataFrame.")

    repo_counts = df[group_col].value_counts()
    unique_groups = len(repo_counts)

    if unique_groups < min_groups_required:
        raise ValueError(
            f"Insufficient independent repositories ({unique_groups} < {min_groups_required}). "
            "Cannot perform a statistically valid, leakage-resistant grouped split. Halting."
        )

    total_rows = len(df)
    if total_rows < 10:
        raise ValueError(f"Dataset has only {total_rows} rows; insufficient for 3-way partition.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.999 <= ratio_sum <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    logger.info(
        "Initiating grouped split on '%s' across %d unique repositories (seed=%d)...",
        group_col,
        unique_groups,
        seed,
    )

    target_test = int(round(total_rows * test_ratio))
    target_val = int(round(total_rows * val_ratio))
    target_train = total_rows - target_test - target_val

    # Deterministic shuffle of unique repository keys
    rng = np.random.default_rng(seed)
    repos = repo_counts.index.tolist()
    rng.shuffle(repos)

    test_repos: Set[str] = set()
    val_repos: Set[str] = set()
    train_repos: Set[str] = set()

    test_count = 0
    val_count = 0
    train_count = 0

    for repo in repos:
        count = int(repo_counts[repo])

        # Fill test bucket first if room exists
        if test_count + count <= target_test:
            test_repos.add(repo)
            test_count += count
        # Fill val bucket next if room exists
        elif val_count + count <= target_val:
            val_repos.add(repo)
            val_count += count
        # Everything else to train
        else:
            train_repos.add(repo)
            train_count += count

    # Safety check: ensure no split is completely starved if groups are available
    if len(test_repos) == 0 and len(repos) >= 3:
        # Move smallest train repo to test
        repo_to_move = repos[-1]
        train_repos.discard(repo_to_move)
        test_repos.add(repo_to_move)

    if len(val_repos) == 0 and len(repos) >= 3:
        # Move second smallest train repo to val
        repo_to_move = repos[-2]
        train_repos.discard(repo_to_move)
        val_repos.add(repo_to_move)

    train_df = df[df[group_col].isin(train_repos)].copy().reset_index(drop=True)
    val_df = df[df[group_col].isin(val_repos)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_repos)].copy().reset_index(drop=True)

    # Attach split_name metadata
    train_df["split_name"] = "train"
    val_df["split_name"] = "validation"
    test_df["split_name"] = "test"

    split_meta: Dict[str, Any] = {
        "strategy": "repository_grouped_split",
        "random_seed": seed,
        "group_column": group_col,
        "total_unique_groups": unique_groups,
        "targets": {
            "target_train_rows": target_train,
            "target_val_rows": target_val,
            "target_test_rows": target_test,
        },
        "actuals": {
            "train_rows": len(train_df),
            "train_pct": round((len(train_df) / total_rows) * 100, 2),
            "train_repos_count": len(train_repos),
            "val_rows": len(val_df),
            "val_pct": round((len(val_df) / total_rows) * 100, 2),
            "val_repos_count": len(val_repos),
            "test_rows": len(test_df),
            "test_pct": round((len(test_df) / total_rows) * 100, 2),
            "test_repos_count": len(test_repos),
        },
    }

    logger.info(
        "Split achieved: Train=%d (%.2f%%, %d repos), Val=%d (%.2f%%, %d repos), Test=%d (%.2f%%, %d repos)",
        len(train_df),
        split_meta["actuals"]["train_pct"],
        len(train_repos),
        len(val_df),
        split_meta["actuals"]["val_pct"],
        len(val_repos),
        len(test_df),
        split_meta["actuals"]["test_pct"],
        len(test_repos),
    )

    return train_df, val_df, test_df, split_meta


# ---------------------------------------------------------------------------
# Stage 6: Output Validation
# ---------------------------------------------------------------------------
def validate_split_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    group_col: str,
) -> Dict[str, Any]:
    """
    Rigorously validate split outputs:
    - Non-empty code and documentation in every row across all splits.
    - Mutually disjoint source code across all splits.
    - Mutually disjoint repositories across all splits.
    - Reopenability confirmation.
    """
    logger.info("Conducting post-split output integrity validation...")

    train_codes = set(train_df[source_col])
    val_codes = set(val_df[source_col])
    test_codes = set(test_df[source_col])

    code_overlap_train_val = len(train_codes & val_codes)
    code_overlap_train_test = len(train_codes & test_codes)
    code_overlap_val_test = len(val_codes & test_codes)
    total_code_overlap = code_overlap_train_val + code_overlap_train_test + code_overlap_val_test

    train_repos = set(train_df[group_col])
    val_repos = set(val_df[group_col])
    test_repos = set(test_df[group_col])

    repo_overlap_train_val = len(train_repos & val_repos)
    repo_overlap_train_test = len(train_repos & test_repos)
    repo_overlap_val_test = len(val_repos & test_repos)
    total_repo_overlap = repo_overlap_train_val + repo_overlap_train_test + repo_overlap_val_test

    # Non-empty checks
    train_empty_code = int((train_df[source_col].str.len() == 0).sum())
    val_empty_code = int((val_df[source_col].str.len() == 0).sum())
    test_empty_code = int((test_df[source_col].str.len() == 0).sum())

    train_empty_doc = int((train_df[target_col].str.len() == 0).sum())
    val_empty_doc = int((val_df[target_col].str.len() == 0).sum())
    test_empty_doc = int((test_df[target_col].str.len() == 0).sum())

    validation_passed = (
        total_code_overlap == 0
        and total_repo_overlap == 0
        and (train_empty_code + val_empty_code + test_empty_code) == 0
        and (train_empty_doc + val_empty_doc + test_empty_doc) == 0
        and len(train_df) > 0
        and len(val_df) > 0
        and len(test_df) > 0
    )

    results: Dict[str, Any] = {
        "all_validations_passed": validation_passed,
        "disjoint_source_code": {
            "passed": total_code_overlap == 0,
            "train_val_overlap_count": code_overlap_train_val,
            "train_test_overlap_count": code_overlap_train_test,
            "val_test_overlap_count": code_overlap_val_test,
        },
        "disjoint_repositories": {
            "passed": total_repo_overlap == 0,
            "train_val_overlap_count": repo_overlap_train_val,
            "train_test_overlap_count": repo_overlap_train_test,
            "val_test_overlap_count": repo_overlap_val_test,
        },
        "non_empty_records": {
            "passed": (train_empty_code + val_empty_code + test_empty_code + train_empty_doc + val_empty_doc + test_empty_doc) == 0,
            "empty_code_counts": {
                "train": train_empty_code,
                "validation": val_empty_code,
                "test": test_empty_code,
            },
            "empty_doc_counts": {
                "train": train_empty_doc,
                "validation": val_empty_doc,
                "test": test_empty_doc,
            },
        },
    }

    if not validation_passed:
        logger.error("Output validation FAILED: %s", results)
        raise RuntimeError(f"Split output integrity validation failed: {results}")

    logger.info("Output integrity validation PASSED: 100% disjoint code and repositories.")
    return results


# ---------------------------------------------------------------------------
# Stage 7: Reporting & Manifest Generation
# ---------------------------------------------------------------------------
def generate_preprocessing_report(manifest: Dict[str, Any]) -> str:
    """Format preprocessing results into a structured research project report."""
    cfg = manifest["configuration"]
    inp = manifest["input_dataset"]
    exc = manifest["exclusions"]
    ded = manifest["deduplication"]
    spl = manifest["splits"]
    val = manifest["validation_results"]

    lines: List[str] = []
    w = 78

    def banner(title: str) -> None:
        lines.append("=" * w)
        lines.append(f"  {title.upper()}")
        lines.append("=" * w)

    def section(title: str) -> None:
        lines.append("")
        lines.append("-" * w)
        lines.append(f"  {title}")
        lines.append("-" * w)

    banner("CODEscribe — Dataset Preprocessing & Splitting Audit Report")
    lines.append("Project: CODEscribe (Automated Markdown API Documentation Generator)")
    lines.append(f"Timestamp (UTC): {manifest['pipeline_metadata']['timestamp_utc']}")
    lines.append(f"Environment    : Python {manifest['pipeline_metadata']['python_version']} | Pandas {manifest['pipeline_metadata']['pandas_version']}")

    section("1. Raw Input Dataset Audit")
    lines.append(f"File Path        : {inp['file_relative_path']}")
    lines.append(f"SHA-256 Checksum : {inp['sha256_checksum']}")
    lines.append(f"Initial Rows     : {inp['total_raw_rows']:,}")
    lines.append(f"Source Column    : {cfg['source_column']}")
    lines.append(f"Target Column    : {cfg['target_column']}")
    lines.append(f"Grouping Column  : {cfg['group_column']}")

    section("2. Example Validation & Exclusion Accounting")
    lines.append(f"Total Excluded Records: {exc['total_excluded']:,}")
    for reason, cnt in exc["exclusion_counts"].items():
        lines.append(f"  - {reason:<45}: {cnt:,}")
    lines.append(f"Retained Records After Validation : {exc['total_retained']:,}")

    section("3. Deduplication & Cross-Split Contamination Policy")
    lines.append(f"Exact Duplicate Pairs Detected     : {ded['exact_duplicate_pairs_detected']:,}")
    lines.append(f"Duplicate Code Strings Detected    : {ded['duplicate_code_strings_detected']:,}")
    lines.append(f"Identical Code / Conflicting Docs  : {ded['identical_code_with_conflicting_doc_count']:,}")
    lines.append(f"Records Pruned by Deduplication    : {ded['records_pruned_by_deduplication']:,}")
    lines.append(f"Final Retained Dataset Size        : {ded['records_retained_after_dedup']:,}")
    lines.append("")
    lines.append(f"Policy: {ded['deduplication_policy']}")

    section("4. Leakage-Resistant Repository-Grouped Partitioning")
    lines.append(f"Splitting Strategy : {spl['strategy']}")
    lines.append(f"Random Seed        : {spl['random_seed']} (Fully Deterministic)")
    lines.append(f"Unique Repositories: {spl['total_unique_groups']:,}")
    lines.append("")
    lines.append(f"{'Partition':<15} {'Rows':<10} {'Percentage':<12} {'Repositories':<15}")
    lines.append("-" * 55)
    act = spl["actuals"]
    lines.append(f"{'Training':<15} {act['train_rows']:<10} {act['train_pct']:>6.2f}%      {act['train_repos_count']:<15}")
    lines.append(f"{'Validation':<15} {act['val_rows']:<10} {act['val_pct']:>6.2f}%      {act['val_repos_count']:<15}")
    lines.append(f"{'Testing':<15} {act['test_rows']:<10} {act['test_pct']:>6.2f}%      {act['test_repos_count']:<15}")
    lines.append("-" * 55)
    total_split_rows = act["train_rows"] + act["val_rows"] + act["test_rows"]
    lines.append(f"{'Total':<15} {total_split_rows:<10} 100.00%      {spl['total_unique_groups']:<15}")

    section("5. Partition Integrity & Leakage Verification")
    lines.append(f"All Validations Passed              : {val['all_validations_passed']}")
    lines.append(f"Mutually Disjoint Source Code       : {val['disjoint_source_code']['passed']} (0 overlapping functions)")
    lines.append(f"Mutually Disjoint Repositories      : {val['disjoint_repositories']['passed']} (0 overlapping repositories)")
    lines.append(f"Non-Empty Code & Documentation      : {val['non_empty_records']['passed']} (0 empty records)")

    section("6. Unresolved Research & Data Governance Notes")
    lines.append("1. TARGET DOCSTRING LEAKAGE IN RAW DATASET:")
    lines.append("   As identified during Phase 1 audit, in 100% of rows in this raw CSV export, the target")
    lines.append("   documentation string is contained verbatim inside 'func_code_string' as an inline docstring.")
    lines.append("   The splitting performed here guarantees repository independence, but before feeding")
    lines.append("   code into CodeT5 during training, docstrings must be removed during tokenization/feature")
    lines.append("   extraction to ensure the model learns semantic summarization rather than string copying.")
    lines.append("")
    lines.append("2. DATASET LICENSE & PROVENANCE NOTICE:")
    lines.append("   The dataset derives from 'espejelomar/code_search_net_python_10000_examples' (GitHub/Microsoft).")
    lines.append("   While individual repositories were mined under open-source licenses, specific terms")
    lines.append("   (MIT, Apache 2.0, BSD, GPL, etc.) require ongoing verification prior to publishing weights.")

    banner("END OF PREPROCESSING AUDIT REPORT")
    return "\n".join(lines)


def save_processed_data_and_reports(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: Dict[str, Any],
    processed_dir: Path,
    outputs_dir: Path,
    config: PreprocessingConfig,
) -> Dict[str, Path]:
    """
    Persist generated partitions, manifest, and reports to disk.
    Verifies that all created CSV files can be reopened successfully.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / config.train_csv_name
    val_path = processed_dir / config.val_csv_name
    test_path = processed_dir / config.test_csv_name
    manifest_path = processed_dir / config.manifest_name

    report_txt_path = outputs_dir / config.report_txt_name
    report_json_path = outputs_dir / config.report_json_name

    logger.info("Writing output partitions to %s...", processed_dir.resolve())
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    # Reopen verification
    logger.info("Verifying reopenability of generated CSV files...")
    reopened_train = pd.read_csv(train_path)
    reopened_val = pd.read_csv(val_path)
    reopened_test = pd.read_csv(test_path)

    if (
        len(reopened_train) != len(train_df)
        or len(reopened_val) != len(val_df)
        or len(reopened_test) != len(test_df)
    ):
        raise IOError("Reopened CSV file row counts do not match generated DataFrames!")

    # Attach file checksums to manifest
    manifest["generated_artifacts"] = {
        "train_csv": {
            "path": str(train_path.relative_to(config.workspace_root)).replace("\\", "/"),
            "rows": len(reopened_train),
            "sha256": compute_file_sha256(train_path),
        },
        "validation_csv": {
            "path": str(val_path.relative_to(config.workspace_root)).replace("\\", "/"),
            "rows": len(reopened_val),
            "sha256": compute_file_sha256(val_path),
        },
        "test_csv": {
            "path": str(test_path.relative_to(config.workspace_root)).replace("\\", "/"),
            "rows": len(reopened_test),
            "sha256": compute_file_sha256(test_path),
        },
    }

    # Save manifest JSON
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Saved dataset manifest to: %s", manifest_path.resolve())

    # Save outputs reports
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Saved preprocessing JSON report to: %s", report_json_path.resolve())

    text_report = generate_preprocessing_report(manifest)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    logger.info("Saved preprocessing text report to: %s", report_txt_path.resolve())

    return {
        "train_csv": train_path,
        "val_csv": val_path,
        "test_csv": test_path,
        "manifest_json": manifest_path,
        "report_json": report_json_path,
        "report_txt": report_txt_path,
    }


# ---------------------------------------------------------------------------
# Master Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(config: PreprocessingConfig) -> Dict[str, Any]:
    """
    Execute complete end-to-end dataset preprocessing pipeline:
    1. Schema validation
    2. Conservative normalization
    3. Example validation & exclusion accounting
    4. Deduplication
    5. Repository-grouped splitting
    6. Output verification
    7. Artifact and report persistence
    """
    raw_path = config.raw_csv_path
    if not raw_path.is_absolute():
        raw_path = config.workspace_root / raw_path

    logger.info("Starting CODEscribe preprocessing pipeline for: %s", raw_path.resolve())

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV does not exist: {raw_path.resolve()}")

    raw_sha256 = compute_file_sha256(raw_path)
    raw_df = pd.read_csv(raw_path, low_memory=False)
    initial_rows = len(raw_df)
    logger.info("Loaded raw CSV: %d rows, %d columns.", initial_rows, len(raw_df.columns))

    # 1. Schema Validation
    validate_schema(
        raw_df,
        required_columns=config.required_columns,
        source_col=config.source_column,
        target_col=config.target_column,
        group_col=config.group_column,
    )

    # 2. Conservative Normalization
    norm_df = normalize_dataframe(
        raw_df,
        source_col=config.source_column,
        target_col=config.target_column,
        group_col=config.group_column,
        normalize_crlf=config.normalize_line_endings,
        strip_outer=config.strip_outer_whitespace,
    )

    # 3. Example Validation
    valid_df, exclusion_counts, sample_exclusions = filter_and_validate_records(
        norm_df,
        source_col=config.source_column,
        target_col=config.target_column,
        group_col=config.group_column,
        min_code_chars=config.min_code_chars,
        min_doc_chars=config.min_doc_chars,
    )

    # 4. Deduplication
    dedup_df, dedup_meta = deduplicate_dataset(
        valid_df,
        source_col=config.source_column,
        target_col=config.target_column,
        deduplicate_identical_code=config.deduplicate_identical_code,
        deduplicate_identical_pairs=config.deduplicate_identical_pairs,
    )

    # 5. Group Splitting
    train_df, val_df, test_df, split_meta = split_by_group(
        dedup_df,
        group_col=config.group_column,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.random_seed,
        min_groups_required=config.min_groups_for_split,
    )

    # 6. Output Validation
    val_results = validate_split_outputs(
        train_df,
        val_df,
        test_df,
        source_col=config.source_column,
        target_col=config.target_column,
        group_col=config.group_column,
    )

    # Assemble Manifest Data
    manifest: Dict[str, Any] = {
        "pipeline_metadata": {
            "project_name": "CODEscribe",
            "phase": "Phase 2 - Preprocessing & Splitting",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "operating_system": f"{platform.system()} {platform.release()}",
        },
        "input_dataset": {
            "file_name": raw_path.name,
            "file_relative_path": str(raw_path.relative_to(config.workspace_root)).replace("\\", "/") if raw_path.is_relative_to(config.workspace_root) else str(raw_path),
            "sha256_checksum": raw_sha256,
            "total_raw_rows": initial_rows,
            "total_raw_columns": len(raw_df.columns),
        },
        "configuration": {
            "source_column": config.source_column,
            "target_column": config.target_column,
            "group_column": config.group_column,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "random_seed": config.random_seed,
            "min_code_chars": config.min_code_chars,
            "min_doc_chars": config.min_doc_chars,
        },
        "exclusions": {
            "total_excluded": sum(exclusion_counts.values()),
            "total_retained": len(valid_df),
            "exclusion_counts": exclusion_counts,
            "sample_exclusions": sample_exclusions,
        },
        "deduplication": dedup_meta,
        "splits": split_meta,
        "validation_results": val_results,
    }

    # 7. Persistence & Verification
    proc_dir = config.processed_dir if config.processed_dir.is_absolute() else config.workspace_root / config.processed_dir
    out_dir = config.outputs_dir if config.outputs_dir.is_absolute() else config.workspace_root / config.outputs_dir

    saved_files = save_processed_data_and_reports(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        manifest=manifest,
        processed_dir=proc_dir,
        outputs_dir=out_dir,
        config=config,
    )

    manifest["saved_files"] = {k: str(v) for k, v in saved_files.items()}
    return manifest


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="CODEscribe Dataset Preprocessing, Deduplication, and Leakage-Resistant Splitting Tool"
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("data/raw/codesearchnet_10000_python_examples.csv"),
        help="Path to raw CSV dataset (default: data/raw/codesearchnet_10000_python_examples.csv)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed CSV splits (default: data/processed)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for reports (default: outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic repository grouping (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Training partition ratio (default: 0.80)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation partition ratio (default: 0.10)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test partition ratio (default: 0.10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()
    configure_logging(verbose=args.verbose)

    config = PreprocessingConfig(
        workspace_root=Path.cwd(),
        raw_csv_path=args.raw_csv,
        processed_dir=args.processed_dir,
        outputs_dir=args.outputs_dir,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    try:
        manifest = run_pipeline(config)

        print("\n" + "=" * 70)
        print("  CODESCRIBE DATASET PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        act = manifest["splits"]["actuals"]
        print(f"  Input Dataset      : {manifest['input_dataset']['file_name']}")
        print(f"  SHA-256 Checksum   : {manifest['input_dataset']['sha256_checksum'][:16]}...")
        print(f"  Total Retained     : {manifest['deduplication']['records_retained_after_dedup']:,} rows")
        print(f"  Training Split     : {act['train_rows']:,} rows ({act['train_pct']}%) | {act['train_repos_count']:,} repos")
        print(f"  Validation Split   : {act['val_rows']:,} rows ({act['val_pct']}%) | {act['val_repos_count']:,} repos")
        print(f"  Testing Split      : {act['test_rows']:,} rows ({act['test_pct']}%) | {act['test_repos_count']:,} repos")
        print(f"  Code Overlap       : 0 records (Mutually Disjoint)")
        print(f"  Repository Overlap : 0 repositories (Mutually Disjoint)")
        print(f"  Manifest JSON      : {config.processed_dir / config.manifest_name}")
        print(f"  Report TXT         : {config.outputs_dir / config.report_txt_name}")
        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        logger.error("Preprocessing pipeline failed: %s", exc, exc_info=args.verbose)
        print(f"\n[ERROR] Preprocessing failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
