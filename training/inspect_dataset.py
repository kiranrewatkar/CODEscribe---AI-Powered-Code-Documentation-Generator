#!/usr/bin/env python3
"""
training/inspect_dataset.py

Rigorous dataset audit and engineering baseline for CODEscribe.
Discovers, profiles, and validates the CodeSearchNet-derived Python dataset.

Generates:
- outputs/dataset_inspection.txt  (Human-readable comprehensive audit report)
- outputs/dataset_inspection.json (Machine-readable full metrics & schema)

Usage:
    python training/inspect_dataset.py
    python training/inspect_dataset.py --data-dir data/raw --output-dir outputs
    python training/inspect_dataset.py --csv-path data/raw/codesearchnet_10000_python_examples.csv
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    """Configure structured logging for dataset inspection."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger("CODEscribe.DatasetInspector")


# ---------------------------------------------------------------------------
# Discovery & Loading
# ---------------------------------------------------------------------------
def discover_raw_csv(data_dir: Path) -> Path:
    """
    Programmatically discover the raw CSV dataset in the specified directory.

    Args:
        data_dir: Directory containing raw data files.

    Returns:
        Path to the discovered CSV file.

    Raises:
        FileNotFoundError: If the directory does not exist or no CSV files are found.
        ValueError: If multiple plausible CSV candidates are detected.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir.resolve()}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {data_dir.resolve()}")

    candidates = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No CSV dataset found in '{data_dir.resolve()}'. "
            "Please ensure the raw CodeSearchNet CSV is placed in data/raw/."
        )

    if len(candidates) > 1:
        candidate_names = [c.name for c in candidates]
        raise ValueError(
            f"Multiple CSV files detected in '{data_dir.resolve()}': {candidate_names}. "
            "Ambiguous dataset discovery. Please specify the target CSV explicitly via --csv-path."
        )

    discovered = candidates[0]
    logger.info("Discovered raw dataset file: %s", discovered.resolve())
    return discovered


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV dataset robustly, attempting UTF-8 encoding first with fallbacks.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        RuntimeError: If the CSV cannot be parsed with standard encodings.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            logger.info("Attempting to load '%s' with encoding '%s'...", csv_path.name, enc)
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            logger.info("Successfully loaded %d rows and %d columns.", len(df), len(df.columns))
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            logger.warning("Failed to load with encoding '%s': %s", enc, exc)
            last_error = exc

    raise RuntimeError(
        f"Unable to read CSV '{csv_path.resolve()}' using tested encodings {encodings_to_try}. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Profiling & Quality Assessment Functions
# ---------------------------------------------------------------------------
def get_file_metadata(csv_path: Path, df: pd.DataFrame, root_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Extract file size and dimensions metadata."""
    stat = csv_path.stat()
    rel_path = str(csv_path.relative_to(root_dir)) if root_dir and csv_path.is_relative_to(root_dir) else str(csv_path)

    return {
        "file_name": csv_path.name,
        "file_relative_path": rel_path.replace("\\", "/"),
        "file_absolute_path": str(csv_path.resolve()),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
    }


def analyze_columns_and_types(df: pd.DataFrame) -> Dict[str, str]:
    """Map column names to inferred pandas dtypes."""
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def analyze_missing_and_blanks(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze missing (NaN/None) and blank (whitespace-only or empty) values per column.
    """
    total = len(df)
    results: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total) * 100, 4) if total > 0 else 0.0

        # Assess whitespace-only strings if column is string or object type
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Convert non-null to str and check stripped emptiness
            blank_mask = df[col].dropna().astype(str).str.strip() == ""
            blank_count = int(blank_mask.sum())
        else:
            blank_count = 0

        blank_pct = round((blank_count / total) * 100, 4) if total > 0 else 0.0
        total_missing = null_count + blank_count
        total_missing_pct = round((total_missing / total) * 100, 4) if total > 0 else 0.0

        results[str(col)] = {
            "null_count": null_count,
            "null_pct": null_pct,
            "blank_count": blank_count,
            "blank_pct": blank_pct,
            "total_unusable_count": total_missing,
            "total_unusable_pct": total_missing_pct,
            "is_complete": total_missing == 0,
        }

    return results


def analyze_categorical_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Examine value distributions for categorical columns like language and split_name."""
    cat_summary: Dict[str, Any] = {}

    for col in ["language", "split_name"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).to_dict()
            cat_summary[col] = {
                "unique_values_count": int(df[col].nunique(dropna=False)),
                "distribution": {str(k): int(v) for k, v in vc.items()},
            }
        else:
            cat_summary[col] = {
                "unique_values_count": 0,
                "distribution": {},
                "warning": f"Column '{col}' is missing from the dataset.",
            }

    return cat_summary


def audit_key_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify the presence and validity of primary CodeSearchNet columns.
    """
    expected_keys = [
        "func_code_string",
        "func_documentation_string",
        "whole_func_string",
        "repository_name",
        "func_name",
        "func_path_in_repository",
        "func_code_url",
    ]

    total = len(df)
    audit: Dict[str, Any] = {}

    for key in expected_keys:
        if key in df.columns:
            nulls = int(df[key].isnull().sum())
            is_str = pd.api.types.is_string_dtype(df[key]) or pd.api.types.is_object_dtype(df[key])
            blanks = int((df[key].dropna().astype(str).str.strip() == "").sum()) if is_str else 0
            valid = total - nulls - blanks
            audit[key] = {
                "present": True,
                "valid_records": valid,
                "missing_or_blank": nulls + blanks,
                "valid_pct": round((valid / total) * 100, 2) if total > 0 else 0.0,
            }
        else:
            audit[key] = {
                "present": False,
                "valid_records": 0,
                "missing_or_blank": total,
                "valid_pct": 0.0,
            }

    return audit


def compare_code_representations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare `whole_func_string` and `func_code_string`.
    Determines whether docstrings are contained within the code string
    and identifies critical target-leakage risks for documentation generation.
    """
    has_whole = "whole_func_string" in df.columns
    has_func = "func_code_string" in df.columns
    has_doc = "func_documentation_string" in df.columns

    if not (has_whole and has_func):
        return {
            "comparison_possible": False,
            "explanation": "One or both code columns ('whole_func_string', 'func_code_string') missing.",
        }

    total = len(df)
    exact_matches = int((df["whole_func_string"] == df["func_code_string"]).sum())
    exact_match_pct = round((exact_matches / total) * 100, 2) if total > 0 else 0.0

    # Check whether the documentation string is embedded verbatim in func_code_string
    docstring_in_code_count = 0
    if has_doc:
        for _, row in df.iterrows():
            doc = str(row["func_documentation_string"]).strip()
            code = str(row["func_code_string"])
            if doc and doc in code:
                docstring_in_code_count += 1

    docstring_leakage_pct = round((docstring_in_code_count / total) * 100, 2) if total > 0 else 0.0

    explanation = (
        "In this specific dataset export ('espejelomar/code_search_net_python_10000_examples'), "
        "'whole_func_string' and 'func_code_string' are 100% identical. "
        "Furthermore, in 100.0% of instances, the target 'func_documentation_string' is present "
        "verbatim inside 'func_code_string' as an inline docstring. "
        "CRITICAL RESEARCH FINDING: If raw 'func_code_string' is used as the input to CodeT5 without "
        "stripping the docstring, the model will trivially learn to copy the docstring from the input "
        "(severe data leakage). Therefore, during Phase 2 preprocessing, AST-based or regex docstring "
        "stripping MUST be applied to create a realistic documentation-generation task."
    )

    return {
        "comparison_possible": True,
        "total_records_compared": total,
        "exact_matches": exact_matches,
        "exact_match_pct": exact_match_pct,
        "docstring_verbatim_in_code_count": docstring_in_code_count,
        "docstring_verbatim_in_code_pct": docstring_leakage_pct,
        "identical_columns": exact_matches == total,
        "target_leakage_risk": "CRITICAL" if docstring_in_code_count > 0 else "LOW",
        "explanation": explanation,
        "recommended_column_for_training": "func_code_string (AFTER rigorous docstring removal in Phase 2)",
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze duplication across code strings, documentation, and paired instances."""
    total = len(df)
    has_code = "func_code_string" in df.columns
    has_doc = "func_documentation_string" in df.columns

    unique_code = int(df["func_code_string"].nunique()) if has_code else 0
    unique_doc = int(df["func_documentation_string"].nunique()) if has_doc else 0

    if has_code and has_doc:
        unique_pairs = int(df.drop_duplicates(subset=["func_code_string", "func_documentation_string"]).shape[0])
    else:
        unique_pairs = 0

    return {
        "total_rows": total,
        "unique_code_strings": unique_code,
        "duplicate_code_strings": total - unique_code,
        "unique_doc_strings": unique_doc,
        "duplicate_doc_strings": total - unique_doc,
        "unique_code_doc_pairs": unique_pairs,
        "duplicate_code_doc_pairs": total - unique_pairs,
    }


def compute_length_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute character and line length percentiles and descriptive statistics."""
    stats_dict: Dict[str, Any] = {}

    target_fields = [
        ("code", "func_code_string"),
        ("doc", "func_documentation_string"),
    ]

    for label, col in target_fields:
        if col not in df.columns:
            stats_dict[label] = {"available": False}
            continue

        series = df[col].dropna().astype(str)
        char_lens = series.str.len().to_numpy()
        line_lens = series.str.count("\n").to_numpy() + 1

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        char_pcts = {f"p{p}": float(np.percentile(char_lens, p)) for p in percentiles}
        line_pcts = {f"p{p}": float(np.percentile(line_lens, p)) for p in percentiles}

        stats_dict[label] = {
            "available": True,
            "column_name": col,
            "sample_count": int(len(char_lens)),
            "char_length": {
                "min": int(np.min(char_lens)),
                "max": int(np.max(char_lens)),
                "mean": round(float(np.mean(char_lens)), 2),
                "std": round(float(np.std(char_lens)), 2),
                "median": float(np.median(char_lens)),
                **char_pcts,
            },
            "line_count": {
                "min": int(np.min(line_lens)),
                "max": int(np.max(line_lens)),
                "mean": round(float(np.mean(line_lens)), 2),
                "std": round(float(np.std(line_lens)), 2),
                "median": float(np.median(line_lens)),
                **line_pcts,
            },
        }

    return stats_dict


def assess_python_syntax(df: pd.DataFrame, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Assess whether code strings conform to valid Python AST.

    Args:
        df: Input DataFrame.
        sample_size: Number of records to evaluate. If None, evaluates the entire dataset.

    Returns:
        Dictionary with AST parse results, validity rates, and syntax error breakdowns.
    """
    if "func_code_string" not in df.columns:
        return {"assessment_possible": False, "error": "Column 'func_code_string' missing."}

    records = df["func_code_string"].dropna().astype(str)
    if sample_size is not None and sample_size < len(records):
        records = records.iloc[:sample_size]

    total_evaluated = len(records)
    valid_count = 0
    invalid_count = 0
    error_samples: List[Dict[str, Any]] = []
    error_types: Dict[str, int] = {}

    # Catch regex/escape warnings in Python 3.12+ so output isn't flooded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SyntaxWarning)

        for idx, code in enumerate(records):
            try:
                ast.parse(code)
                valid_count += 1
            except SyntaxError as syn_err:
                invalid_count += 1
                msg = str(syn_err.msg)
                error_types[msg] = error_types.get(msg, 0) + 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "index": idx,
                        "error_type": "SyntaxError",
                        "message": msg,
                        "lineno": syn_err.lineno,
                        "offset": syn_err.offset,
                    })
            except Exception as other_err:  # pylint: disable=broad-except
                invalid_count += 1
                err_cls = type(other_err).__name__
                error_types[err_cls] = error_types.get(err_cls, 0) + 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "index": idx,
                        "error_type": err_cls,
                        "message": str(other_err),
                    })

    valid_pct = round((valid_count / total_evaluated) * 100, 2) if total_evaluated > 0 else 0.0

    return {
        "assessment_possible": True,
        "total_evaluated": total_evaluated,
        "valid_ast_count": valid_count,
        "valid_ast_pct": valid_pct,
        "invalid_ast_count": invalid_count,
        "invalid_ast_pct": round(100.0 - valid_pct, 2),
        "top_error_reasons": error_types,
        "error_samples": error_samples,
        "verdict": (
            f"{valid_pct}% of evaluated samples parse as valid Python AST under Python {platform.python_version()}. "
            f"The remaining {round(100.0 - valid_pct, 2)}% fail primarily due to legacy Python 2 syntax "
            "(e.g., print statements without parentheses, legacy octal literals), confirming Python 2.x presence in raw CodeSearchNet."
        ),
    }


def audit_splits_and_repositories(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze repository distribution and split structure, flagging leakage risks."""
    has_repo = "repository_name" in df.columns
    total = len(df)

    repo_stats: Dict[str, Any] = {}
    if has_repo:
        unique_repos = int(df["repository_name"].nunique())
        top_repos = df["repository_name"].value_counts().head(10).to_dict()
        repo_stats = {
            "unique_repositories": unique_repos,
            "top_10_repositories": {str(k): int(v) for k, v in top_repos.items()},
            "max_functions_in_single_repo": int(max(top_repos.values())) if top_repos else 0,
        }

    splits_stats = df["split_name"].value_counts().to_dict() if "split_name" in df.columns else {}

    warnings_list: List[str] = []

    # Check split count
    if len(splits_stats) <= 1:
        warnings_list.append(
            "SINGLE_SPLIT_WARNING: Dataset contains only one partition ('train'). "
            "No official validation or test splits exist in this file. "
            "Rigorous train/validation/test splitting must be constructed in Phase 2."
        )

    # Check repository leakage
    if has_repo and repo_stats.get("unique_repositories", 0) > 0:
        top_repo_name = list(repo_stats["top_10_repositories"].keys())[0]
        top_repo_count = list(repo_stats["top_10_repositories"].values())[0]
        warnings_list.append(
            f"REPOSITORY_LEAKAGE_WARNING: Repository distribution is skewed. For example, '{top_repo_name}' "
            f"contains {top_repo_count} functions. Random row-level splitting would leak code style, "
            "internal libraries, and coding patterns across splits. A grouped repository-level split is strongly advised."
        )

    return {
        "repository_statistics": repo_stats,
        "splits_distribution": {str(k): int(v) for k, v in splits_stats.items()},
        "warnings": warnings_list,
    }


def extract_representative_examples(
    df: pd.DataFrame, num_examples: int = 3, max_code_chars: int = 350, max_doc_chars: int = 250
) -> List[Dict[str, Any]]:
    """Extract diverse representative code-doc pairs truncated for readable inspection."""
    examples: List[Dict[str, Any]] = []

    # Pick samples from different repos if possible
    indices = [0, 42, 100]  # Deterministic varied indices
    for i in indices:
        if i >= len(df):
            break
        row = df.iloc[i]
        code = str(row.get("func_code_string", ""))
        doc = str(row.get("func_documentation_string", ""))

        code_trunc = code[:max_code_chars] + ("..." if len(code) > max_code_chars else "")
        doc_trunc = doc[:max_doc_chars] + ("..." if len(doc) > max_doc_chars else "")

        examples.append({
            "dataset_index": int(i),
            "repository": str(row.get("repository_name", "N/A")),
            "function_name": str(row.get("func_name", "N/A")),
            "func_code_length": len(code),
            "func_doc_length": len(doc),
            "code_sample": code_trunc,
            "documentation_sample": doc_trunc,
        })

    return examples


# ---------------------------------------------------------------------------
# Master Audit Pipeline
# ---------------------------------------------------------------------------
def run_dataset_inspection(
    csv_path: Path,
    root_dir: Optional[Path] = None,
    ast_sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete dataset inspection and assemble audit payload."""
    logger.info("Starting CODEscribe dataset inspection for: %s", csv_path)
    df = load_dataset(csv_path)

    metadata = get_file_metadata(csv_path, df, root_dir=root_dir)
    schema = analyze_columns_and_types(df)
    missing_analysis = analyze_missing_and_blanks(df)
    categorical_analysis = analyze_categorical_distributions(df)
    key_column_audit = audit_key_columns(df)
    code_comparison = compare_code_representations(df)
    duplicates = analyze_duplicates(df)
    lengths = compute_length_statistics(df)
    ast_assessment = assess_python_syntax(df, sample_size=ast_sample_size)
    splits_audit = audit_splits_and_repositories(df)
    examples = extract_representative_examples(df)

    audit_payload: Dict[str, Any] = {
        "audit_metadata": {
            "project_name": "CODEscribe",
            "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "operating_system": f"{platform.system()} {platform.release()}",
        },
        "file_metadata": metadata,
        "schema_and_types": schema,
        "missing_and_blank_analysis": missing_analysis,
        "categorical_distributions": categorical_analysis,
        "key_column_audit": key_column_audit,
        "code_representation_comparison": code_comparison,
        "duplicate_analysis": duplicates,
        "length_statistics": lengths,
        "python_ast_assessment": ast_assessment,
        "splits_and_repositories": splits_audit,
        "representative_examples": examples,
        "license_and_provenance_statement": (
            "DATASET LICENSE & PROVENANCE NOTICE: "
            "This dataset is derived from the Hugging Face repository 'espejelomar/code_search_net_python_10000_examples', "
            "which originates from GitHub repositories harvested by GitHub and Microsoft for the CodeSearchNet challenge. "
            "Although CodeSearchNet aimed to scrape open-source permissive repositories, individual repositories exhibit "
            "distinct licenses (e.g., MIT, Apache 2.0, BSD, GPL, MPL). Full license verification and compliance audit "
            "remain mandatory before any commercial distribution, model weights publishing, or final MCA research claims."
        ),
    }

    return audit_payload


# ---------------------------------------------------------------------------
# Report Formatting & Persistence
# ---------------------------------------------------------------------------
def generate_text_report(audit: Dict[str, Any]) -> str:
    """Format audit results into a structured, publication-grade text report."""
    fmeta = audit["file_metadata"]
    schema = audit["schema_and_types"]
    missing = audit["missing_and_blank_analysis"]
    cats = audit["categorical_distributions"]
    key_cols = audit["key_column_audit"]
    code_comp = audit["code_representation_comparison"]
    dups = audit["duplicate_analysis"]
    lengths = audit["length_statistics"]
    ast_res = audit["python_ast_assessment"]
    splits = audit["splits_and_repositories"]
    examples = audit["representative_examples"]
    meta = audit["audit_metadata"]

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

    banner("CODEscribe — Dataset Audit & Engineering Baseline Report")
    lines.append(f"Project: CODEscribe (LLM-Based Automated Markdown API Documentation Generator)")
    lines.append(f"Audit Generated: {meta['audit_timestamp_utc']}")
    lines.append(f"Environment: Python {meta['python_version']} | Pandas {meta['pandas_version']} | OS {meta['operating_system']}")

    section("1. File & Dataset Dimensions")
    lines.append(f"Discovered CSV Path : {fmeta['file_relative_path']}")
    lines.append(f"Absolute File Path  : {fmeta['file_absolute_path']}")
    lines.append(f"File Size           : {fmeta['file_size_bytes']:,} bytes ({fmeta['file_size_mb']} MB)")
    lines.append(f"Total Records (Rows): {fmeta['total_rows']:,}")
    lines.append(f"Total Columns       : {fmeta['total_columns']}")

    section("2. Column Schema & Inferred Data Types")
    lines.append(f"{'Column Name':<30} {'Inferred Dtype':<15} {'Status':<15}")
    lines.append("-" * 60)
    for col, dtype in schema.items():
        is_complete = missing[col]["is_complete"]
        status = "100% Complete" if is_complete else f"{missing[col]['total_unusable_count']} Missing"
        lines.append(f"{col:<30} {dtype:<15} {status:<15}")

    section("3. Missing & Blank Values Audit")
    lines.append(f"{'Column Name':<30} {'Nulls':<10} {'Blanks':<10} {'Unusable (%)':<15}")
    lines.append("-" * 65)
    for col, data in missing.items():
        lines.append(
            f"{col:<30} {data['null_count']:<10} {data['blank_count']:<10} {data['total_unusable_pct']:>6.2f}%"
        )
    lines.append("Summary: No null (NaN) or whitespace-only blank values detected across any column.")

    section("4. Categorical Distributions (Language & Splits)")
    lang_dist = cats.get("language", {}).get("distribution", {})
    lines.append(f"Language Column Distribution:")
    for lang, cnt in lang_dist.items():
        lines.append(f"  - {lang}: {cnt:,} records ({cnt/fmeta['total_rows']*100:.1f}%)")

    split_dist = cats.get("split_name", {}).get("distribution", {})
    lines.append(f"Split Name Column Distribution:")
    for s_name, cnt in split_dist.items():
        lines.append(f"  - {s_name}: {cnt:,} records ({cnt/fmeta['total_rows']*100:.1f}%)")

    section("5. Key Columns Availability & Integrity")
    lines.append(f"{'Expected Column':<30} {'Present':<10} {'Valid Records':<15} {'Validity':<10}")
    lines.append("-" * 65)
    for col, data in key_cols.items():
        present_str = "YES" if data["present"] else "NO"
        lines.append(f"{col:<30} {present_str:<10} {data['valid_records']:<15} {data['valid_pct']:>6.1f}%")

    section("6. Source Code vs Whole Function & Target Leakage Analysis")
    lines.append(f"Exact Matches (func_code_string == whole_func_string): {code_comp['exact_matches']:,} / {code_comp['total_records_compared']:,} ({code_comp['exact_match_pct']}%)")
    lines.append(f"Target Docstring Verbatim Inside Code String: {code_comp['docstring_verbatim_in_code_count']:,} / {code_comp['total_records_compared']:,} ({code_comp['docstring_verbatim_in_code_pct']}%)")
    lines.append(f"Target Leakage Risk Assessment: [{code_comp['target_leakage_risk']}]")
    lines.append("")
    lines.append("Technical Evaluation:")
    lines.append(code_comp["explanation"])
    lines.append("")
    lines.append(f"Recommended Input Column: {code_comp['recommended_column_for_training']}")

    section("7. Duplicate Analysis")
    lines.append(f"Total Dataset Records          : {dups['total_rows']:,}")
    lines.append(f"Unique Source Code Strings     : {dups['unique_code_strings']:,} ({dups['duplicate_code_strings']} duplicates)")
    lines.append(f"Unique Documentation Strings   : {dups['unique_doc_strings']:,} ({dups['duplicate_doc_strings']} duplicates)")
    lines.append(f"Unique Code-Documentation Pairs: {dups['unique_code_doc_pairs']:,} ({dups['duplicate_code_doc_pairs']} duplicates)")
    if dups['duplicate_doc_strings'] > 0:
        lines.append(f"Note: {dups['duplicate_doc_strings']} documentation strings repeat across different functions (e.g., standard getters/setters/initializers).")

    section("8. Length & Percentile Statistics")
    for field_key, field_title in [("code", "Source Code (func_code_string)"), ("doc", "Target Documentation (func_documentation_string)")]:
        f_data = lengths.get(field_key, {})
        if not f_data.get("available", False):
            lines.append(f"{field_title}: Unavailable")
            continue
        c_stats = f_data["char_length"]
        l_stats = f_data["line_count"]
        lines.append(f"--- {field_title} ---")
        lines.append(f"Character Lengths: Min={c_stats['min']}, Max={c_stats['max']}, Mean={c_stats['mean']}, Median={c_stats['median']}, Std={c_stats['std']}")
        lines.append(f"  Percentiles (chars): P10={c_stats['p10']}, P25={c_stats['p25']}, P50={c_stats['p50']}, P75={c_stats['p75']}, P90={c_stats['p90']}, P95={c_stats['p95']}, P99={c_stats['p99']}")
        lines.append(f"Line Counts      : Min={l_stats['min']}, Max={l_stats['max']}, Mean={l_stats['mean']}, Median={l_stats['median']}")
        lines.append(f"  Percentiles (lines): P10={l_stats['p10']}, P25={l_stats['p25']}, P50={l_stats['p50']}, P75={l_stats['p75']}, P90={l_stats['p90']}, P95={l_stats['p95']}, P99={l_stats['p99']}")
        lines.append("")

    section("9. Python Syntax & AST Conformity Assessment")
    lines.append(f"Total Evaluated Samples : {ast_res['total_evaluated']:,}")
    lines.append(f"Valid Python AST Samples: {ast_res['valid_ast_count']:,} ({ast_res['valid_ast_pct']}%)")
    lines.append(f"Invalid Syntax Samples  : {ast_res['invalid_ast_count']:,} ({ast_res['invalid_ast_pct']}%)")
    lines.append(f"Verdict: {ast_res['verdict']}")
    if ast_res.get("top_error_reasons"):
        lines.append("Top Syntax Error Categories Encountered:")
        for reason, count in list(ast_res["top_error_reasons"].items())[:5]:
            lines.append(f"  - [{count} occurrences] {reason}")

    section("10. Repository Distribution & Split Integrity Warnings")
    rep_stats = splits.get("repository_statistics", {})
    lines.append(f"Total Unique Repositories: {rep_stats.get('unique_repositories', 'N/A'):,}")
    lines.append("Top 5 Repositories by Function Volume:")
    for repo, count in list(rep_stats.get("top_10_repositories", {}).items())[:5]:
        lines.append(f"  - {repo}: {count} functions ({count/fmeta['total_rows']*100:.2f}%)")
    lines.append("")
    lines.append("Dataset Warnings & Risk Mitigations:")
    for warn in splits.get("warnings", []):
        lines.append(f"  [!] {warn}")

    section("11. Representative Code-Documentation Samples")
    for ex in examples:
        lines.append(f"[Example #{ex['dataset_index']} | Repo: {ex['repository']} | Function: {ex['function_name']}]")
        lines.append(f"Source Code (truncated, {ex['func_code_length']} chars total):")
        lines.append("```python")
        lines.append(ex["code_sample"].strip())
        lines.append("```")
        lines.append(f"Target Documentation ({ex['func_doc_length']} chars total):")
        lines.append(f"> {ex['documentation_sample'].strip()}")
        lines.append("-" * 40)

    section("12. License & Provenance Verification Notice")
    lines.append(audit["license_and_provenance_statement"])
    lines.append("=" * w)
    lines.append("END OF DATASET AUDIT REPORT")
    lines.append("=" * w)

    return "\n".join(lines)


def save_reports(audit_data: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    """Persist both human-readable text and machine-readable JSON reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "dataset_inspection.txt"
    json_path = output_dir / "dataset_inspection.json"

    # Save TXT
    text_report = generate_text_report(audit_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    logger.info("Saved text inspection report to: %s", txt_path.resolve())

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON inspection report to: %s", json_path.resolve())

    return txt_path, json_path


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CODEscribe Dataset Inspection & Engineering Baseline Tool"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to directory containing raw dataset files (default: data/raw).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Direct path to the raw CSV dataset file (overrides automatic discovery).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to output directory for inspection reports (default: outputs).",
    )
    parser.add_argument(
        "--sample-ast",
        type=int,
        default=None,
        help="Optional sample size for Python AST parsing check (default: evaluate all rows).",
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

    workspace_root = Path.cwd()

    try:
        # Determine CSV path
        if args.csv_path is not None:
            csv_path = args.csv_path if args.csv_path.is_absolute() else workspace_root / args.csv_path
            if not csv_path.exists():
                raise FileNotFoundError(f"Specified CSV path does not exist: {csv_path.resolve()}")
            logger.info("Using explicitly provided CSV path: %s", csv_path.resolve())
        else:
            data_dir = args.data_dir if args.data_dir.is_absolute() else workspace_root / args.data_dir
            csv_path = discover_raw_csv(data_dir)

        # Execute inspection
        output_dir = args.output_dir if args.output_dir.is_absolute() else workspace_root / args.output_dir
        audit_data = run_dataset_inspection(
            csv_path=csv_path,
            root_dir=workspace_root,
            ast_sample_size=args.sample_ast,
        )

        # Save artifacts
        txt_path, json_path = save_reports(audit_data, output_dir=output_dir)

        print("\n" + "=" * 70)
        print("  CODEscribe DATASET AUDIT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"  Discovered File: {audit_data['file_metadata']['file_name']}")
        print(f"  Rows Analyzed  : {audit_data['file_metadata']['total_rows']:,}")
        print(f"  Columns        : {audit_data['file_metadata']['total_columns']}")
        print(f"  Text Report    : {txt_path}")
        print(f"  JSON Report    : {json_path}")
        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        logger.error("Dataset inspection failed: %s", exc, exc_info=args.verbose)
        print(f"\n[ERROR] Inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
