"""
training/config.py

Central configuration dataclass and constants for the CODEscribe preprocessing pipeline.
Maintains reproducible parameters, paths, validation thresholds, and deduplication policies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable configuration for dataset preprocessing, cleaning, and splitting."""

    # Project filesystem layout
    workspace_root: Path = field(default_factory=lambda: Path("."))
    raw_csv_path: Path = field(
        default_factory=lambda: Path("data/raw/codesearchnet_10000_python_examples.csv")
    )
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))

    # File names
    train_csv_name: str = "train.csv"
    val_csv_name: str = "validation.csv"
    test_csv_name: str = "test.csv"
    manifest_name: str = "dataset_manifest.json"
    report_txt_name: str = "preprocessing_report.txt"
    report_json_name: str = "preprocessing_report.json"

    # Schema specification
    source_column: str = "func_code_string"
    target_column: str = "func_documentation_string"
    group_column: str = "repository_name"
    required_columns: Tuple[str, ...] = (
        "func_code_string",
        "func_documentation_string",
        "repository_name",
    )
    metadata_columns: Tuple[str, ...] = (
        "func_name",
        "func_path_in_repository",
        "func_code_url",
        "language",
    )

    # Cleaning & normalization thresholds
    min_code_chars: int = 1
    min_doc_chars: int = 1
    normalize_line_endings: bool = True
    strip_outer_whitespace: bool = True

    # Deduplication policy
    deduplicate_identical_code: bool = True
    deduplicate_identical_pairs: bool = True

    # Splitting parameters (target 80% train, 10% val, 10% test)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42

    # Verification and safety
    min_groups_for_split: int = 3
    enforce_strict_disjoint_groups: bool = True
    enforce_strict_disjoint_code: bool = True
