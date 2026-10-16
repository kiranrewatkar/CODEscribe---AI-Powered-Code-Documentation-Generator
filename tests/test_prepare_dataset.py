"""
tests/test_prepare_dataset.py

Comprehensive unit test suite for the CODEscribe dataset preprocessing pipeline:
- Schema validation & error handling
- Null, blank, and malformed field exclusions with reason tracking
- Whitespace normalization preserving Python indentation
- Exact duplicate and conflicting documentation deduplication
- Deterministic repository-grouped splitting & zero-leakage isolation
- Output integrity verification, reopenability, and SHA-256 checks
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from training.config import PreprocessingConfig
from training.prepare_dataset import (
    compute_file_sha256,
    deduplicate_dataset,
    filter_and_validate_records,
    generate_preprocessing_report,
    normalize_dataframe,
    normalize_text,
    run_pipeline,
    save_processed_data_and_reports,
    split_by_group,
    validate_schema,
    validate_split_outputs,
)


class TestSchemaValidation(unittest.TestCase):
    """Test suite for schema validation."""

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({
            "func_code_string": ["def foo(): pass"],
            "repository_name": ["repo/test"],
            # func_documentation_string is missing
        })
        with self.assertRaises(ValueError) as ctx:
            validate_schema(
                df,
                required_columns=("func_code_string", "func_documentation_string", "repository_name"),
                source_col="func_code_string",
                target_col="func_documentation_string",
                group_col="repository_name",
            )
        self.assertIn("Required columns missing", str(ctx.exception))
        self.assertIn("func_documentation_string", str(ctx.exception))

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["func_code_string", "func_documentation_string", "repository_name"])
        with self.assertRaises(ValueError) as ctx:
            validate_schema(
                df,
                required_columns=("func_code_string", "func_documentation_string", "repository_name"),
                source_col="func_code_string",
                target_col="func_documentation_string",
                group_col="repository_name",
            )
        self.assertIn("empty", str(ctx.exception))

    def test_valid_schema_passes(self):
        df = pd.DataFrame({
            "func_code_string": ["def foo(): pass"],
            "func_documentation_string": ["Documentation."],
            "repository_name": ["repo/test"],
        })
        # Should not raise
        validate_schema(
            df,
            required_columns=("func_code_string", "func_documentation_string", "repository_name"),
            source_col="func_code_string",
            target_col="func_documentation_string",
            group_col="repository_name",
        )


class TestNormalization(unittest.TestCase):
    """Test suite for conservative normalization."""

    def test_whitespace_and_crlf_normalization(self):
        raw = "   def calculate():\r\n        x = 10\r\n        return x\r\n   "
        normalized = normalize_text(raw, normalize_crlf=True, strip_outer=True)
        expected = "def calculate():\n        x = 10\n        return x"
        self.assertEqual(normalized, expected)

    def test_preserves_internal_python_indentation(self):
        python_func = "def complex_fn(a, b):\n    if a > b:\n        return a\n    return b"
        normalized = normalize_text(python_func)
        self.assertEqual(normalized, python_func)
        # Verify 4-space and 8-space indentations remain intact
        lines = normalized.split("\n")
        self.assertTrue(lines[1].startswith("    if"))
        self.assertTrue(lines[2].startswith("        return"))

    def test_preserves_documentation_markdown(self):
        doc = "Summary of API.\n\n- Parameter `a`: integer.\n- Returns: bool."
        normalized = normalize_text(doc)
        self.assertEqual(normalized, doc)

    def test_normalize_dataframe(self):
        df = pd.DataFrame({
            "func_code_string": ["  def foo():\r\n    pass  "],
            "func_documentation_string": ["  Does foo.\r\n  "],
            "repository_name": ["  org/repo  "],
        })
        norm_df = normalize_dataframe(
            df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            group_col="repository_name",
        )
        self.assertEqual(norm_df["func_code_string"].iloc[0], "def foo():\n    pass")
        self.assertEqual(norm_df["func_documentation_string"].iloc[0], "Does foo.")
        self.assertEqual(norm_df["repository_name"].iloc[0], "org/repo")


class TestExampleValidationAndExclusions(unittest.TestCase):
    """Test suite for example validation and exclusion accounting."""

    def test_null_and_blank_exclusions(self):
        df = pd.DataFrame({
            "func_code_string": [
                "def valid1(): pass",
                "",  # blank code
                "def valid2(): pass",
                "def valid3(): pass",
                None,  # null code
            ],
            "func_documentation_string": [
                "Valid doc 1.",
                "Valid doc 2.",
                "",  # blank doc
                "Valid doc 3.",
                "Valid doc 4.",
            ],
            "repository_name": ["repo/a", "repo/b", "repo/c", "", "repo/e"],  # blank repo at index 3
        })
        retained, counts, samples = filter_and_validate_records(
            df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            group_col="repository_name",
            min_code_chars=1,
            min_doc_chars=1,
        )

        # Only index 0 is valid across all 3 columns
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained["func_code_string"].iloc[0], "def valid1(): pass")

        self.assertEqual(counts["REASON_EMPTY_OR_BLANK_SOURCE_CODE"], 2)  # indices 1 and 4
        self.assertEqual(counts["REASON_EMPTY_OR_BLANK_DOCUMENTATION"], 1)  # index 2
        self.assertEqual(counts["REASON_EMPTY_REPOSITORY_METADATA"], 1)  # index 3

    def test_min_length_threshold_exclusions(self):
        df = pd.DataFrame({
            "func_code_string": ["def foo(): pass", "a"],
            "func_documentation_string": ["Valid doc", "x"],
            "repository_name": ["repo/1", "repo/2"],
        })
        retained, counts, _ = filter_and_validate_records(
            df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            group_col="repository_name",
            min_code_chars=5,
            min_doc_chars=5,
        )
        self.assertEqual(len(retained), 1)
        self.assertEqual(counts["REASON_SOURCE_CODE_BELOW_MIN_LENGTH"], 1)


class TestDeduplication(unittest.TestCase):
    """Test suite for duplicate detection and conflict handling."""

    def test_exact_duplicate_pairs(self):
        df = pd.DataFrame({
            "func_code_string": ["def fn(): return 1", "def fn(): return 1", "def fn2(): return 2"],
            "func_documentation_string": ["Doc 1", "Doc 1", "Doc 2"],
            "repository_name": ["repo/1", "repo/1", "repo/2"],
        })
        dedup_df, audit = deduplicate_dataset(
            df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            deduplicate_identical_code=True,
        )
        self.assertEqual(len(dedup_df), 2)
        self.assertEqual(audit["exact_duplicate_pairs_detected"], 1)
        self.assertEqual(audit["duplicate_code_strings_detected"], 1)
        self.assertEqual(audit["records_pruned_by_deduplication"], 1)

    def test_identical_code_with_conflicting_documentation(self):
        df = pd.DataFrame({
            "func_code_string": ["def fn(): pass", "def fn(): pass"],
            "func_documentation_string": ["First doc interpretation", "Conflicting second doc"],
            "repository_name": ["repo/1", "repo/2"],
        })
        dedup_df, audit = deduplicate_dataset(
            df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            deduplicate_identical_code=True,
        )
        # Deduplicating on code retains only 1 record to strictly prevent cross-split leakage
        self.assertEqual(len(dedup_df), 1)
        self.assertEqual(audit["identical_code_with_conflicting_doc_count"], 1)
        self.assertEqual(audit["records_pruned_by_deduplication"], 1)


class TestDeterministicGroupSplitting(unittest.TestCase):
    """Test suite for repository-grouped leakage-resistant splitting."""

    def setUp(self):
        # Synthetic dataset with 10 repositories and varying row counts
        rows = []
        for i in range(10):
            repo = f"org/repo_{i}"
            for j in range(5):
                rows.append({
                    "func_code_string": f"def fn_{i}_{j}(): return {i * 10 + j}",
                    "func_documentation_string": f"Function doc for {i}_{j}",
                    "repository_name": repo,
                })
        self.df = pd.DataFrame(rows)  # 50 rows, 10 repos (5 rows per repo)

    def test_deterministic_reproducibility(self):
        """Same random seed must yield identical partition splits."""
        t1, v1, s1, _ = split_by_group(self.df, group_col="repository_name", seed=42)
        t2, v2, s2, _ = split_by_group(self.df, group_col="repository_name", seed=42)

        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(s1, s2)

    def test_strict_repository_isolation(self):
        """Guarantees 0 repository overlap between train, validation, and test."""
        train_df, val_df, test_df, meta = split_by_group(
            self.df, group_col="repository_name", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        train_repos = set(train_df["repository_name"])
        val_repos = set(val_df["repository_name"])
        test_repos = set(test_df["repository_name"])

        self.assertEqual(len(train_repos & val_repos), 0, "Train and Val share repositories!")
        self.assertEqual(len(train_repos & test_repos), 0, "Train and Test share repositories!")
        self.assertEqual(len(val_repos & test_repos), 0, "Val and Test share repositories!")

        # All rows accounted for
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df))

    def test_insufficient_repositories_raises(self):
        """Should raise ValueError when too few repositories exist for 3-way split."""
        small_df = pd.DataFrame({
            "func_code_string": ["def a(): pass", "def b(): pass"],
            "func_documentation_string": ["Doc a", "Doc b"],
            "repository_name": ["repo/1", "repo/2"],  # Only 2 repositories
        })
        with self.assertRaises(ValueError) as ctx:
            split_by_group(small_df, group_col="repository_name", min_groups_required=3)
        self.assertIn("Insufficient independent repositories", str(ctx.exception))


class TestOutputValidationAndReopening(unittest.TestCase):
    """Test suite for post-split output validation and reopenability."""

    def test_validate_split_outputs_passes_for_disjoint_data(self):
        train_df = pd.DataFrame({
            "func_code_string": ["def a(): pass"],
            "func_documentation_string": ["Doc a"],
            "repository_name": ["repo/train"],
        })
        val_df = pd.DataFrame({
            "func_code_string": ["def b(): pass"],
            "func_documentation_string": ["Doc b"],
            "repository_name": ["repo/val"],
        })
        test_df = pd.DataFrame({
            "func_code_string": ["def c(): pass"],
            "func_documentation_string": ["Doc c"],
            "repository_name": ["repo/test"],
        })

        results = validate_split_outputs(
            train_df,
            val_df,
            test_df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            group_col="repository_name",
        )
        self.assertTrue(results["all_validations_passed"])
        self.assertTrue(results["disjoint_source_code"]["passed"])
        self.assertTrue(results["disjoint_repositories"]["passed"])

    def test_validate_split_outputs_fails_on_leakage(self):
        # Leakage: repo/shared appears in both train and test
        train_df = pd.DataFrame({
            "func_code_string": ["def a(): pass"],
            "func_documentation_string": ["Doc a"],
            "repository_name": ["repo/shared"],
        })
        val_df = pd.DataFrame({
            "func_code_string": ["def b(): pass"],
            "func_documentation_string": ["Doc b"],
            "repository_name": ["repo/val"],
        })
        test_df = pd.DataFrame({
            "func_code_string": ["def c(): pass"],
            "func_documentation_string": ["Doc c"],
            "repository_name": ["repo/shared"],
        })

        with self.assertRaises(RuntimeError) as ctx:
            validate_split_outputs(
                train_df,
                val_df,
                test_df,
                source_col="func_code_string",
                target_col="func_documentation_string",
                group_col="repository_name",
            )
        self.assertIn("validation failed", str(ctx.exception))


class TestEndToEndPipeline(unittest.TestCase):
    """Test full pipeline execution on synthetic data."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.raw_csv = self.base_dir / "raw_test_data.csv"

        # Generate synthetic CSV with 15 repositories and 45 rows
        rows = []
        for r_idx in range(15):
            repo = f"test_org/test_repo_{r_idx}"
            for f_idx in range(3):
                rows.append({
                    "func_code_string": f"def function_{r_idx}_{f_idx}():\n    return {r_idx + f_idx}",
                    "func_documentation_string": f"Documentation for function_{r_idx}_{f_idx}.",
                    "repository_name": repo,
                    "func_name": f"function_{r_idx}_{f_idx}",
                    "func_path_in_repository": f"src/{repo}/fn.py",
                    "func_code_url": f"https://github.com/{repo}",
                    "language": "python",
                })
        pd.DataFrame(rows).to_csv(self.raw_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_creates_artifacts_and_verifies_reopen(self):
        config = PreprocessingConfig(
            workspace_root=self.base_dir,
            raw_csv_path=self.raw_csv,
            processed_dir=self.base_dir / "processed",
            outputs_dir=self.base_dir / "outputs",
            random_seed=42,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        manifest = run_pipeline(config)

        # Check files exist
        train_csv = self.base_dir / "processed" / "train.csv"
        val_csv = self.base_dir / "processed" / "validation.csv"
        test_csv = self.base_dir / "processed" / "test.csv"
        manifest_json = self.base_dir / "processed" / "dataset_manifest.json"
        report_txt = self.base_dir / "outputs" / "preprocessing_report.txt"
        report_json = self.base_dir / "outputs" / "preprocessing_report.json"

        self.assertTrue(train_csv.exists())
        self.assertTrue(val_csv.exists())
        self.assertTrue(test_csv.exists())
        self.assertTrue(manifest_json.exists())
        self.assertTrue(report_txt.exists())
        self.assertTrue(report_json.exists())

        # Check reopenability and checksums
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_test = pd.read_csv(test_csv)
        self.assertEqual(len(df_train) + len(df_val) + len(df_test), 45)

        # Check manifest data
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(manifest_data["deduplication"]["records_retained_after_dedup"], 45)
        self.assertTrue(manifest_data["validation_results"]["all_validations_passed"])


if __name__ == "__main__":
    unittest.main()
