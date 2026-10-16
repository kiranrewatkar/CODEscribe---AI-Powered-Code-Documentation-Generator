"""
tests/test_inspect_dataset.py

Unit tests for CODEscribe dataset inspection suite:
- Raw CSV discovery (single file, missing file, ambiguous multiple files)
- Schema and column validation
- Missing and blank value auditing
- Duplication analysis (code, documentation, and pairs)
- Code representation comparison and target leakage detection
- Python AST syntax parsing and error classification
- Split and repository distribution warnings
- Full inspection report generation
"""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from training.inspect_dataset import (
    analyze_categorical_distributions,
    analyze_columns_and_types,
    analyze_duplicates,
    analyze_missing_and_blanks,
    assess_python_syntax,
    audit_key_columns,
    audit_splits_and_repositories,
    compare_code_representations,
    compute_length_statistics,
    discover_raw_csv,
    generate_text_report,
    get_file_metadata,
    load_dataset,
    run_dataset_inspection,
    save_reports,
)


class TestCSVDiscovery(unittest.TestCase):
    """Test suite for discover_raw_csv."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_single_csv_success(self):
        """Should successfully return path when exactly one CSV is present."""
        csv_file = self.dir_path / "valid_data.csv"
        csv_file.write_text("a,b\n1,2", encoding="utf-8")
        # Add non-csv archive to ensure it is ignored
        (self.dir_path / "archive.zip").write_text("dummy", encoding="utf-8")

        result = discover_raw_csv(self.dir_path)
        self.assertEqual(result, csv_file)

    def test_discover_missing_dir_raises(self):
        """Should raise FileNotFoundError when data directory does not exist."""
        nonexistent = self.dir_path / "nonexistent_subdir"
        with self.assertRaises(FileNotFoundError):
            discover_raw_csv(nonexistent)

    def test_discover_no_csv_raises(self):
        """Should raise FileNotFoundError when directory contains no CSV files."""
        (self.dir_path / "archive.zip").write_text("dummy", encoding="utf-8")
        (self.dir_path / "readme.txt").write_text("dummy", encoding="utf-8")

        with self.assertRaises(FileNotFoundError) as ctx:
            discover_raw_csv(self.dir_path)
        self.assertIn("No CSV dataset found", str(ctx.exception))

    def test_discover_multiple_csv_raises(self):
        """Should raise ValueError when multiple CSV candidates exist to prevent silent error."""
        (self.dir_path / "dataset_part1.csv").write_text("a\n1", encoding="utf-8")
        (self.dir_path / "dataset_part2.csv").write_text("b\n2", encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            discover_raw_csv(self.dir_path)
        self.assertIn("Multiple CSV files detected", str(ctx.exception))


class TestMissingAndBlanks(unittest.TestCase):
    """Test suite for missing and whitespace-only blank value analysis."""

    def test_clean_dataframe(self):
        """Clean dataframe should have 0 nulls, 0 blanks, is_complete=True."""
        df = pd.DataFrame({
            "col_a": ["def foo(): pass", "def bar(): pass"],
            "col_b": ["Does foo", "Does bar"],
            "col_c": [1, 2],
        })
        results = analyze_missing_and_blanks(df)
        for col, stats in results.items():
            self.assertEqual(stats["null_count"], 0)
            self.assertEqual(stats["blank_count"], 0)
            self.assertEqual(stats["total_unusable_count"], 0)
            self.assertTrue(stats["is_complete"])

    def test_detected_nulls_and_blanks(self):
        """Should correctly count NaN as null and empty/whitespace as blanks."""
        df = pd.DataFrame({
            "code": ["def foo(): pass", None, "def bar(): pass", "   "],
            "doc": ["doc 1", "   \t\n", "doc 3", None],
            "num": [10, np.nan, 30, 40],
        })
        results = analyze_missing_and_blanks(df)

        # code column: 1 null (None), 1 blank ("   ") -> 2 unusable
        self.assertEqual(results["code"]["null_count"], 1)
        self.assertEqual(results["code"]["blank_count"], 1)
        self.assertEqual(results["code"]["total_unusable_count"], 2)
        self.assertFalse(results["code"]["is_complete"])

        # doc column: 1 null, 1 blank -> 2 unusable
        self.assertEqual(results["doc"]["null_count"], 1)
        self.assertEqual(results["doc"]["blank_count"], 1)

        # num column: 1 null, 0 blanks (numeric column)
        self.assertEqual(results["num"]["null_count"], 1)
        self.assertEqual(results["num"]["blank_count"], 0)


class TestDuplicateAnalysis(unittest.TestCase):
    """Test suite for code, doc, and pair duplicate detection."""

    def test_duplicate_counts(self):
        df = pd.DataFrame({
            "func_code_string": [
                "def add(a, b): return a + b",
                "def sub(a, b): return a - b",
                "def add(a, b): return a + b",  # duplicate code
            ],
            "func_documentation_string": [
                "Add two numbers.",
                "Sub two numbers.",
                "Add two numbers.",  # duplicate pair with row 0
            ],
        })
        dups = analyze_duplicates(df)
        self.assertEqual(dups["total_rows"], 3)
        self.assertEqual(dups["unique_code_strings"], 2)
        self.assertEqual(dups["duplicate_code_strings"], 1)
        self.assertEqual(dups["unique_doc_strings"], 2)
        self.assertEqual(dups["duplicate_doc_strings"], 1)
        self.assertEqual(dups["unique_code_doc_pairs"], 2)
        self.assertEqual(dups["duplicate_code_doc_pairs"], 1)


class TestCodeComparisonAndLeakage(unittest.TestCase):
    """Test suite for comparing whole_func vs func_code and identifying target leakage."""

    def test_identical_columns_with_embedded_docstring(self):
        """Should detect exact match and flag CRITICAL target leakage."""
        code_with_doc = 'def compute():\n    """Compute something."""\n    return 42'
        df = pd.DataFrame({
            "whole_func_string": [code_with_doc],
            "func_code_string": [code_with_doc],
            "func_documentation_string": ["Compute something."],
        })
        res = compare_code_representations(df)
        self.assertTrue(res["comparison_possible"])
        self.assertTrue(res["identical_columns"])
        self.assertEqual(res["exact_matches"], 1)
        self.assertEqual(res["docstring_verbatim_in_code_count"], 1)
        self.assertEqual(res["target_leakage_risk"], "CRITICAL")
        self.assertIn("Phase 2", res["explanation"])

    def test_stripped_docstring_no_leakage(self):
        """Should detect no leakage when docstring is not embedded in code string."""
        df = pd.DataFrame({
            "whole_func_string": ['def compute():\n    """Compute something."""\n    return 42'],
            "func_code_string": ["def compute():\n    return 42"],
            "func_documentation_string": ["Compute something."],
        })
        res = compare_code_representations(df)
        self.assertFalse(res["identical_columns"])
        self.assertEqual(res["docstring_verbatim_in_code_count"], 0)
        self.assertEqual(res["target_leakage_risk"], "LOW")


class TestPythonASTAssessment(unittest.TestCase):
    """Test suite for AST parsing conformity assessment."""

    def test_valid_and_invalid_syntax(self):
        df = pd.DataFrame({
            "func_code_string": [
                "def valid_fn(x: int) -> int:\n    return x * 2",
                "def broken_syntax(x\n    return",  # SyntaxError
                'print "Python 2 print"',  # SyntaxError under Python 3
            ]
        })
        ast_res = assess_python_syntax(df)
        self.assertTrue(ast_res["assessment_possible"])
        self.assertEqual(ast_res["total_evaluated"], 3)
        self.assertEqual(ast_res["valid_ast_count"], 1)
        self.assertEqual(ast_res["invalid_ast_count"], 2)
        self.assertAlmostEqual(ast_res["valid_ast_pct"], 33.33, places=1)
        self.assertIn("SyntaxError", [s["error_type"] for s in ast_res["error_samples"]])


class TestAuditSplitsAndRepositories(unittest.TestCase):
    """Test suite for split and repository distribution checks."""

    def test_single_split_and_skewed_repos_triggers_warnings(self):
        df = pd.DataFrame({
            "repository_name": ["saltstack/salt"] * 10 + ["other/repo"] * 2,
            "split_name": ["train"] * 12,
        })
        splits_audit = audit_splits_and_repositories(df)
        self.assertEqual(splits_audit["repository_statistics"]["unique_repositories"], 2)
        self.assertEqual(splits_audit["splits_distribution"]["train"], 12)

        warnings_text = " ".join(splits_audit["warnings"])
        self.assertIn("SINGLE_SPLIT_WARNING", warnings_text)
        self.assertIn("REPOSITORY_LEAKAGE_WARNING", warnings_text)


class TestLengthStatistics(unittest.TestCase):
    """Test suite for length statistics computation."""

    def test_length_computations(self):
        df = pd.DataFrame({
            "func_code_string": ["a" * 10, "b" * 20, "c" * 30],
            "func_documentation_string": ["x" * 5, "y" * 10, "z" * 15],
        })
        stats = compute_length_statistics(df)
        code_chars = stats["code"]["char_length"]
        self.assertEqual(code_chars["min"], 10)
        self.assertEqual(code_chars["max"], 30)
        self.assertEqual(code_chars["median"], 20.0)
        self.assertEqual(code_chars["mean"], 20.0)


class TestKeyColumnAudit(unittest.TestCase):
    """Test suite for checking availability of core columns."""

    def test_missing_column_flagged(self):
        df = pd.DataFrame({
            "func_code_string": ["def foo(): pass"],
            # func_documentation_string is missing
        })
        res = audit_key_columns(df)
        self.assertTrue(res["func_code_string"]["present"])
        self.assertFalse(res["func_documentation_string"]["present"])


class TestEndToEndReportGeneration(unittest.TestCase):
    """Test full pipeline report generation with mock dataset."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_and_save_reports(self):
        csv_file = self.dir_path / "mock_dataset.csv"
        df = pd.DataFrame({
            "Unnamed: 0": [0, 1],
            "repository_name": ["repo/a", "repo/b"],
            "func_path_in_repository": ["a.py", "b.py"],
            "func_name": ["fn1", "fn2"],
            "whole_func_string": ['def fn1():\n    """Doc 1"""\n    return 1', 'def fn2():\n    """Doc 2"""\n    return 2'],
            "language": ["python", "python"],
            "func_code_string": ['def fn1():\n    """Doc 1"""\n    return 1', 'def fn2():\n    """Doc 2"""\n    return 2'],
            "func_code_tokens": ["tokens1", "tokens2"],
            "func_documentation_string": ["Doc 1", "Doc 2"],
            "func_documentation_tokens": ["dtoks1", "dtoks2"],
            "split_name": ["train", "train"],
            "func_code_url": ["http://url1", "http://url2"],
        })
        df.to_csv(csv_file, index=False)

        audit = run_dataset_inspection(csv_file, root_dir=self.dir_path)
        output_dir = self.dir_path / "outputs"
        txt_path, json_path = save_reports(audit, output_dir=output_dir)

        self.assertTrue(txt_path.exists())
        self.assertTrue(json_path.exists())

        # Check json can be loaded
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["file_metadata"]["total_rows"], 2)
        self.assertEqual(data["file_metadata"]["total_columns"], 12)

        # Check text report content
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("CODEscribe", text)
        self.assertIn("Target Leakage Risk Assessment", text)


if __name__ == "__main__":
    unittest.main()
