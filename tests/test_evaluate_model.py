"""
tests/test_evaluate_model.py

Unit test suite for the CODEscribe held-out test evaluation pipeline:
- Argument parsing (defaults, overrides, flag validation)
- Test-data isolation (strictly isolated from train/val splits)
- Model path validation (fails safely on missing files, no silent download)
- Output report schema and CSV structure
- Sample limiting mechanics
- End-to-end evaluation using a synthetic mocked model/tokenizer (CPU, offline)
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from training.evaluate_model import (
    evaluate_model,
    parse_arguments,
    verify_local_model_path,
)


class TestEvaluateModel(unittest.TestCase):
    """Test suite for evaluate_model.py functionality."""

    def test_argument_parsing_defaults(self):
        """Verify default CLI arguments match project specifications."""
        args = parse_arguments([])
        self.assertEqual(args.model_path, "outputs/pilot_run_accelerate16/model")
        self.assertEqual(args.test_data_path, "data/processed/test.csv")
        self.assertEqual(args.output_dir, "outputs/evaluation_pilot_accelerate16")
        self.assertIsNone(args.sample_limit)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.max_source_length, 512)
        self.assertEqual(args.max_target_length, 128)
        self.assertEqual(args.num_beams, 1)
        self.assertEqual(args.seed, 42)
        self.assertIsNone(args.device)

    def test_argument_parsing_overrides(self):
        """Verify custom CLI flags override defaults correctly."""
        custom_argv = [
            "--model-path", "custom/checkpoint",
            "--test-data-path", "custom/test.csv",
            "--output-dir", "custom/outputs",
            "--sample-limit", "25",
            "--batch-size", "8",
            "--num-beams", "2",
            "--device", "cpu",
            "--seed", "123",
        ]
        args = parse_arguments(custom_argv)
        self.assertEqual(args.model_path, "custom/checkpoint")
        self.assertEqual(args.test_data_path, "custom/test.csv")
        self.assertEqual(args.output_dir, "custom/outputs")
        self.assertEqual(args.sample_limit, 25)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.num_beams, 2)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.seed, 123)

    def test_verify_local_model_path_missing_dir(self):
        """Verify that non-existent model directories raise FileNotFoundError."""
        non_existent = Path("non_existent_model_checkpoint_directory_12345")
        with self.assertRaises(FileNotFoundError) as ctx:
            verify_local_model_path(non_existent)
        self.assertIn("does not exist", str(ctx.exception))

    def test_verify_local_model_path_missing_weights(self):
        """Verify that directory missing weights raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create config.json but omit weights
            (tmp_path / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as ctx:
                verify_local_model_path(tmp_path)
            self.assertIn("No valid weights file found", str(ctx.exception))

    def test_verify_local_model_path_success(self):
        """Verify that directory with config.json and weights passes validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "config.json").write_text("{}", encoding="utf-8")
            (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")
            # Should not raise
            verify_local_model_path(tmp_path)

    def test_test_data_isolation(self):
        """Verify that evaluation strictly isolates test data and never reads train or val splits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_csv = tmp_path / "test.csv"
            # Create a 2-row test dataset
            df = pd.DataFrame({
                "func_code_string": [
                    "def greet(name):\n    \"\"\"Say hello.\"\"\"\n    return f'Hello {name}'",
                    "def add(a, b):\n    \"\"\"Add numbers.\"\"\"\n    return a + b",
                ],
                "func_documentation_string": [
                    "Say hello.",
                    "Add numbers.",
                ],
                "repository_name": ["repo/one", "repo/two"],
                "func_name": ["greet", "add"],
            })
            df.to_csv(test_csv, index=False)

            out_dir = tmp_path / "eval_out"

            args = parse_arguments([
                "--test-data-path", str(test_csv),
                "--output-dir", str(out_dir),
                "--device", "cpu",
            ])

            # Track all read_csv calls
            original_read_csv = pd.read_csv
            accessed_paths = []

            def tracked_read_csv(filepath, *pos_args, **kwargs):
                accessed_paths.append(str(filepath))
                return original_read_csv(filepath, *pos_args, **kwargs)

            with patch("pandas.read_csv", side_effect=tracked_read_csv):
                with patch("training.evaluate_model.load_model_and_tokenizer") as mock_load:
                    with patch("training.evaluate_model.generate_predictions_batch") as mock_gen:
                        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
                        mock_gen.return_value = ["Generated doc 1", "Generated doc 2"]

                        report = evaluate_model(args)

            # Assert no train.csv or validation.csv was ever opened
            for path in accessed_paths:
                self.assertNotIn("train.csv", path.lower())
                self.assertNotIn("validation.csv", path.lower())
            self.assertEqual(len(accessed_paths), 1)
            self.assertIn("test.csv", accessed_paths[0].lower())

    def test_output_schema_and_sample_limiting_with_mock(self):
        """
        Verify output JSON report schema, CSV columns, sample limit behavior,
        and syntax error rejection recording on synthetic data.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_csv = tmp_path / "synthetic_test.csv"
            out_dir = tmp_path / "eval_output"

            # 3 rows: 2 valid functions, 1 syntax error
            synthetic_df = pd.DataFrame({
                "func_code_string": [
                    "def compute_square(x):\n    \"\"\"Return square of x.\"\"\"\n    return x * x",
                    "def compute_cube(x):\n    \"\"\"Return cube of x.\"\"\"\n    return x ** 3",
                    "def invalid_syntax(x:\n    return x",
                ],
                "func_documentation_string": [
                    "Return square of x.",
                    "Return cube of x.",
                    "Invalid function.",
                ],
                "repository_name": ["repo/math", "repo/math", "repo/broken"],
                "func_name": ["compute_square", "compute_cube", "invalid_syntax"],
            })
            synthetic_df.to_csv(test_csv, index=False)

            # Test with sample_limit = 1 (should evaluate only the first eligible row)
            args = parse_arguments([
                "--test-data-path", str(test_csv),
                "--output-dir", str(out_dir),
                "--sample-limit", "1",
                "--device", "cpu",
            ])

            with patch("training.evaluate_model.load_model_and_tokenizer") as mock_load:
                with patch("training.evaluate_model.generate_predictions_batch") as mock_gen:
                    mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
                    mock_gen.return_value = ["Return square of x."]

                    report = evaluate_model(args)

            # 1. Verify Dataset Counts
            self.assertEqual(report["dataset_counts"]["total_test_rows"], 3)
            self.assertEqual(report["dataset_counts"]["eligible_test_rows"], 2)
            self.assertEqual(report["dataset_counts"]["evaluated_rows"], 1)
            self.assertEqual(report["dataset_counts"]["rejected_rows"], 1)
            self.assertEqual(report["dataset_counts"]["skipped_sample_limit_rows"], 1)
            self.assertIn("REJECT_SYNTAX_ERROR", report["dataset_counts"]["rejection_reasons"])

            # 2. Verify Output Files Exist
            json_file = out_dir / "test_evaluation.json"
            csv_file = out_dir / "predictions.csv"
            self.assertTrue(json_file.exists())
            self.assertTrue(csv_file.exists())

            # 3. Verify JSON Schema
            with open(json_file, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            required_top_keys = {
                "metadata",
                "dataset_counts",
                "generation_parameters",
                "metrics",
                "runtime",
                "architecture_verification",
            }
            self.assertEqual(set(loaded_json.keys()), required_top_keys)
            self.assertIn("rouge1", loaded_json["metrics"])
            self.assertIn("rouge2", loaded_json["metrics"])
            self.assertIn("rougeL", loaded_json["metrics"])
            self.assertIn("bleu", loaded_json["metrics"])
            self.assertEqual(loaded_json["metrics"]["metric_scale"], "percentage (0 to 100)")

            # 4. Verify CSV Schema and Status Values
            pred_df = pd.read_csv(csv_file)
            expected_csv_cols = [
                "test_row_index",
                "repository_name",
                "func_name",
                "reference_documentation",
                "generated_documentation",
                "status",
            ]
            self.assertEqual(list(pred_df.columns), expected_csv_cols)
            self.assertEqual(len(pred_df), 3)

            # Check status of each row
            statuses = list(pred_df["status"])
            self.assertEqual(statuses[0], "EVALUATED")
            self.assertEqual(statuses[1], "SKIPPED_SAMPLE_LIMIT")
            self.assertEqual(statuses[2], "REJECTED_SYNTAX_ERROR")


if __name__ == "__main__":
    unittest.main()
