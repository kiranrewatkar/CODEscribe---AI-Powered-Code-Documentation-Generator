"""
tests/test_data_pipeline.py

Unit test suite for the data pipeline, feature formatting, and partition isolation.
"""

import unittest
import pandas as pd

from training.data_pipeline import (
    CodeDocumentationDataset,
    prepare_leakage_safe_dataframe,
)


class TestDataPipeline(unittest.TestCase):
    """Test suite for data pipeline feature preparation."""

    def setUp(self):
        self.df = pd.DataFrame({
            "func_code_string": [
                "def add(a, b):\n    \"\"\"Add two integers.\"\"\"\n    return a + b",
                "def sub(a, b):\n    \"\"\"Subtract b from a.\"\"\"\n    return a - b",
                "def broken(:\n    return",  # Syntax error
            ],
            "func_documentation_string": [
                "Add two integers.",
                "Subtract b from a.",
                "Broken function.",
            ],
            "repository_name": ["repo/a", "repo/b", "repo/c"],
            "func_name": ["add", "sub", "broken"],
        })

    def test_prepare_leakage_safe_dataframe(self):
        clean_df, audit = prepare_leakage_safe_dataframe(
            self.df,
            source_col="func_code_string",
            target_col="func_documentation_string",
            task_prefix="document python code: ",
        )

        # 2 valid rows, 1 rejected
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(audit["total_rejected"], 1)
        self.assertIn("REJECT_SYNTAX_ERROR", audit["rejected_reasons"])

        # Check prefix
        row0 = clean_df.iloc[0]
        self.assertTrue(row0["input_text"].startswith("document python code: def add(a, b):"))
        self.assertNotIn("Add two integers.", row0["input_text"])
        self.assertEqual(row0["target_text"], "Add two integers.")

    def test_code_documentation_dataset_indexing(self):
        clean_df, _ = prepare_leakage_safe_dataframe(
            self.df.iloc[:2],
            task_prefix="document python code: ",
        )
        ds = CodeDocumentationDataset(clean_df)
        self.assertEqual(len(ds), 2)

        item = ds[0]
        self.assertIn("input_text", item)
        self.assertIn("target_text", item)
        self.assertEqual(item["target_text"], "Add two integers.")
        self.assertEqual(item["func_name"], "add")

    def test_test_set_never_mixed_with_train(self):
        # Verify that training features never include test split data
        train_df = pd.DataFrame({
            "func_code_string": ["def train_fn():\n    \"\"\"Doc\"\"\"\n    return 1"],
            "func_documentation_string": ["Doc"],
            "repository_name": ["repo/train"],
        })
        test_df = pd.DataFrame({
            "func_code_string": ["def test_fn():\n    \"\"\"Doc\"\"\"\n    return 2"],
            "func_documentation_string": ["Doc"],
            "repository_name": ["repo/test"],
        })

        clean_train, _ = prepare_leakage_safe_dataframe(train_df)
        clean_test, _ = prepare_leakage_safe_dataframe(test_df)

        train_texts = set(clean_train["input_text"])
        test_texts = set(clean_test["input_text"])

        self.assertEqual(len(train_texts & test_texts), 0, "Train and test inputs overlap!")


if __name__ == "__main__":
    unittest.main()
