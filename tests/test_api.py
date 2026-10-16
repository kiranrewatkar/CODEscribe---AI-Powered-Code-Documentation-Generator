"""
tests/test_api.py

Unit and integration test suite for the CODEscribe Phase UI-3 local inference API:
- Health check endpoint (/api/health)
- Curated examples endpoint (/api/examples)
- Documentation generation endpoint (/api/document) with mocked model
- Input validation (empty code, whitespace, character limits)
- Python syntax error rejection (HTTP 422)
- Generation parameter bounds validation
- Model unreadiness handling (HTTP 503)
- Markdown assembly and Mermaid inclusion toggling
- Pure-unit tests for generator preprocessing and AST extraction

Note: Unit tests use a mocked generator and do not require GPU inference
or model weight downloads.
"""

import ast
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.generator import (
    ASTMetadata,
    CodeScribeGenerator,
    CodeSyntaxError,
    GenerationResult,
    ModelNotLoadedError,
    TimingMetadata,
    assemble_markdown_document,
    extract_function_ast_info,
)
from app.server import app, get_generator, set_generator


class TestLocalInferenceAPI(unittest.TestCase):
    """Test suite for app/server.py endpoints and app/generator.py logic."""

    def setUp(self):
        """Set up test client and isolate generator."""
        self.original_generator = get_generator()
        self.client = TestClient(app)

    def tearDown(self):
        """Restore original generator state."""
        set_generator(self.original_generator)

    def test_health_endpoint(self):
        """Verify /api/health reports system status and metadata."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = True
        mock_gen.model_path = "model/codescribe"
        mock_gen.device = "cpu"
        set_generator(mock_gen)

        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertIn("torch_version", data)
        self.assertIn("transformers_version", data)
        self.assertIn("device", data)

    def test_examples_endpoint(self):
        """Verify /api/examples returns valid curated Python code snippets."""
        response = self.client.get("/api/examples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        examples = data.get("examples", [])
        self.assertGreaterEqual(len(examples), 3)

        for ex in examples:
            self.assertIn("id", ex)
            self.assertIn("title", ex)
            self.assertIn("description", ex)
            self.assertIn("code", ex)
            # Verify every example parses cleanly as valid Python AST
            tree = ast.parse(ex["code"])
            self.assertIsNotNone(tree)

    def test_document_generation_with_mocked_model(self):
        """Verify /api/document endpoint returns complete DocumentResponse schema."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = True
        mock_gen.device = "cpu"

        # Mock generator output
        mock_gen.generate_documentation.return_value = GenerationResult(
            success=True,
            markdown="# `add`\n\n```python\ndef add(a: int, b: int) -> int\n```\n\nAdd two numbers.",
            docstring="Add two numbers.",
            mermaid_code="graph TD\n    N0([\"Start: add(a, b)\"]) --> N1([\"End\"])",
            ast_metadata=ASTMetadata(
                function_names=["add"],
                parameters=["a: int", "b: int"],
                return_annotation="int",
                docstring_stripped=False,
                syntax_valid=True,
            ),
            timing_ms=TimingMetadata(
                preprocessing_ms=1.5,
                inference_ms=12.0,
                mermaid_ms=0.8,
                total_ms=14.3,
            ),
            device="cpu",
            status="SUCCESS",
        )
        set_generator(mock_gen)

        payload = {
            "code": "def add(a: int, b: int) -> int:\n    return a + b\n",
            "num_beams": 1,
            "max_new_tokens": 128,
            "include_mermaid": True,
        }

        response = self.client.post("/api/document", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertIn("add", data["markdown"])
        self.assertEqual(data["docstring"], "Add two numbers.")
        self.assertIn("graph TD", data["mermaid_code"])
        self.assertEqual(data["ast_metadata"]["function_names"], ["add"])
        self.assertEqual(data["ast_metadata"]["parameters"], ["a: int", "b: int"])
        self.assertEqual(data["ast_metadata"]["return_annotation"], "int")
        self.assertGreater(data["timing_ms"]["total_ms"], 0)

    def test_document_generation_without_mermaid(self):
        """Verify /api/document respects include_mermaid=False."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = True
        mock_gen.device = "cpu"

        mock_gen.generate_documentation.return_value = GenerationResult(
            success=True,
            markdown="# `test_fn`\n\nTest function doc.",
            docstring="Test function doc.",
            mermaid_code=None,
            ast_metadata=ASTMetadata(function_names=["test_fn"]),
            timing_ms=TimingMetadata(total_ms=5.0),
            device="cpu",
            status="SUCCESS",
        )
        set_generator(mock_gen)

        payload = {
            "code": "def test_fn(): pass",
            "include_mermaid": False,
        }

        response = self.client.post("/api/document", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["mermaid_code"])

    def test_input_validation_empty_and_whitespace(self):
        """Verify empty and whitespace inputs are rejected with HTTP 422."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = True
        set_generator(mock_gen)

        # Empty string
        res1 = self.client.post("/api/document", json={"code": ""})
        self.assertEqual(res1.status_code, 422)

        # Whitespace-only string
        res2 = self.client.post("/api/document", json={"code": "   \n\t   "})
        self.assertEqual(res2.status_code, 422)

    def test_syntax_error_rejection(self):
        """Verify Python syntax errors raise HTTP 422 with diagnostic details."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = True
        mock_gen.generate_documentation.side_effect = CodeSyntaxError("Syntax error at line 1: invalid syntax")
        set_generator(mock_gen)

        payload = {"code": "def broken(:\n    return"}
        response = self.client.post("/api/document", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("syntax error", response.json()["detail"].lower())

    def test_parameter_bounds_validation(self):
        """Verify generation parameter bounds (num_beams, max_new_tokens, temperature)."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = True
        set_generator(mock_gen)

        valid_code = "def valid(): pass"

        # num_beams out of bounds
        r1 = self.client.post("/api/document", json={"code": valid_code, "num_beams": 0})
        self.assertEqual(r1.status_code, 422)

        r2 = self.client.post("/api/document", json={"code": valid_code, "num_beams": 10})
        self.assertEqual(r2.status_code, 422)

        # max_new_tokens out of bounds
        r3 = self.client.post("/api/document", json={"code": valid_code, "max_new_tokens": 5})
        self.assertEqual(r3.status_code, 422)

        r4 = self.client.post("/api/document", json={"code": valid_code, "max_new_tokens": 1000})
        self.assertEqual(r4.status_code, 422)

        # temperature out of bounds
        r5 = self.client.post("/api/document", json={"code": valid_code, "temperature": 0.0})
        self.assertEqual(r5.status_code, 422)

    def test_model_not_loaded_returns_503(self):
        """Verify HTTP 503 Service Unavailable when model is not loaded."""
        mock_gen = MagicMock(spec=CodeScribeGenerator)
        mock_gen.is_loaded = False
        set_generator(mock_gen)

        response = self.client.post("/api/document", json={"code": "def fn(): pass"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("not loaded", response.json()["detail"].lower())

    def test_ast_metadata_extraction(self):
        """Verify AST extraction retrieves function names, typed parameters, and return types."""
        code = (
            "def calculate_total(subtotal: float, tax_rate: float = 0.05) -> float:\n"
            "    return subtotal * (1 + tax_rate)\n"
        )
        tree = ast.parse(code)
        info = extract_function_ast_info(tree)

        self.assertEqual(info.function_names, ["calculate_total"])
        self.assertIn("subtotal: float", info.parameters)
        self.assertIn("tax_rate: float", info.parameters)
        self.assertEqual(info.return_annotation, "float")
        self.assertTrue(info.syntax_valid)

    def test_markdown_document_assembly(self):
        """Verify assemble_markdown_document generates expected headings and code blocks."""
        doc = assemble_markdown_document(
            function_name="process_data",
            docstring="Process incoming records and filter null values.",
            parameters=["records: list", "filter_null: bool"],
            return_annotation="list",
            source_code="def process_data(records, filter_null=True): return []",
            mermaid_code="graph TD\n    A --> B",
        )

        self.assertIn("# `process_data`", doc)
        self.assertIn("def process_data(records: list, filter_null: bool) -> list", doc)
        self.assertIn("## Description", doc)
        self.assertIn("Process incoming records and filter null values.", doc)
        self.assertIn("## Parameters", doc)
        self.assertIn("**`records`** (`list`)", doc)
        self.assertIn("## Returns", doc)
        self.assertIn("## Control Flow Diagram", doc)
        self.assertIn("```mermaid", doc)
        self.assertIn("graph TD", doc)

    def test_generator_pipeline_with_mock_t5(self):
        """Verify full CodeScribeGenerator pipeline using a mocked T5 model and tokenizer."""
        gen = CodeScribeGenerator(model_path="dummy_path")

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.config.decoder_start_token_id = 0
        mock_model.generate.return_value = [[10, 20, 30]]

        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 2
        mock_tok.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value="tensor_ids")),
            "attention_mask": MagicMock(to=MagicMock(return_value="tensor_mask")),
        }
        mock_tok.decode.return_value = "Compute Fibonacci sequence up to N."

        gen.model = mock_model
        gen.tokenizer = mock_tok
        gen.is_loaded = True
        gen.device = "cpu"

        code = (
            "def fib(n: int) -> int:\n"
            "    \"\"\"Old docstring to strip.\"\"\"\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)\n"
        )

        result = gen.generate_documentation(code)

        self.assertTrue(result.success)
        self.assertEqual(result.docstring, "Compute Fibonacci sequence up to N.")
        self.assertTrue(result.ast_metadata.docstring_stripped)
        self.assertEqual(result.ast_metadata.function_names, ["fib"])
        self.assertIsNotNone(result.mermaid_code)
        self.assertIn("graph TD", result.mermaid_code)
        self.assertGreater(result.timing_ms.total_ms, 0)


if __name__ == "__main__":
    unittest.main()
