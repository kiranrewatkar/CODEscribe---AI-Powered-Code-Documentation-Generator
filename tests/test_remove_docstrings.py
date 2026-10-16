"""
tests/test_remove_docstrings.py

Unit test suite for the AST-based docstring removal and leakage prevention engine.
"""

import ast
import unittest

from training.remove_docstrings import (
    find_docstring_spans,
    is_target_leaked,
    remove_docstrings_from_source,
)


class TestDocstringRemoval(unittest.TestCase):
    """Test suite for remove_docstrings_from_source."""

    def test_multiline_docstring_removed(self):
        source = (
            "def calculate_total(price, tax):\n"
            "    \"\"\"\n"
            "    Calculate total price including tax.\n"
            "    Multi-line description.\n"
            "    \"\"\"\n"
            "    return price * (1 + tax)\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Calculate total price including tax.")
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Calculate total price", clean_code)
        self.assertIn("return price * (1 + tax)", clean_code)
        # Verify valid AST
        ast.parse(clean_code)

    def test_single_line_docstring_removed(self):
        source = (
            "def is_valid(token):\n"
            "    \"\"\"Validate user security token.\"\"\"\n"
            "    return len(token) > 10\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Validate user security token.")
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Validate user security token", clean_code)
        self.assertIn("return len(token) > 10", clean_code)
        ast.parse(clean_code)

    def test_single_statement_function_replaced_with_pass(self):
        source = (
            "def abstract_handler(event):\n"
            "    \"\"\"Handle event in subclass.\"\"\"\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Handle event in subclass.")
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Handle event in subclass", clean_code)
        self.assertIn("pass", clean_code)
        # Crucial: verify syntax parses cleanly with pass
        ast.parse(clean_code)

    def test_ordinary_string_literals_preserved(self):
        source = (
            "def format_greeting(name):\n"
            "    \"\"\"Greeting docstring.\"\"\"\n"
            "    prefix = 'Hello, '\n"
            "    suffix = '!'\n"
            "    query = \"SELECT * FROM users WHERE status = 'active'\"\n"
            "    return f'{prefix}{name}{suffix}'\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Greeting docstring.")
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Greeting docstring", clean_code)
        self.assertIn("prefix = 'Hello, '", clean_code)
        self.assertIn("SELECT * FROM users", clean_code)
        ast.parse(clean_code)

    def test_comments_and_decorators_preserved(self):
        source = (
            "@decorator_one\n"
            "@decorator_two(param=True)\n"
            "def decorated_fn(x):\n"
            "    # This is a critical internal comment\n"
            "    \"\"\"Docstring for decorated function.\"\"\"\n"
            "    # Another comment before logic\n"
            "    return x ** 2\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Docstring for decorated function.")
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Docstring for decorated function", clean_code)
        self.assertIn("@decorator_one", clean_code)
        self.assertIn("@decorator_two(param=True)", clean_code)
        self.assertIn("# This is a critical internal comment", clean_code)
        self.assertIn("return x ** 2", clean_code)
        ast.parse(clean_code)

    def test_nested_functions_and_classes(self):
        source = (
            "class Outer:\n"
            "    \"\"\"Outer class doc.\"\"\"\n"
            "    def method(self):\n"
            "        \"\"\"Method doc.\"\"\"\n"
            "        def helper():\n"
            "            \"\"\"Helper doc.\"\"\"\n"
            "            return 1\n"
            "        return helper()\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source)
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Outer class doc", clean_code)
        self.assertNotIn("Method doc", clean_code)
        self.assertNotIn("Helper doc", clean_code)
        self.assertIn("return helper()", clean_code)
        ast.parse(clean_code)

    def test_async_functions(self):
        source = (
            "async def fetch_payload(url):\n"
            "    \"\"\"Asynchronously fetch HTTP payload.\"\"\"\n"
            "    await asyncio.sleep(0.1)\n"
            "    return {'status': 200}\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Asynchronously fetch HTTP payload.")
        self.assertEqual(status, "SUCCESS")
        self.assertNotIn("Asynchronously fetch HTTP payload", clean_code)
        self.assertIn("await asyncio.sleep(0.1)", clean_code)
        ast.parse(clean_code)

    def test_syntax_error_rejected(self):
        broken_source = "def broken(:\n    return"
        clean_code, status, meta = remove_docstrings_from_source(broken_source)
        self.assertIsNone(clean_code)
        self.assertEqual(status, "REJECT_SYNTAX_ERROR")
        self.assertIn("error_message", meta)

    def test_target_leak_detected_and_rejected(self):
        # Docstring removed, but author also put the exact documentation text into a comment or string
        leaky_source = (
            "def calculate(a, b):\n"
            "    \"\"\"Compute exponential decay rate.\"\"\"\n"
            "    # Compute exponential decay rate\n"
            "    return a * (0.5 ** b)\n"
        )
        clean_code, status, _ = remove_docstrings_from_source(
            leaky_source, target_doc="Compute exponential decay rate."
        )
        # Because the target documentation text remains in the comment, multi-tier leakage check catches it!
        self.assertEqual(status, "REJECT_TARGET_LEAK_DETECTED")
        self.assertIsNone(clean_code)

    def test_no_docstring_code_accepted(self):
        source = "def pure_code(a, b):\n    return a + b\n"
        clean_code, status, _ = remove_docstrings_from_source(source, target_doc="Add numbers")
        self.assertEqual(status, "NO_DOCSTRING_PRESENT")
        self.assertEqual(clean_code, source)


if __name__ == "__main__":
    unittest.main()
