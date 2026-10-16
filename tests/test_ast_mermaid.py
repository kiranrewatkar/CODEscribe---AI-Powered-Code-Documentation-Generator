"""
tests/test_ast_mermaid.py

Unit test suite for the deterministic Python AST-to-Mermaid generator:
- Simple functions and sequential statements
- Conditional branching (if/elif/else) and nested conditions
- For and while loops with loopback, break, and continue edges
- Return statements and empty or docstring-only functions
- Syntax error handling without crashes
- Label escaping (quotes, angle brackets, brackets)
- Deterministic output across multiple identical runs
- Class methods and exception handling
"""

import unittest

from app.ast_mermaid import (
    escape_mermaid_label,
    generate_mermaid_flowchart,
    MermaidResult,
)


class TestASTMermaidGenerator(unittest.TestCase):
    """Test suite for app/ast_mermaid.py Control Flow Graph generation."""

    def test_simple_function_and_sequential_statements(self):
        """Verify sequential assignments and function entry/exit."""
        code = (
            "def add_numbers(a, b):\n"
            "    c = a + b\n"
            "    total = c * 2\n"
            "    return total\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        self.assertIsNone(res.error_message)
        self.assertEqual(res.function_names, ["add_numbers"])
        self.assertIn("Start: add_numbers(a, b)", res.mermaid_code)
        self.assertIn("End (add_numbers)", res.mermaid_code)
        self.assertIn("c = a + b", res.mermaid_code)
        self.assertIn("total = c * 2", res.mermaid_code)
        self.assertIn("return total", res.mermaid_code)
        self.assertIn("assignment", res.supported_constructs)
        self.assertIn("return", res.supported_constructs)
        self.assertGreater(res.node_count, 3)
        self.assertGreater(res.edge_count, 3)

    def test_conditional_branches_if_else(self):
        """Verify if/else branching and edge labels."""
        code = (
            "def check_sign(x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    else:\n"
            "        return -1\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        self.assertIn("if_condition", res.supported_constructs)
        self.assertIn("else_branch", res.supported_constructs)
        self.assertIn("-->|True|", res.mermaid_code)
        self.assertIn("-->|False|", res.mermaid_code)
        self.assertIn("return 1", res.mermaid_code)
        self.assertIn("return -1", res.mermaid_code)

    def test_nested_conditionals(self):
        """Verify nested if/else statements."""
        code = (
            "def classify(val):\n"
            "    if val >= 0:\n"
            "        if val == 0:\n"
            "            res = 'zero'\n"
            "        else:\n"
            "            res = 'positive'\n"
            "    else:\n"
            "        res = 'negative'\n"
            "    return res\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        self.assertIn("if val &gt;= 0", res.mermaid_code)
        self.assertIn("if val == 0", res.mermaid_code)
        self.assertIn("res = 'zero'", res.mermaid_code)
        self.assertIn("res = 'positive'", res.mermaid_code)
        self.assertIn("res = 'negative'", res.mermaid_code)

    def test_while_loop_with_continue_and_break(self):
        """Verify while loop condition, loopback, break, and continue edges."""
        code = (
            "def countdown(n):\n"
            "    while n > 0:\n"
            "        if n == 5:\n"
            "            n -= 1\n"
            "            continue\n"
            "        if n == 2:\n"
            "            break\n"
            "        n -= 1\n"
            "    return n\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        self.assertIn("while_loop", res.supported_constructs)
        self.assertIn("break", res.supported_constructs)
        self.assertIn("continue", res.supported_constructs)
        self.assertIn("while n &gt; 0", res.mermaid_code)
        self.assertIn("-->|True|", res.mermaid_code)
        self.assertIn("-->|False|", res.mermaid_code)

    def test_for_loop_with_next_and_done(self):
        """Verify for loop target/iter, Next edge, and Done edge."""
        code = (
            "def sum_items(items):\n"
            "    total = 0\n"
            "    for x in items:\n"
            "        total += x\n"
            "    return total\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        self.assertIn("for_loop", res.supported_constructs)
        self.assertIn("for x in items", res.mermaid_code)
        self.assertIn("-->|Next|", res.mermaid_code)
        self.assertIn("-->|Done|", res.mermaid_code)

    def test_empty_function_and_docstring_only(self):
        """Verify safe handling of empty bodies and docstring-only functions."""
        code_pass = "def noop():\n    pass\n"
        res_pass = generate_mermaid_flowchart(code_pass)
        self.assertTrue(res_pass.success)
        self.assertIn("Start: noop()", res_pass.mermaid_code)
        self.assertIn("pass", res_pass.mermaid_code)
        self.assertIn("End (noop)", res_pass.mermaid_code)

        code_doc = "def doc_only():\n    '''Only a docstring.'''\n"
        res_doc = generate_mermaid_flowchart(code_doc)
        self.assertTrue(res_doc.success)
        self.assertIn("Start: doc_only()", res_doc.mermaid_code)
        self.assertIn("End (doc_only)", res_doc.mermaid_code)

    def test_syntax_errors_handled_gracefully(self):
        """Verify that syntax errors produce clean diagnostics and never crash."""
        bad_code = "def broken(x:\n    return x\n"
        res = generate_mermaid_flowchart(bad_code)
        self.assertFalse(res.success)
        self.assertEqual(res.mermaid_code, "")
        self.assertIsNotNone(res.error_message)
        self.assertIn("SyntaxError", res.error_message)
        self.assertIn("line", res.error_message.lower())

    def test_mermaid_label_escaping(self):
        """Verify that quotes, angle brackets, and brackets are safely escaped."""
        escaped = escape_mermaid_label('x < 10 and y > 20 and msg == "hello" and d["a"] == 1')
        self.assertNotIn('"', escaped)
        self.assertNotIn('<', escaped)
        self.assertNotIn('>', escaped)
        self.assertIn('&lt;', escaped)
        self.assertIn('&gt;', escaped)
        self.assertIn("'", escaped)

        code = (
            "def compare_values(a, b):\n"
            "    if a < b and b > 0:\n"
            "        return 'less < greater'\n"
            "    return 'other'\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        # Raw '<' and '>' must not appear inside node labels
        self.assertIn("&lt;", res.mermaid_code)
        self.assertIn("&gt;", res.mermaid_code)

    def test_deterministic_output(self):
        """Verify that multiple runs on identical code produce identical outputs."""
        code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        run1 = generate_mermaid_flowchart(code)
        run2 = generate_mermaid_flowchart(code)
        run3 = generate_mermaid_flowchart(code)

        self.assertEqual(run1.mermaid_code, run2.mermaid_code)
        self.assertEqual(run2.mermaid_code, run3.mermaid_code)
        self.assertEqual(run1.node_count, run2.node_count)
        self.assertEqual(run1.edge_count, run2.edge_count)

    def test_unusual_and_empty_source(self):
        """Verify empty string and whitespace produce safe fallback."""
        res_empty = generate_mermaid_flowchart("")
        self.assertTrue(res_empty.success)
        self.assertIn("Empty Source", res_empty.mermaid_code)

        res_spaces = generate_mermaid_flowchart("   \n\t  ")
        self.assertTrue(res_spaces.success)
        self.assertIn("Empty Source", res_spaces.mermaid_code)

    def test_class_methods_and_exceptions(self):
        """Verify classes, methods, and try/except handling."""
        code = (
            "class MathService:\n"
            "    def divide(self, a, b):\n"
            "        try:\n"
            "            return a / b\n"
            "        except ZeroDivisionError as err:\n"
            "            raise ValueError('Cannot divide by zero')\n"
        )
        res = generate_mermaid_flowchart(code)
        self.assertTrue(res.success)
        self.assertIn("MathService.divide", res.function_names)
        self.assertIn("try_block", res.supported_constructs)
        self.assertIn("except_handler", res.supported_constructs)
        self.assertIn("raise", res.supported_constructs)


if __name__ == "__main__":
    unittest.main()
