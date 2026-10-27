"""
tests/test_audit.py

Unit test suite for CODEscribe Phase 1 Evidence-Grounded Documentation Audit.

Validates:
- Pure static AST fact extraction without code execution
- Complete documentation consistency across parameters, returns, and signatures
- Detection of missing parameters in the dedicated Parameters section (not merely signature block)
- Missing Markdown sections (Parameters, Returns, Raises)
- Return value consistency: annotations, explicit return values, void functions, and discrepancies
- Exception auditing: explicitly raised exceptions vs documentation mentions
- Generator detection and auditing (yield statements)
- Lexical scoping isolation: nested functions do not bleed statements into outer functions
- Async functions and variadic parameters (*args, **kwargs, defaults)
- Robust syntax error handling and malformed source resilience
- Target function selection in multi-function files
- JSON serializability of audit report
- Regression: Fenced code block comments ignored by Markdown section parser
- Regression: Parameter-name entry matching prevents prose false positives
- Regression: Bare raise auditing requires documentation evidence
- Regression: Explicit "None" / "No exceptions" in Raises section
- Regression: async def and decorated signatures in code blocks
- Regression: Titles appearing after multi-line preambles (> 15 lines)
- Regression: Explicit "None" in void function Returns section
- Regression: Missing target_function handling (FUNCTION_NOT_FOUND)
- Regression: Source code with no functions (NO_FUNCTION_FOUND)
- Regression: return None vs bare return fact extraction
"""

import json
import unittest

from app.audit import (
    AuditFinding,
    AuditReport,
    AuditStatus,
    AuditSummary,
    DocEvidence,
    ParameterFact,
    RaiseFact,
    ReturnFact,
    SourceEvidence,
    audit_documentation,
    extract_ast_facts,
    extract_markdown_evidence,
)


class TestDocumentationAudit(unittest.TestCase):
    """Test suite for app/audit.py Evidence-Grounded Documentation Audit Engine."""

    def test_complete_documentation_passes_all_checks(self):
        """Verify that well-documented code with matching sections achieves CONSISTENT status."""
        code = (
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        )
        markdown = (
            "# `binary_search`\n\n"
            "```python\n"
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "```\n\n"
            "## Description\n\n"
            "Searches for target within sorted array.\n\n"
            "## Parameters\n\n"
            "- **`arr`** (`list[int]`): The sorted array to search.\n"
            "- **`target`** (`int`): Target integer value.\n\n"
            "## Returns\n\n"
            "- **`int`**: Index of the target, or -1 if not found.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.function_name, "binary_search")
        self.assertEqual(report.summary.overall_status, "CONSISTENT")
        self.assertEqual(report.summary.total_parameters, 2)
        self.assertEqual(report.summary.parameters_mentioned, 2)
        self.assertEqual(report.summary.parameters_coverage_ratio, 1.0)
        self.assertEqual(report.summary.return_status, AuditStatus.MENTIONED)
        self.assertEqual(report.summary.exceptions_status, AuditStatus.NOT_APPLICABLE)

    def test_missing_parameter_mentions(self):
        """Verify that parameters missing from the Parameters section are flagged as NOT_MENTIONED."""
        code = (
            "def calculate_total(subtotal: float, tax_rate: float, discount: float) -> float:\n"
            "    return (subtotal * (1.0 + tax_rate)) - discount\n"
        )
        # Markdown only documents `subtotal`, omitting `tax_rate` and `discount`
        markdown = (
            "# `calculate_total`\n\n"
            "```python\n"
            "def calculate_total(subtotal: float, tax_rate: float, discount: float) -> float:\n"
            "```\n\n"
            "## Description\n\n"
            "Computes net invoice total.\n\n"
            "## Parameters\n\n"
            "- **`subtotal`** (`float`): Raw subtotal before tax.\n\n"
            "## Returns\n\n"
            "- **`float`**: Final amount.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.summary.overall_status, "INCOMPLETE")
        self.assertEqual(report.summary.total_parameters, 3)
        self.assertEqual(report.summary.parameters_mentioned, 1)
        self.assertEqual(report.summary.parameters_coverage_ratio, 0.33)

        # Check findings for individual parameters
        findings_by_subject = {f.subject: f for f in report.findings if f.category == "parameter"}
        self.assertEqual(findings_by_subject["subtotal"].status, AuditStatus.MENTIONED)
        self.assertEqual(findings_by_subject["tax_rate"].status, AuditStatus.NOT_MENTIONED)
        self.assertEqual(findings_by_subject["discount"].status, AuditStatus.NOT_MENTIONED)

    def test_parameter_in_signature_code_block_does_not_count_as_documented(self):
        """
        Verify that presence of a parameter name in the signature code block or title
        does not count as documented if the Parameters section is absent.
        """
        code = "def process_data(payload: dict) -> bool:\n    return len(payload) > 0\n"
        # Markdown has signature block containing `payload`, but NO `## Parameters` section
        markdown = (
            "# `process_data`\n\n"
            "```python\n"
            "def process_data(payload: dict) -> bool:\n"
            "```\n\n"
            "## Description\n\n"
            "Processes incoming data dictionary.\n\n"
            "## Returns\n\n"
            "- `bool`: True if valid.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.summary.total_parameters, 1)
        self.assertEqual(report.summary.parameters_mentioned, 0)
        self.assertEqual(report.summary.parameters_coverage_ratio, 0.0)

        param_finding = next(f for f in report.findings if f.subject == "payload")
        self.assertEqual(param_finding.status, AuditStatus.NOT_MENTIONED)
        self.assertIn("Parameters", param_finding.message)

    def test_void_function_with_unexpected_returns_section(self):
        """Verify that a void function documented as returning a value triggers POSSIBLE_DISCREPANCY."""
        code = (
            "def log_event(message: str) -> None:\n"
            "    print(f'[LOG] {message}')\n"
        )
        # Markdown erroneously claims the function returns an integer status code
        markdown = (
            "# `log_event`\n\n"
            "## Description\n\n"
            "Logs event message to console.\n\n"
            "## Parameters\n\n"
            "- `message`: Event text.\n\n"
            "## Returns\n\n"
            "- `int`: 0 on success, 1 on error.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.summary.return_status, AuditStatus.POSSIBLE_DISCREPANCY)
        self.assertEqual(report.summary.overall_status, "DISCREPANCY_DETECTED")

    def test_void_function_without_returns_section_is_not_applicable(self):
        """Verify that a void function appropriately omitting a Returns section is NOT_APPLICABLE."""
        code = "def reset_counter():\n    global counter\n    counter = 0\n"
        markdown = (
            "# `reset_counter`\n\n"
            "## Description\n\n"
            "Resets global counter.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.summary.return_status, AuditStatus.NOT_APPLICABLE)
        self.assertEqual(report.summary.overall_status, "CONSISTENT")

    def test_explicit_exception_handling(self):
        """Verify that explicit raise statements are audited against documentation."""
        code = (
            "def divide(a: float, b: float) -> float:\n"
            "    if b == 0.0:\n"
            "        raise ZeroDivisionError('Division by zero is undefined')\n"
            "    if a < 0:\n"
            "        raise ValueError('Negative numerator not allowed')\n"
            "    return a / b\n"
        )

        # Markdown mentions ZeroDivisionError in Raises section, but omits ValueError
        markdown = (
            "# `divide`\n\n"
            "## Description\n\n"
            "Performs division.\n\n"
            "## Parameters\n\n"
            "- `a`: Numerator.\n"
            "- `b`: Denominator.\n\n"
            "## Returns\n\n"
            "- `float`: Result of a / b.\n\n"
            "## Raises\n\n"
            "- `ZeroDivisionError`: When denominator is zero.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.summary.exceptions_status, AuditStatus.NOT_MENTIONED)
        self.assertEqual(report.summary.overall_status, "INCOMPLETE")

        exc_findings = {f.subject: f for f in report.findings if f.category == "exception"}
        self.assertEqual(exc_findings["ZeroDivisionError"].status, AuditStatus.MENTIONED)
        self.assertEqual(exc_findings["ValueError"].status, AuditStatus.NOT_MENTIONED)

    def test_generator_detection_and_auditing(self):
        """Verify generator functions (yield statements) are audited for yield/stream mentions."""
        code = (
            "def count_up(n: int):\n"
            "    for i in range(n):\n"
            "        yield i\n"
        )

        # Markdown only documents standard returns without mentioning generator semantics
        doc_without_generator = (
            "# `count_up`\n\n"
            "## Description\n\n"
            "Counts numbers up to n.\n\n"
            "## Parameters\n\n"
            "- `n`: Upper bound.\n\n"
            "## Returns\n\n"
            "- `int`: Generated integer.\n"
        )

        report1 = audit_documentation(code, doc_without_generator)
        self.assertTrue(report1.source_evidence.is_generator)
        self.assertEqual(report1.summary.return_status, AuditStatus.POSSIBLE_DISCREPANCY)
        self.assertEqual(report1.summary.overall_status, "DISCREPANCY_DETECTED")

        # Markdown that acknowledges generator/yield semantics
        doc_with_generator = (
            "# `count_up`\n\n"
            "## Description\n\n"
            "Generator yielding numbers up to n.\n\n"
            "## Parameters\n\n"
            "- `n`: Upper bound.\n\n"
            "## Returns\n\n"
            "- Yields sequential integers from 0 to n - 1.\n"
        )

        report2 = audit_documentation(code, doc_with_generator)
        self.assertTrue(report2.source_evidence.is_generator)
        self.assertEqual(report2.summary.return_status, AuditStatus.MENTIONED)
        self.assertEqual(report2.summary.overall_status, "CONSISTENT")

    def test_nested_functions_lexical_scoping(self):
        """
        Verify that nested function statements (returns, raises, yields, parameters)
        do not bleed into the outer target function's facts.
        """
        code = (
            "def outer(x: int) -> int:\n"
            "    def inner(y: int):\n"
            "        if y < 0:\n"
            "            raise ValueError('negative y')\n"
            "        return y * 2\n"
            "    return inner(x)\n"
        )

        source_facts = extract_ast_facts(code, target_function="outer")
        self.assertEqual(source_facts.function_name, "outer")
        self.assertEqual([p.name for p in source_facts.parameters], ["x"])
        self.assertEqual(source_facts.nested_functions, ["inner"])
        # Outer itself does not raise ValueError; inner does
        self.assertEqual(len(source_facts.raised_exceptions), 0)
        # Outer has 1 return statement: `return inner(x)`
        self.assertEqual(len(source_facts.return_statements), 1)

        # Audit outer with matching documentation
        markdown = (
            "# `outer`\n\n"
            "## Description\n\n"
            "Processes x using nested helper.\n\n"
            "## Parameters\n\n"
            "- `x` (`int`): Input number.\n\n"
            "## Returns\n\n"
            "- `int`: Transformed integer.\n"
        )

        report = audit_documentation(code, markdown, target_function="outer")
        self.assertTrue(report.success)
        self.assertEqual(report.summary.overall_status, "CONSISTENT")
        self.assertEqual(report.summary.total_parameters, 1)

    def test_async_function_support(self):
        """Verify async function definitions are recognized and audited safely."""
        code = (
            "async def fetch_user(user_id: str) -> dict:\n"
            "    res = await client.get(f'/users/{user_id}')\n"
            "    return res.json()\n"
        )
        source_facts = extract_ast_facts(code)
        self.assertTrue(source_facts.is_async)
        self.assertEqual(source_facts.function_name, "fetch_user")

        markdown = (
            "# `fetch_user`\n\n"
            "## Description\n\n"
            "Asynchronously retrieves user profile.\n\n"
            "## Parameters\n\n"
            "- `user_id`: Target user identifier.\n\n"
            "## Returns\n\n"
            "- `dict`: User profile object.\n"
        )
        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertTrue(report.source_evidence.is_async)
        self.assertEqual(report.summary.overall_status, "CONSISTENT")

    def test_default_parameters_and_variadic_args(self):
        """Verify positional-only, defaults, *args, kw-only, and **kwargs are accurately captured."""
        code = (
            "def complex_signature(a: int, b: str = 'default', *args: int, flag: bool = True, **kwargs):\n"
            "    return True\n"
        )
        source_facts = extract_ast_facts(code)
        param_map = {p.name: p for p in source_facts.parameters}

        self.assertIn("a", param_map)
        self.assertFalse(param_map["a"].has_default)

        self.assertIn("b", param_map)
        self.assertTrue(param_map["b"].has_default)
        self.assertEqual(param_map["b"].default_repr, "'default'")

        self.assertIn("*args", param_map)
        self.assertEqual(param_map["*args"].kind, "vararg")

        self.assertIn("flag", param_map)
        self.assertEqual(param_map["flag"].kind, "kwonly")
        self.assertTrue(param_map["flag"].has_default)
        self.assertEqual(param_map["flag"].default_repr, "True")

        self.assertIn("**kwargs", param_map)
        self.assertEqual(param_map["**kwargs"].kind, "kwarg")

    def test_syntax_error_handling(self):
        """Verify invalid Python syntax fails safely without crashing and produces SYNTAX_ERROR status."""
        code = "def broken_code(a, b\n    return a + b"  # Missing closing paren
        markdown = "# `broken_code`\n"

        report = audit_documentation(code, markdown)
        self.assertFalse(report.success)
        self.assertEqual(report.summary.overall_status, "SYNTAX_ERROR")
        self.assertFalse(report.source_evidence.syntax_valid)
        self.assertIsNotNone(report.source_evidence.syntax_error)
        self.assertIn("Syntax error", report.source_evidence.syntax_error)

    def test_empty_source_code_handling(self):
        """Verify empty source code string returns a safe error report."""
        report = audit_documentation("", "# empty\n")
        self.assertFalse(report.success)
        self.assertEqual(report.summary.overall_status, "SYNTAX_ERROR")

    def test_target_function_selection(self):
        """Verify target_function parameter selects the intended function in a multi-function file."""
        code = (
            "def helper():\n"
            "    return 1\n\n"
            "def main_process(name: str) -> str:\n"
            "    return f'Hello {name}'\n"
        )
        # Target main_process specifically
        source_facts = extract_ast_facts(code, target_function="main_process")
        self.assertEqual(source_facts.function_name, "main_process")
        self.assertEqual([p.name for p in source_facts.parameters], ["name"])

    def test_report_to_dict_and_json_serializability(self):
        """Verify that the full AuditReport converts to a dictionary and is JSON-serializable."""
        code = "def ping() -> str:\n    return 'pong'\n"
        markdown = (
            "# `ping`\n\n"
            "## Description\n\n"
            "Health check ping.\n\n"
            "## Returns\n\n"
            "- `str`: Pong response.\n"
        )

        report = audit_documentation(code, markdown)
        report_dict = report.to_dict()
        serialized = json.dumps(report_dict)
        self.assertIsInstance(serialized, str)

        parsed = json.loads(serialized)
        self.assertEqual(parsed["function_name"], "ping")
        self.assertEqual(parsed["summary"]["overall_status"], "CONSISTENT")
        self.assertIn("disclaimer", parsed)
        self.assertTrue(len(parsed["findings"]) > 0)
        self.assertTrue(parsed["findings"][0]["is_heuristic"])

    # ========================================================================
    # Focused Regression Tests for Confirmed Review Issues
    # ========================================================================

    def test_fenced_code_block_headings_ignored(self):
        """Verify that # comments inside fenced code blocks do not fragment sections."""
        markdown = (
            "## Description\n\n"
            "Overview of the algorithm.\n\n"
            "```python\n"
            "# Parameters to consider:\n"
            "# This is a Python comment, not a heading\n"
            "x = 10\n"
            "```\n\n"
            "## Parameters\n\n"
            "- **`a`**: First parameter.\n"
        )
        sections, doc_evidence = extract_markdown_evidence(markdown)

        # Ensure '# Parameters to consider:' was not treated as a heading
        self.assertNotIn("parameters to consider:", sections)
        self.assertIn("description", sections)
        self.assertIn("parameters", sections)
        # Description should contain the entire code block including comments
        self.assertIn("# Parameters to consider:", sections["description"])
        # Parameters section should only contain parameter 'a'
        self.assertIn("- **`a`**: First parameter.", sections["parameters"])

    def test_parameter_entry_matching_prevents_false_positive_prose(self):
        """
        Verify that words like 'a', 'to', 'count', 'data' in ordinary prose do not
        falsely count short or common parameter names as documented.
        """
        code = "def process(a: int, target: int) -> int:\n    return a + target\n"
        # Documentation documents `target`, but completely omits `a`.
        # However, the description for `target` uses the English article ' a '.
        markdown = (
            "# `process`\n\n"
            "## Description\n\n"
            "Processes items.\n\n"
            "## Parameters\n\n"
            "- **`target`**: Takes a target index within an array.\n\n"
            "## Returns\n\n"
            "- `int`: Sum.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        findings_by_subject = {f.subject: f for f in report.findings if f.category == "parameter"}

        # Target should be mentioned; 'a' MUST be NOT_MENTIONED despite ' a ' in prose
        self.assertEqual(findings_by_subject["target"].status, AuditStatus.MENTIONED)
        self.assertEqual(findings_by_subject["a"].status, AuditStatus.NOT_MENTIONED)
        self.assertEqual(report.summary.parameters_mentioned, 1)
        self.assertEqual(report.summary.parameters_coverage_ratio, 0.5)

    def test_bare_raise_auditing(self):
        """Verify bare raise statements are never marked MENTIONED without documentation evidence."""
        code = (
            "def retry_task():\n"
            "    try:\n"
            "        do_work()\n"
            "    except Exception:\n"
            "        cleanup()\n"
            "        raise\n"
        )

        # Documentation omits any mention of exception propagation
        doc_omitting_raise = (
            "# `retry_task`\n\n"
            "## Description\n\n"
            "Retries work.\n"
        )
        report1 = audit_documentation(code, doc_omitting_raise)
        self.assertEqual(report1.summary.exceptions_status, AuditStatus.NOT_MENTIONED)
        reraise_finding = next(f for f in report1.findings if f.subject == "<re-raise>")
        self.assertEqual(reraise_finding.status, AuditStatus.NOT_MENTIONED)

        # Documentation mentions exception raising / propagation
        doc_acknowledging_raise = (
            "# `retry_task`\n\n"
            "## Description\n\n"
            "Retries work; re-raises caught exceptions after cleanup.\n"
        )
        report2 = audit_documentation(code, doc_acknowledging_raise)
        self.assertEqual(report2.summary.exceptions_status, AuditStatus.MENTIONED)
        reraise_finding2 = next(f for f in report2.findings if f.subject == "<re-raise>")
        self.assertEqual(reraise_finding2.status, AuditStatus.MENTIONED)

    def test_explicit_none_no_exceptions_in_raises(self):
        """Verify that a Raises section explicitly stating 'None' or 'No exceptions' is NOT_APPLICABLE."""
        code = "def simple_add(a: int, b: int) -> int:\n    return a + b\n"
        markdown = (
            "# `simple_add`\n\n"
            "## Description\n\n"
            "Adds two integers.\n\n"
            "## Parameters\n\n"
            "- **`a`**: First.\n"
            "- **`b`**: Second.\n\n"
            "## Returns\n\n"
            "- `int`: Total.\n\n"
            "## Raises\n\n"
            "No exceptions are raised.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        # Should evaluate to NOT_APPLICABLE, not POSSIBLE_DISCREPANCY
        self.assertEqual(report.summary.exceptions_status, AuditStatus.NOT_APPLICABLE)
        self.assertEqual(report.summary.overall_status, "CONSISTENT")

    def test_async_def_and_decorated_signatures_in_code_block(self):
        """Verify async def and decorated functions are extracted in signature code blocks."""
        doc_async = (
            "```python\n"
            "async def fetch_status(endpoint: str) -> dict:\n"
            "```\n\n"
            "## Description\n\n"
            "Fetches status asynchronously.\n"
        )
        _, doc_ev_async = extract_markdown_evidence(doc_async)
        self.assertTrue(doc_ev_async.signature_block_found)
        self.assertIn("async def fetch_status", doc_ev_async.signature_text or "")

        doc_decorated = (
            "```python\n"
            "@classmethod\n"
            "def create_instance(cls, config: dict):\n"
            "```\n\n"
            "## Description\n\n"
            "Factory method.\n"
        )
        _, doc_ev_dec = extract_markdown_evidence(doc_decorated)
        self.assertTrue(doc_ev_dec.signature_block_found)
        self.assertIn("def create_instance", doc_ev_dec.signature_text or "")

    def test_header_after_line_15(self):
        """Verify function name is found in header even after long multi-line preambles."""
        code = "def compute():\n    return 42\n"
        # 25 lines of preamble before the function header
        preamble = "\n".join([f"<!-- license or comment line {i} -->" for i in range(25)])
        markdown = (
            f"{preamble}\n\n"
            "# `compute`\n\n"
            "## Description\n\n"
            "Computes result.\n\n"
            "## Returns\n\n"
            "- `int`: Result 42.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        sig_finding = next(f for f in report.findings if f.category == "signature")
        self.assertEqual(sig_finding.status, AuditStatus.MENTIONED)

    def test_void_function_with_explicit_none_return(self):
        """Verify void function documenting 'None' in Returns section is NOT_APPLICABLE with accurate message."""
        code = "def close_connection():\n    pass\n"
        markdown = (
            "# `close_connection`\n\n"
            "## Description\n\n"
            "Closes active socket.\n\n"
            "## Returns\n\n"
            "- `None`: Does not return a value.\n"
        )

        report = audit_documentation(code, markdown)
        self.assertTrue(report.success)
        self.assertEqual(report.summary.return_status, AuditStatus.NOT_APPLICABLE)
        ret_finding = next(f for f in report.findings if f.category == "return")
        self.assertEqual(ret_finding.status, AuditStatus.NOT_APPLICABLE)
        self.assertIn("documents None", ret_finding.message)

    def test_missing_target_function_explicitly_fails(self):
        """Verify passing a non-existent target_function fails explicitly without fallback."""
        code = "def actual_function():\n    return 1\n"
        report = audit_documentation(code, "# doc", target_function="non_existent")

        self.assertFalse(report.success)
        self.assertEqual(report.summary.overall_status, "FUNCTION_NOT_FOUND")
        target_finding = next(f for f in report.findings if f.category == "target")
        self.assertIn("not found in source code", target_finding.message)

    def test_source_code_with_no_functions(self):
        """Verify module/script code without functions reports NO_FUNCTION_FOUND gracefully."""
        code = "x = 10\ny = 20\ntotal = x + y\n"
        markdown = "# Script\nExecutes calculations.\n"

        report = audit_documentation(code, markdown)
        self.assertFalse(report.success)
        self.assertEqual(report.summary.overall_status, "NO_FUNCTION_FOUND")
        struct_finding = next(f for f in report.findings if f.category == "structure")
        self.assertIn("no function definition", struct_finding.message.lower())

    def test_return_none_versus_bare_return_facts(self):
        """Verify return None vs bare return statement fact extraction in AST."""
        code = (
            "def test_returns(mode: int):\n"
            "    if mode == 1:\n"
            "        return\n"
            "    elif mode == 2:\n"
            "        return None\n"
            "    else:\n"
            "        return 'valid'\n"
        )
        source_facts = extract_ast_facts(code)
        self.assertEqual(len(source_facts.return_statements), 3)

        ret_bare = source_facts.return_statements[0]
        self.assertFalse(ret_bare.has_value)
        self.assertIsNone(ret_bare.value_repr)

        ret_none = source_facts.return_statements[1]
        self.assertTrue(ret_none.has_value)
        self.assertEqual(ret_none.value_repr, "None")

        ret_val = source_facts.return_statements[2]
        self.assertTrue(ret_val.has_value)
        self.assertEqual(ret_val.value_repr, "'valid'")


if __name__ == "__main__":
    unittest.main()
