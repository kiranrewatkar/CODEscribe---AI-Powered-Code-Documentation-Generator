"""
evaluation_workbook_runner.py

Temporary, standalone evaluation runner for CODEscribe's Evidence-Grounded
Documentation Audit Engine.

Evaluates 20 representative and rigorous audit workbook cases covering:
1. Standard, fully documented function
2. Partially documented parameters (missing arguments)
3. Parameter in signature code block only (missing Parameters section)
4. Parameter in ordinary prose (e.g. English article 'a', 'target', 'data')
5. Void function omitting Returns section appropriately
6. Void function documenting None appropriately
7. Void function with unexpected Returns section claiming a return value
8. Single return statement with return annotation
9. Multiple return statements with consistent documentation
10. Multiple return statements with missing Returns section
11. Explicit typed exception (e.g. ValueError) documented
12. Explicit typed exception (e.g. ValueError) omitted from documentation
13. Bare re-raise (raise) with exception propagation documented
14. Bare re-raise (raise) with exception documentation omitted
15. Function raising no exceptions with explicit "No exceptions" in documentation
16. Function raising no exceptions with unexpected Raises section documenting errors
17. Generator function with yield semantics documented
18. Generator function with yield semantics omitted from documentation
19. Nested function scoping (inner function returns/raises isolated from outer)
20. Async function with decorator and multi-line preamble

Safety & Isolation:
- Never executes user-submitted source code (pure AST & lexical matching).
- Kept outside application and production code.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, NamedTuple, Optional

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.audit import AuditReport, AuditStatus, audit_documentation


class WorkbookCase(NamedTuple):
    case_id: int
    name: str
    category: str
    source_code: str
    markdown_doc: str
    target_function: Optional[str]
    expected_overall_status: str
    expected_param_coverage: Optional[float]
    expected_return_status: AuditStatus
    expected_exceptions_status: AuditStatus
    description: str


WORKBOOK_CASES: List[WorkbookCase] = [
    # Case 1: Standard fully documented function
    WorkbookCase(
        case_id=1,
        name="Fully documented binary search",
        category="parameter_and_return",
        source_code=(
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
        ),
        markdown_doc=(
            "# `binary_search`\n\n"
            "```python\n"
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "```\n\n"
            "## Description\n\n"
            "Performs binary search on a sorted array.\n\n"
            "## Parameters\n\n"
            "- **`arr`** (`list[int]`): Sorted array to search.\n"
            "- **`target`** (`int`): Value to find.\n\n"
            "## Returns\n\n"
            "- **`int`**: Index of element or -1.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="All parameters, return annotation, and signature match documentation.",
    ),

    # Case 2: Partially documented parameters (missing arguments)
    WorkbookCase(
        case_id=2,
        name="Partially documented function (omitted parameters)",
        category="parameter",
        source_code=(
            "def calculate_total(subtotal: float, tax_rate: float, discount: float) -> float:\n"
            "    return (subtotal * (1.0 + tax_rate)) - discount\n"
        ),
        markdown_doc=(
            "# `calculate_total`\n\n"
            "## Description\n\n"
            "Computes total bill.\n\n"
            "## Parameters\n\n"
            "- **`subtotal`** (`float`): Raw subtotal.\n\n"
            "## Returns\n\n"
            "- `float`: Final computed total.\n"
        ),
        target_function=None,
        expected_overall_status="INCOMPLETE",
        expected_param_coverage=0.33,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Parameters tax_rate and discount are omitted from Parameters section.",
    ),

    # Case 3: Parameter in signature code block only
    WorkbookCase(
        case_id=3,
        name="Parameter in signature code block but missing Parameters section",
        category="parameter",
        source_code=(
            "def process_data(payload: dict) -> bool:\n"
            "    return len(payload) > 0\n"
        ),
        markdown_doc=(
            "# `process_data`\n\n"
            "```python\n"
            "def process_data(payload: dict) -> bool:\n"
            "```\n\n"
            "## Description\n\n"
            "Processes incoming data dictionary.\n\n"
            "## Returns\n\n"
            "- `bool`: True if non-empty.\n"
        ),
        target_function=None,
        expected_overall_status="INCOMPLETE",
        expected_param_coverage=0.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Presence in code block does not count as documented in Parameters section.",
    ),

    # Case 4: Parameter mentioned only in ordinary prose
    WorkbookCase(
        case_id=4,
        name="Parameter mentioned in ordinary prose of another item",
        category="parameter",
        source_code=(
            "def process(a: int, target: int) -> int:\n"
            "    return a + target\n"
        ),
        markdown_doc=(
            "# `process`\n\n"
            "## Description\n\n"
            "Processes input values.\n\n"
            "## Parameters\n\n"
            "- **`target`**: Takes a target index within an array.\n\n"
            "## Returns\n\n"
            "- `int`: Sum.\n"
        ),
        target_function=None,
        expected_overall_status="INCOMPLETE",
        expected_param_coverage=0.5,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="English article ' a ' in target's prose must not count as defining parameter 'a'.",
    ),

    # Case 5: Void function omitting Returns section appropriately
    WorkbookCase(
        case_id=5,
        name="Void function omitting Returns section",
        category="return",
        source_code=(
            "def reset_counter():\n"
            "    global counter\n"
            "    counter = 0\n"
        ),
        markdown_doc=(
            "# `reset_counter`\n\n"
            "## Description\n\n"
            "Resets global counter to zero.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=None,
        expected_return_status=AuditStatus.NOT_APPLICABLE,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Void function has no return value; Returns section appropriately omitted.",
    ),

    # Case 6: Void function documenting None appropriately
    WorkbookCase(
        case_id=6,
        name="Void function documenting None in Returns section",
        category="return",
        source_code=(
            "def close_socket(sock) -> None:\n"
            "    sock.close()\n"
        ),
        markdown_doc=(
            "# `close_socket`\n\n"
            "## Description\n\n"
            "Closes socket cleanly.\n\n"
            "## Parameters\n\n"
            "- **`sock`**: The socket instance.\n\n"
            "## Returns\n\n"
            "- `None`: Does not return any value.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.NOT_APPLICABLE,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Explicit documentation of None is recognized as NOT_APPLICABLE without discrepancy.",
    ),

    # Case 7: Void function with unexpected Returns section claiming a value
    WorkbookCase(
        case_id=7,
        name="Void function with unexpected Returns section claiming return value",
        category="return",
        source_code=(
            "def log_message(msg: str):\n"
            "    print(msg)\n"
        ),
        markdown_doc=(
            "# `log_message`\n\n"
            "## Description\n\n"
            "Prints log message.\n\n"
            "## Parameters\n\n"
            "- **`msg`**: Message string.\n\n"
            "## Returns\n\n"
            "- `int`: 0 on success, 1 on error.\n"
        ),
        target_function=None,
        expected_overall_status="DISCREPANCY_DETECTED",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.POSSIBLE_DISCREPANCY,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Documentation claims an int return value for a function that returns nothing.",
    ),

    # Case 8: Single return statement with return annotation
    WorkbookCase(
        case_id=8,
        name="Single return statement with return annotation",
        category="return",
        source_code=(
            "def square(x: float) -> float:\n"
            "    return x * x\n"
        ),
        markdown_doc=(
            "# `square`\n\n"
            "## Description\n\n"
            "Computes square of a number.\n\n"
            "## Parameters\n\n"
            "- **`x`** (`float`): Input number.\n\n"
            "## Returns\n\n"
            "- **`float`**: The squared value.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Standard single-return function matches documentation.",
    ),

    # Case 9: Multiple return statements with consistent documentation
    WorkbookCase(
        case_id=9,
        name="Multiple return branches consistently documented",
        category="return",
        source_code=(
            "def sign(x: int) -> int:\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    elif x < 0:\n"
            "        return -1\n"
            "    return 0\n"
        ),
        markdown_doc=(
            "# `sign`\n\n"
            "## Description\n\n"
            "Determines sign of integer.\n\n"
            "## Parameters\n\n"
            "- **`x`**: Input integer.\n\n"
            "## Returns\n\n"
            "- `int`: 1 for positive, -1 for negative, 0 for zero.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Multiple return branches all return values and match Returns section.",
    ),

    # Case 10: Multiple return statements with missing Returns section
    WorkbookCase(
        case_id=10,
        name="Multiple return statements with missing Returns section",
        category="return",
        source_code=(
            "def classify(val: int) -> str:\n"
            "    if val >= 0:\n"
            "        return 'pos'\n"
            "    return 'neg'\n"
        ),
        markdown_doc=(
            "# `classify`\n\n"
            "## Description\n\n"
            "Classifies value sign.\n\n"
            "## Parameters\n\n"
            "- **`val`**: Input value.\n"
        ),
        target_function=None,
        expected_overall_status="INCOMPLETE",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.NOT_MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Source returns strings, but documentation omits Returns section.",
    ),

    # Case 11: Explicit typed exception documented
    WorkbookCase(
        case_id=11,
        name="Explicit exception documented in Raises section",
        category="exception",
        source_code=(
            "def safe_divide(a: float, b: float) -> float:\n"
            "    if b == 0.0:\n"
            "        raise ValueError('Cannot divide by zero')\n"
            "    return a / b\n"
        ),
        markdown_doc=(
            "# `safe_divide`\n\n"
            "## Description\n\n"
            "Divides two numbers safely.\n\n"
            "## Parameters\n\n"
            "- **`a`**: Numerator.\n"
            "- **`b`**: Denominator.\n\n"
            "## Returns\n\n"
            "- `float`: Quotient.\n\n"
            "## Raises\n\n"
            "- `ValueError`: If denominator is zero.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.MENTIONED,
        description="Explicit ValueError raised in source is documented in Raises section.",
    ),

    # Case 12: Explicit typed exception omitted from documentation
    WorkbookCase(
        case_id=12,
        name="Explicit exception omitted from documentation",
        category="exception",
        source_code=(
            "def parse_positive(text: str) -> int:\n"
            "    val = int(text)\n"
            "    if val < 0:\n"
            "        raise ValueError('Must be positive')\n"
            "    return val\n"
        ),
        markdown_doc=(
            "# `parse_positive`\n\n"
            "## Description\n\n"
            "Parses positive integer from text.\n\n"
            "## Parameters\n\n"
            "- **`text`**: String input.\n\n"
            "## Returns\n\n"
            "- `int`: Parsed positive integer.\n"
        ),
        target_function=None,
        expected_overall_status="INCOMPLETE",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_MENTIONED,
        description="Explicit raise ValueError in source is omitted from documentation.",
    ),

    # Case 13: Bare re-raise with exception propagation documented
    WorkbookCase(
        case_id=13,
        name="Bare raise with exception handling documented",
        category="exception",
        source_code=(
            "def execute_task():\n"
            "    try:\n"
            "        run_step()\n"
            "    except Exception:\n"
            "        cleanup()\n"
            "        raise\n"
        ),
        markdown_doc=(
            "# `execute_task`\n\n"
            "## Description\n\n"
            "Executes task; catches and re-raises exceptions after cleanup.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=None,
        expected_return_status=AuditStatus.NOT_APPLICABLE,
        expected_exceptions_status=AuditStatus.MENTIONED,
        description="Bare raise is recognized and acknowledged in documentation text.",
    ),

    # Case 14: Bare re-raise with exception documentation omitted
    WorkbookCase(
        case_id=14,
        name="Bare raise with exception documentation omitted",
        category="exception",
        source_code=(
            "def retry_task():\n"
            "    try:\n"
            "        call_service()\n"
            "    except Exception:\n"
            "        notify_admin()\n"
            "        raise\n"
        ),
        markdown_doc=(
            "# `retry_task`\n\n"
            "## Description\n\n"
            "Retries external service call.\n"
        ),
        target_function=None,
        expected_overall_status="INCOMPLETE",
        expected_param_coverage=None,
        expected_return_status=AuditStatus.NOT_APPLICABLE,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE if False else AuditStatus.NOT_MENTIONED,
        description="Bare raise is omitted from documentation, resulting in NOT_MENTIONED.",
    ),

    # Case 15: Function raising no exceptions with explicit "No exceptions"
    WorkbookCase(
        case_id=15,
        name="No exceptions raised with explicit 'No exceptions' in documentation",
        category="exception",
        source_code=(
            "def add_two(x: int, y: int) -> int:\n"
            "    return x + y\n"
        ),
        markdown_doc=(
            "# `add_two`\n\n"
            "## Description\n\n"
            "Adds two numbers.\n\n"
            "## Parameters\n\n"
            "- **`x`**: First.\n"
            "- **`y`**: Second.\n\n"
            "## Returns\n\n"
            "- `int`: Total.\n\n"
            "## Raises\n\n"
            "No exceptions are raised.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Explicit declaration of no exceptions is accepted without discrepancy.",
    ),

    # Case 16: Function raising no exceptions with unexpected Raises section documenting errors
    WorkbookCase(
        case_id=16,
        name="No exceptions in source but documentation claims exceptions",
        category="exception",
        source_code=(
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n"
        ),
        markdown_doc=(
            "# `multiply`\n\n"
            "## Description\n\n"
            "Multiplies two numbers.\n\n"
            "## Parameters\n\n"
            "- **`a`**: First.\n"
            "- **`b`**: Second.\n\n"
            "## Returns\n\n"
            "- `int`: Product.\n\n"
            "## Raises\n\n"
            "- `ArithmeticError`: If multiplication overflows.\n"
        ),
        target_function=None,
        expected_overall_status="DISCREPANCY_DETECTED",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.POSSIBLE_DISCREPANCY,
        description="Documentation documents an exception that is never explicitly raised in AST.",
    ),

    # Case 17: Generator function with yield semantics documented
    WorkbookCase(
        case_id=17,
        name="Generator function with yield documented",
        category="generator",
        source_code=(
            "def count_up(n: int):\n"
            "    for i in range(n):\n"
            "        yield i\n"
        ),
        markdown_doc=(
            "# `count_up`\n\n"
            "## Description\n\n"
            "Generator yielding numbers up to n.\n\n"
            "## Parameters\n\n"
            "- **`n`**: Maximum count.\n\n"
            "## Returns\n\n"
            "- Yields sequential integers from 0 to n - 1.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Generator function yield semantics acknowledged in documentation.",
    ),

    # Case 18: Generator function with yield semantics omitted
    WorkbookCase(
        case_id=18,
        name="Generator function omitting yield/stream semantics",
        category="generator",
        source_code=(
            "def generate_ids(count: int):\n"
            "    for i in range(count):\n"
            "        yield f'id_{i}'\n"
        ),
        markdown_doc=(
            "# `generate_ids`\n\n"
            "## Description\n\n"
            "Generates identifier string.\n\n"
            "## Parameters\n\n"
            "- **`count`**: Number of ids.\n\n"
            "## Returns\n\n"
            "- `str`: Identifier string.\n"
        ),
        target_function=None,
        expected_overall_status="DISCREPANCY_DETECTED",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.POSSIBLE_DISCREPANCY,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Documentation claims single return value for a generator function.",
    ),

    # Case 19: Nested function scoping isolation
    WorkbookCase(
        case_id=19,
        name="Nested function scoping isolation",
        category="scoping",
        source_code=(
            "def outer(x: int) -> int:\n"
            "    def inner(y: int):\n"
            "        if y < 0:\n"
            "            raise ValueError('negative y')\n"
            "        return y * 2\n"
            "    return inner(x)\n"
        ),
        markdown_doc=(
            "# `outer`\n\n"
            "## Description\n\n"
            "Processes x using nested helper.\n\n"
            "## Parameters\n\n"
            "- **`x`** (`int`): Input number.\n\n"
            "## Returns\n\n"
            "- `int`: Transformed integer.\n"
        ),
        target_function="outer",
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Inner function's parameter y and ValueError do not bleed into outer function.",
    ),

    # Case 20: Async function with decorator and multi-line preamble
    WorkbookCase(
        case_id=20,
        name="Async function with decorator and multi-line preamble",
        category="async_and_preamble",
        source_code=(
            "async def fetch_user(user_id: str) -> dict:\n"
            "    res = await client.get(f'/users/{user_id}')\n"
            "    return res.json()\n"
        ),
        markdown_doc=(
            "<!-- Project Header -->\n"
            "<!-- Build: Passing -->\n"
            "<!-- License: Apache 2.0 -->\n"
            "<!-- Line 4 -->\n"
            "<!-- Line 5 -->\n"
            "<!-- Line 6 -->\n"
            "<!-- Line 7 -->\n"
            "<!-- Line 8 -->\n"
            "<!-- Line 9 -->\n"
            "<!-- Line 10 -->\n"
            "<!-- Line 11 -->\n"
            "<!-- Line 12 -->\n"
            "<!-- Line 13 -->\n"
            "<!-- Line 14 -->\n"
            "<!-- Line 15 -->\n"
            "<!-- Line 16 -->\n"
            "<!-- Line 17 -->\n"
            "<!-- Line 18 -->\n\n"
            "# `fetch_user`\n\n"
            "```python\n"
            "@router.get('/users/{user_id}')\n"
            "async def fetch_user(user_id: str) -> dict:\n"
            "```\n\n"
            "## Description\n\n"
            "Asynchronously fetches user data from backend service.\n\n"
            "## Parameters\n\n"
            "- **`user_id`**: Identifier string.\n\n"
            "## Returns\n\n"
            "- `dict`: JSON user dictionary.\n"
        ),
        target_function=None,
        expected_overall_status="CONSISTENT",
        expected_param_coverage=1.0,
        expected_return_status=AuditStatus.MENTIONED,
        expected_exceptions_status=AuditStatus.NOT_APPLICABLE,
        description="Header appears after 18 lines; decorated async signature in code block matches.",
    ),
]


def run_evaluation() -> Dict[str, Any]:
    """Execute all 20 workbook cases and compile structured evaluation results."""
    case_results = []
    total_cases = len(WORKBOOK_CASES)
    passed_cases = 0
    ambiguous_cases = 0
    failed_cases = 0

    for case in WORKBOOK_CASES:
        report: AuditReport = audit_documentation(
            source_code=case.source_code,
            markdown_doc=case.markdown_doc,
            target_function=case.target_function,
        )

        overall_match = report.summary.overall_status == case.expected_overall_status
        return_match = report.summary.return_status == case.expected_return_status
        exc_match = report.summary.exceptions_status == case.expected_exceptions_status

        # Parameter coverage match
        actual_cov = report.summary.parameters_coverage_ratio
        if case.expected_param_coverage is None:
            cov_match = actual_cov is None
        else:
            cov_match = (actual_cov is not None) and (abs(actual_cov - case.expected_param_coverage) < 0.05)

        is_passed = overall_match and return_match and exc_match and cov_match

        if is_passed:
            passed_cases += 1
            verdict = "PASS"
        else:
            failed_cases += 1
            verdict = "FAIL"

        key_findings = [
            {
                "category": f.category,
                "subject": f.subject,
                "status": f.status.value,
                "message": f.message,
            }
            for f in report.findings
        ]

        case_results.append({
            "case_id": case.case_id,
            "name": case.name,
            "category": case.category,
            "description": case.description,
            "expected": {
                "overall_status": case.expected_overall_status,
                "parameters_coverage_ratio": case.expected_param_coverage,
                "return_status": case.expected_return_status.value,
                "exceptions_status": case.expected_exceptions_status.value,
            },
            "actual": {
                "overall_status": report.summary.overall_status,
                "parameters_coverage_ratio": report.summary.parameters_coverage_ratio,
                "return_status": report.summary.return_status.value,
                "exceptions_status": report.summary.exceptions_status.value,
            },
            "verdict": verdict,
            "matches": {
                "overall_status": overall_match,
                "parameter_coverage": cov_match,
                "return_status": return_match,
                "exceptions_status": exc_match,
            },
            "key_findings": key_findings,
            "source_evidence": {
                "function_name": report.source_evidence.function_name,
                "is_async": report.source_evidence.is_async,
                "is_generator": report.source_evidence.is_generator,
                "parameters": [p.name for p in report.source_evidence.parameters],
                "return_annotation": report.source_evidence.return_annotation,
                "raised_exceptions": [e.exception_type for e in report.source_evidence.raised_exceptions],
            },
        })

    summary = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "ambiguous_cases": ambiguous_cases,
        "pass_rate_percentage": round((passed_cases / total_cases) * 100, 2),
    }

    return {
        "summary": summary,
        "case_results": case_results,
    }


if __name__ == "__main__":
    eval_result = run_evaluation()
    print(json.dumps(eval_result, indent=2))
