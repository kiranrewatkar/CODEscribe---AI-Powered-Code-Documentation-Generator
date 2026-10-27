"""
app/optimization.py

Lightweight, standalone Python Static Code Optimization Suggester for CODEscribe.

Analyzes Python source code using standard library AST to identify practical,
structural optimization patterns without executing or evaluating user code.

Rules Supported:
- OPT001: Mutable list/dict/set default arguments in regular and async functions.
- OPT002: Inefficient membership check against literal lists or tuples.
- OPT003: Redundant list comprehension passed to reduction functions (sum, any, all, min, max).
- OPT004: In-loop augmented addition (potential string concatenation accumulator).

Core Guarantees:
- Pure static AST analysis: zero code execution (no eval, exec, or imports).
- Deterministic, dependency-free (standard library only).
- Proper lexical scope tracking for nested functions and classes.
- Explicit non-binding heuristic disclaimer.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("CODEscribe.Optimization")

OPTIMIZATION_DISCLAIMER = (
    "Optimization suggestions are heuristic static observations derived solely "
    "from Python AST structure. They do not analyze runtime types, dynamic data "
    "flow, or memory profiles, and do not guarantee runtime performance improvements. "
    "Source code was not executed or evaluated."
)

REDUCTION_FUNCTIONS: Set[str] = {"sum", "any", "all", "min", "max"}


@dataclass
class OptimizationFinding:
    """Represents an individual static optimization finding."""

    rule_id: str
    title: str
    description: str
    suggestion: str
    line_number: int
    column_offset: int
    function_scope: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to a serializable dictionary."""
        return asdict(self)


@dataclass
class OptimizationResult:
    """Result container for static optimization analysis."""

    success: bool
    findings: List[OptimizationFinding] = field(default_factory=list)
    error_message: Optional[str] = None
    disclaimer: str = OPTIMIZATION_DISCLAIMER
    total_findings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "success": self.success,
            "findings": [f.to_dict() for f in self.findings],
            "error_message": self.error_message,
            "disclaimer": self.disclaimer,
            "total_findings": self.total_findings,
        }


class OptimizationVisitor(ast.NodeVisitor):
    """
    AST visitor inspecting Python syntax trees for isolated optimization patterns.
    Tracks lexical function scopes and active loops cleanly without inter-procedural leakage.
    """

    def __init__(self) -> None:
        self.findings: List[OptimizationFinding] = []
        self._seen_keys: Set[Tuple[str, int, int]] = set()
        self._scope_stack: List[str] = []
        self._loop_depth: int = 0

    @property
    def current_function_scope(self) -> Optional[str]:
        """Return the current innermost function scope name, or None if at module level."""
        for scope in reversed(self._scope_stack):
            if not scope.startswith("<class:"):
                return scope
        return None

    def _add_finding(
        self,
        rule_id: str,
        title: str,
        description: str,
        suggestion: str,
        node: ast.AST,
    ) -> None:
        """Register a finding deduplicated by (rule_id, line_number, column_offset)."""
        lineno = getattr(node, "lineno", 1)
        col_offset = getattr(node, "col_offset", 0)
        key = (rule_id, lineno, col_offset)
        if key in self._seen_keys:
            return

        self._seen_keys.add(key)
        self.findings.append(
            OptimizationFinding(
                rule_id=rule_id,
                title=title,
                description=description,
                suggestion=suggestion,
                line_number=lineno,
                column_offset=col_offset,
                function_scope=self.current_function_scope,
            )
        )

    def _is_mutable_default_expr(self, expr: ast.expr) -> Optional[str]:
        """
        Check if an expression is a mutable list, dict, or set literal / constructor.
        Returns type description if mutable, or None.
        """
        if isinstance(expr, ast.List):
            return "list"
        if isinstance(expr, ast.Dict):
            return "dict"
        if isinstance(expr, ast.Set):
            return "set"
        # Check for set() or dict() or list() call with 0 or 1 args
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            if expr.func.id in {"list", "dict", "set"}:
                return expr.func.id
        return None

    def _check_mutable_defaults(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """
        Check for OPT001: Mutable list, dict, or set defaults in positional or kw-only arguments.
        """
        args = node.args
        all_defaults: List[Optional[ast.expr]] = []

        # args.defaults corresponds to the last N positional arguments
        all_defaults.extend(args.defaults)
        # args.kw_defaults contains defaults for keyword-only arguments (or None)
        all_defaults.extend([d for d in args.kw_defaults if d is not None])

        for default_expr in all_defaults:
            mutable_type = self._is_mutable_default_expr(default_expr)
            if mutable_type:
                self._add_finding(
                    rule_id="OPT001",
                    title="Mutable Default Argument",
                    description=(
                        f"Function '{node.name}' defines a mutable {mutable_type} as a default "
                        "argument value. Default arguments are instantiated once when the function "
                        "is defined and shared across all invocations."
                    ),
                    suggestion=(
                        "Use 'None' as the default argument value and initialize the mutable "
                        "container inside the function body (e.g., 'if arg is None: arg = []')."
                    ),
                    node=default_expr,
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit synchronous function definition."""
        self._scope_stack.append(node.name)
        self._check_mutable_defaults(node)
        saved_loop_depth = self._loop_depth
        self._loop_depth = 0
        try:
            self.generic_visit(node)
        finally:
            self._loop_depth = saved_loop_depth
            self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit asynchronous function definition."""
        self._scope_stack.append(node.name)
        self._check_mutable_defaults(node)
        saved_loop_depth = self._loop_depth
        self._loop_depth = 0
        try:
            self.generic_visit(node)
        finally:
            self._loop_depth = saved_loop_depth
            self._scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition, tracking class scope."""
        self._scope_stack.append(f"<class:{node.name}>")
        saved_loop_depth = self._loop_depth
        self._loop_depth = 0
        try:
            self.generic_visit(node)
        finally:
            self._loop_depth = saved_loop_depth
            self._scope_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        """Visit for-loop, tracking loop depth."""
        self._loop_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Visit async for-loop, tracking loop depth."""
        self._loop_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        """Visit while-loop, tracking loop depth."""
        self._loop_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._loop_depth -= 1

    @staticmethod
    def _elements_are_hashable_constants(elements: list) -> bool:
        """
        Return True if every element in the literal sequence is a hashable constant
        (ast.Constant, or a tuple of constants) that can safely be placed in a set.
        Mutable literals (ast.List, ast.Dict, ast.Set) cannot be hashed.
        """
        for el in elements:
            if isinstance(el, (ast.List, ast.Dict, ast.Set)):
                return False
        return True

    def visit_Compare(self, node: ast.Compare) -> None:
        """
        Check for OPT002: Inefficient membership check against literal lists or tuples.

        The suggestion is tailored based on whether the literal elements are likely hashable:
        - Hashable constants → recommend a set literal or module-level frozenset.
        - Potentially unhashable elements → note that a set cannot be used directly
          and suggest lifting the collection to a module-level constant if the check
          is repeated, or leaving it as-is for a one-off lookup.
        """
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if isinstance(comparator, (ast.List, ast.Tuple)):
                    container_type = "list" if isinstance(comparator, ast.List) else "tuple"
                    elements = comparator.elts

                    if self._elements_are_hashable_constants(elements):
                        suggestion = (
                            "If this check is performed repeatedly or on large sequences, replace "
                            "the literal with a set literal (e.g., '{...}') for O(1) average-time "
                            "lookup. For a module-level constant, prefer 'frozenset({...})' to "
                            "avoid rebuilding the set on every function call. For a short, one-off "
                            "check the performance difference is negligible."
                        )
                    else:
                        suggestion = (
                            "The literal sequence contains elements that may not be hashable "
                            "(e.g., lists or dicts), so it cannot be directly converted to a set. "
                            "If this check is performed frequently, consider restructuring the data "
                            "so that the lookup keys are hashable, then use a set or frozenset. "
                            "For a short, one-off check the performance difference is negligible."
                        )

                    self._add_finding(
                        rule_id="OPT002",
                        title="Inefficient Literal Membership Check",
                        description=(
                            f"Membership check ('in' / 'not in') performed against a literal {container_type}. "
                            f"Evaluating membership in a {container_type} literal requires O(n) linear search."
                        ),
                        suggestion=suggestion,
                        node=comparator,
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Check for OPT003: List comprehension passed as single positional argument
        to reduction functions (sum, any, all, min, max).

        When keyword arguments are present (e.g., default=, key=, start=) the
        generator-expression replacement must be parenthesized, because bare
        generator expressions adjacent to keyword arguments are a SyntaxError in Python.
        """
        if isinstance(node.func, ast.Name) and node.func.id in REDUCTION_FUNCTIONS:
            # Detect: exactly one positional arg and it's a list comprehension
            if len(node.args) == 1 and isinstance(node.args[0], ast.ListComp):
                func_name = node.func.id
                has_kwargs = bool(node.keywords)

                if has_kwargs:
                    # Keyword args require the generator expression to be parenthesized:
                    # e.g. min((x for x in data), default=0)
                    kwargs_hint = ", ".join(
                        f"{kw.arg}=..." for kw in node.keywords if kw.arg is not None
                    )
                    suggestion = (
                        f"Pass a generator expression instead of a list comprehension into '{func_name}()' "
                        f"to avoid allocating an intermediate list. Because keyword arguments are present, "
                        f"wrap the generator in parentheses: "
                        f"'{func_name}((x for x in ...), {kwargs_hint})'."
                    )
                else:
                    suggestion = (
                        f"Pass a generator expression directly into '{func_name}()' (omit the outer brackets, "
                        f"e.g., '{func_name}(x for x in ...)') to stream items lazily with O(1) memory overhead."
                    )

                self._add_finding(
                    rule_id="OPT003",
                    title="Redundant List Comprehension in Reduction",
                    description=(
                        f"Built-in '{func_name}()' receives a list comprehension argument. "
                        "An entire intermediate list is allocated in memory before reduction begins."
                    ),
                    suggestion=suggestion,
                    node=node.args[0],
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        Check for OPT004: In-loop augmented addition (potential string concatenation accumulator).
        Reported as potential rather than confirmed string accumulation.
        """
        if self._loop_depth > 0 and isinstance(node.op, ast.Add):
            self._add_finding(
                rule_id="OPT004",
                title="Potential In-Loop String Accumulation",
                description=(
                    "Augmented addition ('+=') detected inside a loop body. If this accumulator is "
                    "building a string, repeated concatenation creates new string objects at each "
                    "iteration, resulting in potential quadratic O(N^2) memory and runtime overhead."
                ),
                suggestion=(
                    "If concatenating strings, collect segments into a list using '.append()' and "
                    "assemble them with ''.join(...) outside the loop."
                ),
                node=node,
            )
        self.generic_visit(node)


def analyze_optimizations(source_code: str) -> OptimizationResult:
    """
    Statically inspect Python source code for structural optimization opportunities.

    Parameters
    ----------
    source_code : str
        Python source code to be parsed and analyzed.

    Returns
    -------
    OptimizationResult
        Structured report containing findings, execution status, and disclaimer.
    """
    if not isinstance(source_code, str):
        return OptimizationResult(
            success=False,
            error_message="Invalid input: source code must be a string.",
            total_findings=0,
        )

    # Empty or whitespace/comment-only strings should succeed with 0 findings
    if not source_code.strip():
        return OptimizationResult(
            success=True,
            findings=[],
            total_findings=0,
        )

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        line_desc = f" at line {e.lineno}" if e.lineno is not None else ""
        col_desc = f", column {e.offset}" if e.offset is not None else ""
        msg = f"Syntax error parsing Python source{line_desc}{col_desc}: {e.msg}"
        return OptimizationResult(
            success=False,
            error_message=msg,
            total_findings=0,
        )
    except Exception as e:
        return OptimizationResult(
            success=False,
            error_message=f"Failed to parse Python source: {str(e)}",
            total_findings=0,
        )

    visitor = OptimizationVisitor()
    visitor.visit(tree)

    return OptimizationResult(
        success=True,
        findings=visitor.findings,
        total_findings=len(visitor.findings),
    )
