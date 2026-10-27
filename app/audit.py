"""
app/audit.py

Evidence-Grounded Documentation Audit Engine for CODEscribe.

Performs deterministic, static AST fact extraction and heuristic lexical text
analysis to audit generated Markdown documentation against Python source code.

Core Principles & Safeguards:
1. Strict static analysis: Never executes user-submitted code (no eval, exec, or import).
2. Honest framing: All comparisons are labeled as heuristic lexical checks.
   Does not claim semantic correctness, runtime verification, or guaranteed truth.
3. Scoped AST traversal: Respects lexical scopes, isolating target function facts
   from nested function or class definitions.
4. Robust resilience: Gracefully handles syntax errors, missing sections, empty code,
   unannotated functions, variadic arguments, generators, and async functions.
5. Standard library only: Zero third-party dependencies; fully deterministic.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("CODEscribe.Audit")

AUDIT_DISCLAIMER = (
    "This audit is a heuristic static analysis comparing Python AST structural "
    "facts with Markdown text. It does not execute source code, evaluate dynamic "
    "control flow, verify runtime behavior, or guarantee semantic documentation accuracy. "
    "Presence or absence of an AST node does not reflect dynamic execution paths, "
    "monkey-patching, or exceptions raised by external dependencies."
)


class AuditStatus(str, Enum):
    """Status classification for individual audit findings."""

    MENTIONED = "MENTIONED"
    NOT_MENTIONED = "NOT_MENTIONED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    POSSIBLE_DISCREPANCY = "POSSIBLE_DISCREPANCY"


@dataclass
class ParameterFact:
    """AST fact for a function parameter."""

    name: str
    kind: str  # 'posonly', 'standard', 'vararg', 'kwonly', 'kwarg'
    annotation: Optional[str] = None
    has_default: bool = False
    default_repr: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReturnFact:
    """AST fact for an explicit return statement."""

    has_value: bool
    value_repr: Optional[str] = None
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RaiseFact:
    """AST fact for an explicitly raised exception."""

    exception_type: str
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceEvidence:
    """Structured ground-truth facts extracted via standard library AST."""

    function_name: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    parameters: List[ParameterFact] = field(default_factory=list)
    return_annotation: Optional[str] = None
    return_statements: List[ReturnFact] = field(default_factory=list)
    raised_exceptions: List[RaiseFact] = field(default_factory=list)
    nested_functions: List[str] = field(default_factory=list)
    has_docstring: bool = False
    syntax_valid: bool = True
    syntax_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_annotation": self.return_annotation,
            "return_statements": [r.to_dict() for r in self.return_statements],
            "raised_exceptions": [e.to_dict() for e in self.raised_exceptions],
            "nested_functions": list(self.nested_functions),
            "has_docstring": self.has_docstring,
            "syntax_valid": self.syntax_valid,
            "syntax_error": self.syntax_error,
        }


@dataclass
class DocEvidence:
    """Structured documentation evidence extracted from Markdown."""

    has_parameters_section: bool = False
    has_returns_section: bool = False
    has_raises_section: bool = False
    has_description_section: bool = False
    sections_present: List[str] = field(default_factory=list)
    signature_block_found: bool = False
    signature_text: Optional[str] = None
    parameters_section_text: Optional[str] = None
    returns_section_text: Optional[str] = None
    raises_section_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditFinding:
    """Individual evidence-grounded audit finding."""

    category: str  # 'parameter', 'return', 'exception', 'generator', 'signature', 'target', 'structure'
    subject: str
    status: AuditStatus
    source_evidence: str
    doc_evidence: str
    message: str
    is_heuristic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "subject": self.subject,
            "status": self.status.value if isinstance(self.status, AuditStatus) else str(self.status),
            "source_evidence": self.source_evidence,
            "doc_evidence": self.doc_evidence,
            "message": self.message,
            "is_heuristic": self.is_heuristic,
        }


@dataclass
class AuditSummary:
    """Aggregated numerical summary and high-level audit status."""

    total_parameters: int = 0
    parameters_mentioned: int = 0
    parameters_coverage_ratio: Optional[float] = None
    return_status: AuditStatus = AuditStatus.NOT_APPLICABLE
    exceptions_status: AuditStatus = AuditStatus.NOT_APPLICABLE
    overall_status: str = "CONSISTENT"  # 'CONSISTENT', 'INCOMPLETE', 'DISCREPANCY_DETECTED', 'SYNTAX_ERROR', 'FUNCTION_NOT_FOUND', 'NO_FUNCTION_FOUND'
    disclaimer: str = AUDIT_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_parameters": self.total_parameters,
            "parameters_mentioned": self.parameters_mentioned,
            "parameters_coverage_ratio": self.parameters_coverage_ratio,
            "return_status": self.return_status.value if isinstance(self.return_status, AuditStatus) else str(self.return_status),
            "exceptions_status": self.exceptions_status.value if isinstance(self.exceptions_status, AuditStatus) else str(self.exceptions_status),
            "overall_status": self.overall_status,
            "disclaimer": self.disclaimer,
        }


@dataclass
class AuditReport:
    """Complete structured evidence-grounded documentation audit report."""

    success: bool
    function_name: Optional[str]
    summary: AuditSummary
    findings: List[AuditFinding] = field(default_factory=list)
    source_evidence: SourceEvidence = field(default_factory=SourceEvidence)
    doc_evidence: DocEvidence = field(default_factory=DocEvidence)
    disclaimer: str = AUDIT_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "function_name": self.function_name,
            "summary": self.summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "source_evidence": self.source_evidence.to_dict(),
            "doc_evidence": self.doc_evidence.to_dict(),
            "disclaimer": self.disclaimer,
        }


# ============================================================================
# AST Fact Extraction (Pure Standard Library)
# ============================================================================

def _safe_unparse(node: Optional[ast.AST]) -> Optional[str]:
    """Safely unparse an AST expression node to a string."""
    if node is None:
        return None
    try:
        return ast.unparse(node).strip()
    except Exception:
        return None


def _extract_exception_type(exc_node: Optional[ast.AST]) -> str:
    """
    Extract the human-readable exception type from an ast.Raise expression node.
    Supports identifiers, module attributes, constructor calls, and re-raises.
    """
    if exc_node is None:
        return "<re-raise>"
    if isinstance(exc_node, ast.Name):
        return exc_node.id
    if isinstance(exc_node, ast.Attribute):
        return exc_node.attr
    if isinstance(exc_node, ast.Call):
        if isinstance(exc_node.func, ast.Name):
            return exc_node.func.id
        if isinstance(exc_node.func, ast.Attribute):
            return exc_node.func.attr
        unparsed = _safe_unparse(exc_node.func)
        return unparsed if unparsed else "Exception"
    unparsed = _safe_unparse(exc_node)
    return unparsed if unparsed else "Exception"


class _ScopedFunctionScanner:
    """
    Walks statement trees within a single target function's lexical scope.
    Stops traversal at nested function and class boundaries so that statements
    belonging to inner functions or classes are not falsely attributed to the outer.
    """

    def __init__(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.func_node = func_node
        self.return_nodes: List[ast.Return] = []
        self.yield_nodes: List[ast.Yield | ast.YieldFrom] = []
        self.raise_nodes: List[ast.Raise] = []
        self.nested_functions: List[str] = []

    def scan(self) -> None:
        for stmt in self.func_node.body:
            self._visit(stmt)

    def _visit(self, node: ast.AST) -> None:
        # Stop descent at nested function or class definitions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.nested_functions.append(node.name)
            return
        if isinstance(node, ast.ClassDef):
            return

        if isinstance(node, ast.Return):
            self.return_nodes.append(node)
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            self.yield_nodes.append(node)
        elif isinstance(node, ast.Raise):
            self.raise_nodes.append(node)

        for child in ast.iter_child_nodes(node):
            self._visit(child)


def _extract_parameters(args: ast.arguments) -> List[ParameterFact]:
    """
    Extract structured parameter facts matching positional-only, standard,
    keyword-only, variadic (*args), and keyword variadic (**kwargs) parameters
    with their annotations and defaults.
    """
    parameters: List[ParameterFact] = []

    # Positional-only and standard positional parameters share the `defaults` list
    pos_args: List[Tuple[ast.arg, str]] = []
    if hasattr(args, "posonlyargs"):
        for arg in args.posonlyargs:
            pos_args.append((arg, "posonly"))
    for arg in args.args:
        pos_args.append((arg, "standard"))

    num_pos = len(pos_args)
    num_defaults = len(args.defaults)
    default_offset = num_pos - num_defaults

    for idx, (arg, kind) in enumerate(pos_args):
        annotation = _safe_unparse(arg.annotation)
        has_default = idx >= default_offset
        default_repr = None
        if has_default:
            default_expr = args.defaults[idx - default_offset]
            default_repr = _safe_unparse(default_expr)
        parameters.append(
            ParameterFact(
                name=arg.arg,
                kind=kind,
                annotation=annotation,
                has_default=has_default,
                default_repr=default_repr,
            )
        )

    # Variadic *args
    if args.vararg:
        parameters.append(
            ParameterFact(
                name=f"*{args.vararg.arg}",
                kind="vararg",
                annotation=_safe_unparse(args.vararg.annotation),
                has_default=False,
                default_repr=None,
            )
        )

    # Keyword-only arguments
    for idx, arg in enumerate(args.kwonlyargs):
        annotation = _safe_unparse(arg.annotation)
        default_expr = args.kw_defaults[idx] if idx < len(args.kw_defaults) else None
        has_default = default_expr is not None
        default_repr = _safe_unparse(default_expr) if has_default else None
        parameters.append(
            ParameterFact(
                name=arg.arg,
                kind="kwonly",
                annotation=annotation,
                has_default=has_default,
                default_repr=default_repr,
            )
        )

    # Variadic **kwargs
    if args.kwarg:
        parameters.append(
            ParameterFact(
                name=f"**{args.kwarg.arg}",
                kind="kwarg",
                annotation=_safe_unparse(args.kwarg.annotation),
                has_default=False,
                default_repr=None,
            )
        )

    return parameters


def extract_ast_facts(
    source_code: str, target_function: Optional[str] = None
) -> SourceEvidence:
    """
    Extract factual syntactic metadata from Python source code safely via AST.

    Args:
        source_code: Raw Python source code string.
        target_function: Optional function name to isolate. If None, targets the first
                         FunctionDef or AsyncFunctionDef encountered.

    Returns:
        SourceEvidence dataclass containing extracted facts or syntax error details.
    """
    if not source_code or not source_code.strip():
        return SourceEvidence(
            function_name=None,
            syntax_valid=False,
            syntax_error="Submitted source code is empty or whitespace-only.",
        )

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return SourceEvidence(
            function_name=None,
            syntax_valid=False,
            syntax_error=f"Syntax error at line {exc.lineno}: {exc.msg}",
        )
    except Exception as exc:
        return SourceEvidence(
            function_name=None,
            syntax_valid=False,
            syntax_error=f"Failed to parse source code: {str(exc)}",
        )

    # Discover candidate function nodes
    functions: List[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)

    if not functions:
        return SourceEvidence(
            function_name=None,
            syntax_valid=True,
            syntax_error=None,
        )

    # Select target function
    target_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef] = None
    if target_function:
        for fn in functions:
            if fn.name == target_function:
                target_node = fn
                break
        if target_node is None:
            return SourceEvidence(
                function_name=target_function,
                syntax_valid=False,
                syntax_error=f"Target function '{target_function}' not found in source code.",
            )
    else:
        target_node = functions[0]

    # Scoped body scan (prevents nested function statement bleed)
    scanner = _ScopedFunctionScanner(target_node)
    scanner.scan()

    # Extract parameters
    params = _extract_parameters(target_node.args)

    # Extract return annotation
    return_annotation = _safe_unparse(target_node.returns)

    # Extract return facts
    return_statements: List[ReturnFact] = []
    for ret in scanner.return_nodes:
        has_val = ret.value is not None
        val_repr = _safe_unparse(ret.value) if has_val else None
        return_statements.append(
            ReturnFact(
                has_value=has_val,
                value_repr=val_repr,
                line_number=getattr(ret, "lineno", None),
            )
        )

    # Extract explicit raise facts
    raised_exceptions: List[RaiseFact] = []
    for r in scanner.raise_nodes:
        exc_type = _extract_exception_type(r.exc)
        raised_exceptions.append(
            RaiseFact(
                exception_type=exc_type,
                line_number=getattr(r, "lineno", None),
            )
        )

    is_gen = len(scanner.yield_nodes) > 0
    is_async_fn = isinstance(target_node, ast.AsyncFunctionDef)
    has_doc = ast.get_docstring(target_node) is not None

    return SourceEvidence(
        function_name=target_node.name,
        is_async=is_async_fn,
        is_generator=is_gen,
        parameters=params,
        return_annotation=return_annotation,
        return_statements=return_statements,
        raised_exceptions=raised_exceptions,
        nested_functions=scanner.nested_functions,
        has_docstring=has_doc,
        syntax_valid=True,
        syntax_error=None,
    )


# ============================================================================
# Markdown Documentation Analysis (Heuristic Lexical Parser)
# ============================================================================

def _normalize_heading(title: str) -> str:
    """Normalize a markdown section heading title into a canonical key."""
    clean = re.sub(r"[`*_\'\"]", "", title).strip().lower()
    if clean in ("parameters", "params", "args", "arguments", "inputs"):
        return "parameters"
    if clean in ("returns", "return", "return value", "returns value", "yields", "output", "outputs"):
        return "returns"
    if clean in ("raises", "exceptions", "errors", "raise"):
        return "raises"
    if clean in ("description", "overview", "summary", "details"):
        return "description"
    if "control flow" in clean or "flowchart" in clean or "mermaid" in clean:
        return "control_flow"
    return clean


def extract_markdown_evidence(markdown_text: str) -> Tuple[Dict[str, str], DocEvidence]:
    """
    Parse a Markdown documentation string into structured sections and
    evidence metrics without external Markdown rendering dependencies.

    Correctly ignores Markdown heading patterns within fenced code blocks.

    Returns:
        Tuple of (sections_dict, DocEvidence).
    """
    if not markdown_text or not markdown_text.strip():
        return {}, DocEvidence()

    sections: Dict[str, str] = {}
    current_key = "_preamble"
    current_lines: List[str] = []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    signature_block_found = False
    signature_text: Optional[str] = None

    # Detect code block signature if present: supports standard def, async def, and decorated functions
    code_block_match = re.search(
        r"```python\s+((?:@[^\n]+\n\s*)*(?:async\s+)?def\s+[^`]+?)```",
        markdown_text,
        re.DOTALL,
    )
    if code_block_match:
        signature_block_found = True
        signature_text = code_block_match.group(1).strip()

    in_code_block = False
    fence_delimiter: Optional[str] = None

    for line in markdown_text.splitlines():
        stripped = line.strip()

        # Track fenced code block boundaries (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            if not in_code_block:
                in_code_block = True
                fence_delimiter = fence
            elif fence == fence_delimiter:
                in_code_block = False
                fence_delimiter = None
            current_lines.append(line)
            continue

        # If inside a code block, do NOT parse # lines as Markdown headings
        if not in_code_block and heading_pattern.match(stripped):
            match = heading_pattern.match(stripped)
            assert match is not None
            # Save accumulated content of previous section
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            heading_title = match.group(2).strip()
            current_key = _normalize_heading(heading_title)
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    has_params = "parameters" in sections and bool(sections["parameters"])
    has_returns = "returns" in sections and bool(sections["returns"])
    has_raises = "raises" in sections and bool(sections["raises"])
    has_desc = "description" in sections and bool(sections["description"])

    doc_evidence = DocEvidence(
        has_parameters_section=has_params,
        has_returns_section=has_returns,
        has_raises_section=has_raises,
        has_description_section=has_desc,
        sections_present=list(sections.keys()),
        signature_block_found=signature_block_found,
        signature_text=signature_text,
        parameters_section_text=sections.get("parameters"),
        returns_section_text=sections.get("returns"),
        raises_section_text=sections.get("raises"),
    )

    return sections, doc_evidence


# ============================================================================
# Audit Cross-Verification Engine (Evidence-Grounded)
# ============================================================================

def _is_parameter_documented(param_name: str, params_text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a parameter is defined/introduced as a documented entry in the Parameters section.
    Requires the parameter name to appear in an entry-defining position (e.g. bullet item,
    code span term, bold term, table row cell, or name: type definition), preventing false-positive
    matches from ordinary prose mentioning words like 'a', 'to', 'count', 'data'.
    """
    base_name = param_name.lstrip("*")
    if not base_name or not params_text:
        return False, None

    escaped = re.escape(base_name)
    # Entry-defining patterns avoiding f-string curly-brace evaluation:
    patterns = [
        re.compile(r"`\*{0,2}" + escaped + r"`"),
        re.compile(r"\*\*\*{0,2}" + escaped + r"\*\*"),
        re.compile(r"^\s*[-*+]\s+(?:\*{0,2})" + escaped + r"(?:\s*\(.*?\))?\s*:", re.MULTILINE),
        re.compile(r"^\s*(?:\*{0,2})" + escaped + r"\s*:\s+", re.MULTILINE),
        re.compile(r"\|\s*(?:`\*{0,2}" + escaped + r"`|\*{0,2}" + escaped + r")\s*\|"),
    ]

    for line in params_text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        for pat in patterns:
            if pat.search(line):
                return True, trimmed

    return False, None


def audit_documentation(
    source_code: str,
    markdown_doc: str,
    target_function: Optional[str] = None,
) -> AuditReport:
    """
    Execute an Evidence-Grounded Documentation Audit comparing AST source facts
    with generated Markdown documentation.

    Args:
        source_code: Raw Python source code string.
        markdown_doc: Generated Markdown documentation string.
        target_function: Optional function name to audit.

    Returns:
        Structured AuditReport with findings, source evidence, and documentation evidence.
    """
    # 1. Extract AST ground truth facts
    source_facts = extract_ast_facts(source_code, target_function=target_function)

    # Handle syntax error or missing target function
    if not source_facts.syntax_valid:
        is_target_missing = "not found in source code" in (source_facts.syntax_error or "")
        overall_stat = "FUNCTION_NOT_FOUND" if is_target_missing else "SYNTAX_ERROR"
        category = "target" if is_target_missing else "syntax"
        subject = source_facts.function_name or "source_code"
        summary = AuditSummary(
            total_parameters=0,
            parameters_mentioned=0,
            parameters_coverage_ratio=None,
            return_status=AuditStatus.NOT_APPLICABLE,
            exceptions_status=AuditStatus.NOT_APPLICABLE,
            overall_status=overall_stat,
        )
        finding = AuditFinding(
            category=category,
            subject=subject,
            status=AuditStatus.POSSIBLE_DISCREPANCY,
            source_evidence=source_facts.syntax_error or "Source analysis failed.",
            doc_evidence="N/A",
            message=(
                f"Target function '{source_facts.function_name}' was not found in source code."
                if is_target_missing
                else f"Cannot audit documentation due to Python syntax error: {source_facts.syntax_error}"
            ),
            is_heuristic=False,
        )
        return AuditReport(
            success=False,
            function_name=source_facts.function_name,
            summary=summary,
            findings=[finding],
            source_evidence=source_facts,
            doc_evidence=DocEvidence(),
        )

    # Handle source code with no function definitions
    if source_facts.function_name is None:
        summary = AuditSummary(
            total_parameters=0,
            parameters_mentioned=0,
            parameters_coverage_ratio=None,
            return_status=AuditStatus.NOT_APPLICABLE,
            exceptions_status=AuditStatus.NOT_APPLICABLE,
            overall_status="NO_FUNCTION_FOUND",
        )
        finding = AuditFinding(
            category="structure",
            subject="source_code",
            status=AuditStatus.NOT_APPLICABLE,
            source_evidence="Source code contains no function definitions (module-level script only).",
            doc_evidence="N/A",
            message="No function definition found in source code; documentation audit requires a function to evaluate.",
            is_heuristic=False,
        )
        return AuditReport(
            success=False,
            function_name=None,
            summary=summary,
            findings=[finding],
            source_evidence=source_facts,
            doc_evidence=DocEvidence(),
        )

    # 2. Extract documentation evidence
    sections, doc_facts = extract_markdown_evidence(markdown_doc)

    findings: List[AuditFinding] = []
    fn_name = source_facts.function_name

    # ------------------------------------------------------------------------
    # Check 1: Signature & Function Name Presence (No arbitrary line cap)
    # ------------------------------------------------------------------------
    sig_found = False
    sig_evidence = "Function name not found in title or signature code block."
    if markdown_doc:
        name_pattern = re.compile(rf"(#\s+`?{re.escape(fn_name)}`?|(?:async\s+)?def\s+{re.escape(fn_name)}\b)")
        for line in markdown_doc.splitlines():
            if name_pattern.search(line):
                sig_found = True
                sig_evidence = f"Function name '{fn_name}' identified in header/signature: '{line.strip()}'"
                break

    findings.append(
        AuditFinding(
            category="signature",
            subject=fn_name,
            status=AuditStatus.MENTIONED if sig_found else AuditStatus.POSSIBLE_DISCREPANCY,
            source_evidence=f"AST function definition for '{fn_name}'",
            doc_evidence=sig_evidence,
            message=(
                f"Function '{fn_name}' is clearly identified in the documentation header/signature."
                if sig_found
                else f"Documentation does not clearly introduce function '{fn_name}' in the header."
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Check 2: Parameters Coverage (Strictly within Parameters section)
    # ------------------------------------------------------------------------
    params = source_facts.parameters
    total_params = len(params)
    mentioned_params = 0
    params_section_text = sections.get("parameters", "")

    if total_params > 0:
        for p in params:
            clean_name = p.name.lstrip("*")
            if not doc_facts.has_parameters_section:
                findings.append(
                    AuditFinding(
                        category="parameter",
                        subject=p.name,
                        status=AuditStatus.NOT_MENTIONED,
                        source_evidence=f"AST parameter '{p.name}' (kind: {p.kind}, type: {p.annotation or 'unannotated'})",
                        doc_evidence="Documentation lacks a dedicated 'Parameters' / 'Args' section.",
                        message=(
                            f"Parameter '{p.name}' is present in source AST but is not documented "
                            f"(no 'Parameters' section found; mentions in code blocks do not count as parameter documentation)."
                        ),
                    )
                )
            else:
                found, matched_line = _is_parameter_documented(p.name, params_section_text)
                if found:
                    mentioned_params += 1
                    findings.append(
                        AuditFinding(
                            category="parameter",
                            subject=p.name,
                            status=AuditStatus.MENTIONED,
                            source_evidence=f"AST parameter '{p.name}' (kind: {p.kind}, type: {p.annotation or 'unannotated'})",
                            doc_evidence=f"Mentioned in Parameters section: '{matched_line}'",
                            message=f"Parameter '{p.name}' is mentioned within the Parameters section.",
                        )
                    )
                else:
                    findings.append(
                        AuditFinding(
                            category="parameter",
                            subject=p.name,
                            status=AuditStatus.NOT_MENTIONED,
                            source_evidence=f"AST parameter '{p.name}' (kind: {p.kind}, type: {p.annotation or 'unannotated'})",
                            doc_evidence=f"Parameters section exists ({len(params_section_text)} chars) but does not define '{clean_name}'.",
                            message=f"Parameter '{p.name}' is missing from the Parameters section.",
                        )
                    )
        coverage_ratio: Optional[float] = round(mentioned_params / total_params, 2)
    else:
        # Function takes no parameters
        coverage_ratio = None
        if doc_facts.has_parameters_section and len(params_section_text.strip()) > 10:
            findings.append(
                AuditFinding(
                    category="parameter",
                    subject="<none>",
                    status=AuditStatus.POSSIBLE_DISCREPANCY,
                    source_evidence="AST function takes zero parameters.",
                    doc_evidence=f"Parameters section is present: '{params_section_text[:80]}...'",
                    message="Documentation contains a non-empty Parameters section, but the source function takes no arguments.",
                )
            )
        else:
            findings.append(
                AuditFinding(
                    category="parameter",
                    subject="<none>",
                    status=AuditStatus.NOT_APPLICABLE,
                    source_evidence="AST function takes zero parameters.",
                    doc_evidence="Parameters section is appropriately omitted or empty.",
                    message="Function takes no parameters; no parameter documentation required.",
                )
            )

    # ------------------------------------------------------------------------
    # Check 3: Return Value & Generator Auditing
    # ------------------------------------------------------------------------
    returns_section_text = sections.get("returns", "")
    has_meaningful_return_annotation = (
        source_facts.return_annotation is not None
        and source_facts.return_annotation.strip() not in ("None", "NoneType")
    )
    non_none_returns = [
        r for r in source_facts.return_statements if r.has_value and r.value_repr != "None"
    ]
    source_has_return_value = bool(non_none_returns) or has_meaningful_return_annotation

    return_audit_status: AuditStatus

    if source_facts.is_generator:
        # Generator function (yield / yield from)
        gen_tokens = ("yield", "generator", "iterator", "iterable", "stream")
        all_doc_lower = markdown_doc.lower()
        gen_mentioned = any(tok in all_doc_lower for tok in gen_tokens)

        if gen_mentioned:
            return_audit_status = AuditStatus.MENTIONED
            findings.append(
                AuditFinding(
                    category="generator",
                    subject="yield",
                    status=AuditStatus.MENTIONED,
                    source_evidence="Function body contains AST Yield/YieldFrom statements.",
                    doc_evidence="Documentation mentions generator/yield/iterator semantics.",
                    message="Function operates as a generator; documentation acknowledges generator/yield semantics.",
                )
            )
        else:
            return_audit_status = AuditStatus.POSSIBLE_DISCREPANCY
            findings.append(
                AuditFinding(
                    category="generator",
                    subject="yield",
                    status=AuditStatus.POSSIBLE_DISCREPANCY,
                    source_evidence="Function body contains AST Yield/YieldFrom statements.",
                    doc_evidence="Documentation does not mention 'yield', 'generator', or 'iterator'.",
                    message="Function contains yield statements (is a generator), but documentation does not clearly mention generator/yield semantics.",
                )
            )

    elif source_has_return_value:
        ret_label = source_facts.return_annotation or "value"
        if doc_facts.has_returns_section:
            return_audit_status = AuditStatus.MENTIONED
            findings.append(
                AuditFinding(
                    category="return",
                    subject=ret_label,
                    status=AuditStatus.MENTIONED,
                    source_evidence=(
                        f"AST return annotation: '{source_facts.return_annotation}'; "
                        f"{len(non_none_returns)} return statement(s) with values."
                    ),
                    doc_evidence=f"Returns section present: '{returns_section_text[:100]}'",
                    message="Return value is documented in a dedicated Returns section.",
                )
            )
        else:
            return_audit_status = AuditStatus.NOT_MENTIONED
            findings.append(
                AuditFinding(
                    category="return",
                    subject=ret_label,
                    status=AuditStatus.NOT_MENTIONED,
                    source_evidence=(
                        f"AST return annotation: '{source_facts.return_annotation}'; "
                        f"{len(non_none_returns)} return statement(s) with values."
                    ),
                    doc_evidence="Documentation lacks a dedicated 'Returns' section.",
                    message="Function returns a computed value, but no Returns section is present in documentation.",
                )
            )
    else:
        # Void or returns-None function
        ret_annot_note = (
            f"return annotation is '{source_facts.return_annotation}'"
            if source_facts.return_annotation
            else "no return annotation"
        )
        if doc_facts.has_returns_section:
            returns_clean = returns_section_text.strip().lower()
            indicates_none = (
                "none" in returns_clean
                or "no return" in returns_clean
                or "void" in returns_clean
                or returns_clean in ("n/a", "none.", "does not return a value.")
            )
            if indicates_none:
                return_audit_status = AuditStatus.NOT_APPLICABLE
                findings.append(
                    AuditFinding(
                        category="return",
                        subject="void",
                        status=AuditStatus.NOT_APPLICABLE,
                        source_evidence=f"Function produces no return value ({ret_annot_note}) and no non-None return statement.",
                        doc_evidence=f"Returns section explicitly documents None: '{returns_section_text[:100]}'",
                        message="Function produces no return value; Returns section appropriately documents None / no return value.",
                    )
                )
            else:
                return_audit_status = AuditStatus.POSSIBLE_DISCREPANCY
                findings.append(
                    AuditFinding(
                        category="return",
                        subject="void",
                        status=AuditStatus.POSSIBLE_DISCREPANCY,
                        source_evidence=f"Function produces no return value ({ret_annot_note}) and no non-None return statement.",
                        doc_evidence=f"Returns section present claiming a return value: '{returns_section_text[:100]}'",
                        message="Documentation contains a Returns section claiming a return value, but the source function has no non-None return statement.",
                    )
                )
        else:
            return_audit_status = AuditStatus.NOT_APPLICABLE
            findings.append(
                AuditFinding(
                    category="return",
                    subject="void",
                    status=AuditStatus.NOT_APPLICABLE,
                    source_evidence=f"Function produces no return value ({ret_annot_note}) and no non-None return statement.",
                    doc_evidence="Returns section is appropriately omitted.",
                    message="Function produces no return value; Returns section appropriately omitted.",
                )
            )

    # ------------------------------------------------------------------------
    # Check 4: Explicit Exception Auditing
    # ------------------------------------------------------------------------
    raised_types: Set[str] = {
        r.exception_type for r in source_facts.raised_exceptions if r.exception_type != "<re-raise>"
    }
    has_reraise = any(r.exception_type == "<re-raise>" for r in source_facts.raised_exceptions)
    all_doc_text = markdown_doc.lower()
    raises_section_text = sections.get("raises", "")

    exceptions_audit_status: AuditStatus

    if raised_types or has_reraise:
        missing_exceptions: List[str] = []
        found_exceptions: List[str] = []

        for exc in raised_types:
            in_raises_section = bool(raises_section_text and exc.lower() in raises_section_text.lower())
            in_full_doc = exc.lower() in all_doc_text

            if in_raises_section or in_full_doc:
                found_exceptions.append(exc)
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject=exc,
                        status=AuditStatus.MENTIONED,
                        source_evidence=f"AST contains 'raise {exc}(...)' statement.",
                        doc_evidence=f"Exception '{exc}' is mentioned in documentation.",
                        message=f"Explicitly raised exception '{exc}' is acknowledged in documentation.",
                    )
                )
            else:
                missing_exceptions.append(exc)
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject=exc,
                        status=AuditStatus.NOT_MENTIONED,
                        source_evidence=f"AST contains 'raise {exc}(...)' statement.",
                        doc_evidence=f"Exception '{exc}' is not mentioned in documentation.",
                        message=f"Explicitly raised exception '{exc}' is not mentioned in documentation.",
                    )
                )

        if has_reraise:
            # Check if documentation mentions re-raising or exception propagation
            reraise_tokens = ("re-raise", "reraise", "propagate", "propagates", "raises", "raise", "exception")
            reraise_acknowledged = bool(
                doc_facts.has_raises_section
                or any(t in all_doc_text for t in reraise_tokens)
            )
            if reraise_acknowledged:
                found_exceptions.append("<re-raise>")
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject="<re-raise>",
                        status=AuditStatus.MENTIONED,
                        source_evidence="Function contains bare 'raise' statement in exception handler.",
                        doc_evidence="Documentation acknowledges exception propagation or raising.",
                        message="Function contains bare 'raise' statement; documentation acknowledges exception handling.",
                    )
                )
            else:
                missing_exceptions.append("<re-raise>")
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject="<re-raise>",
                        status=AuditStatus.NOT_MENTIONED,
                        source_evidence="Function contains bare 'raise' statement in exception handler.",
                        doc_evidence="Documentation does not mention exception handling, re-raising, or propagation.",
                        message="Function contains bare 'raise' statement, but documentation does not acknowledge exception propagation.",
                    )
                )

        if missing_exceptions:
            exceptions_audit_status = AuditStatus.NOT_MENTIONED
        else:
            exceptions_audit_status = AuditStatus.MENTIONED

    else:
        # Source has no explicit raise statements
        if doc_facts.has_raises_section:
            clean_raises = raises_section_text.strip().lower()
            indicates_no_exc = (
                clean_raises in ("none", "none.", "n/a", "no", "no.", "no exceptions.", "no exceptions", "does not raise.")
                or "no exception" in clean_raises
                or "no exceptions" in clean_raises
                or "does not raise" in clean_raises
                or "not applicable" in clean_raises
            )
            if indicates_no_exc:
                exceptions_audit_status = AuditStatus.NOT_APPLICABLE
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject="<none>",
                        status=AuditStatus.NOT_APPLICABLE,
                        source_evidence="No explicit 'raise' statements in function AST.",
                        doc_evidence=f"Documentation explicitly notes no exceptions: '{raises_section_text.strip()}'",
                        message="No explicit raise statements found in AST; documentation explicitly indicates no exceptions are raised.",
                    )
                )
            elif len(clean_raises) > 3:
                exceptions_audit_status = AuditStatus.POSSIBLE_DISCREPANCY
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject="<none>",
                        status=AuditStatus.POSSIBLE_DISCREPANCY,
                        source_evidence="Source AST contains no explicit 'raise' statements in this function.",
                        doc_evidence=f"Documentation includes a 'Raises' section: '{raises_section_text[:100]}'",
                        message=(
                            "Documentation contains a 'Raises' section, but no explicit 'raise' statement "
                            "was found in the AST of this function (note: called functions or dynamic code might still raise errors)."
                        ),
                    )
                )
            else:
                exceptions_audit_status = AuditStatus.NOT_APPLICABLE
                findings.append(
                    AuditFinding(
                        category="exception",
                        subject="<none>",
                        status=AuditStatus.NOT_APPLICABLE,
                        source_evidence="No explicit 'raise' statements in function AST.",
                        doc_evidence="Raises section is empty.",
                        message="No explicit raise statements found in AST; Raises section is empty.",
                    )
                )
        else:
            exceptions_audit_status = AuditStatus.NOT_APPLICABLE
            findings.append(
                AuditFinding(
                    category="exception",
                    subject="<none>",
                    status=AuditStatus.NOT_APPLICABLE,
                    source_evidence="No explicit 'raise' statements in function AST.",
                    doc_evidence="Raises section is appropriately omitted.",
                    message="No explicit raise statements found in AST; documentation appropriately omits Raises section.",
                )
            )

    # ------------------------------------------------------------------------
    # Overall Audit Status Synthesis
    # ------------------------------------------------------------------------
    has_discrepancy = any(f.status == AuditStatus.POSSIBLE_DISCREPANCY for f in findings)
    has_unmentioned = any(f.status == AuditStatus.NOT_MENTIONED for f in findings)

    if has_discrepancy:
        overall_status = "DISCREPANCY_DETECTED"
    elif has_unmentioned:
        overall_status = "INCOMPLETE"
    else:
        overall_status = "CONSISTENT"

    summary = AuditSummary(
        total_parameters=total_params,
        parameters_mentioned=mentioned_params,
        parameters_coverage_ratio=coverage_ratio,
        return_status=return_audit_status,
        exceptions_status=exceptions_audit_status,
        overall_status=overall_status,
        disclaimer=AUDIT_DISCLAIMER,
    )

    return AuditReport(
        success=True,
        function_name=source_facts.function_name,
        summary=summary,
        findings=findings,
        source_evidence=source_facts,
        doc_evidence=doc_facts,
        disclaimer=AUDIT_DISCLAIMER,
    )
