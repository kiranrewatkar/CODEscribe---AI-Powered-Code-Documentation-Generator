"""
training/remove_docstrings.py

Robust, AST-guided docstring removal and data leakage prevention engine for CODEscribe.

Functions:
- Identifies genuine docstrings in modules, classes, and functions (sync & async, nested).
- Removes docstring expressions from source code while preserving Python indentation,
  decorators, comments, and ordinary string literals.
- Replaces docstrings with 'pass' when the docstring was the sole statement in a scope.
- Enforces strict multi-tier leakage verification to ensure target documentation
  is absent from model inputs.
- Preserves raw and processed CSV integrity by operating as an in-memory/caching transformation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings

logger = logging.getLogger("CODEscribe.DocstringRemover")


@dataclass(frozen=True)
class DocstringSpan:
    """Location and metadata for an AST-identified docstring."""

    start_line: int  # 1-indexed
    start_col: int   # 0-indexed
    end_line: int    # 1-indexed
    end_col: int     # 0-indexed
    docstring_text: str
    is_single_statement: bool
    scope_type: str
    scope_name: str


def find_docstring_spans(tree: ast.AST) -> List[DocstringSpan]:
    """
    Traverse an AST to locate genuine docstring expression spans.

    Per PEP 257, a docstring is an expression statement containing a string literal
    as the first statement in a module, class, or function body.

    Ordinary string assignments (e.g. `x = "..."`), return expressions, or strings
    inside expressions are not docstrings and will be preserved.

    Args:
        tree: Parsed Python AST.

    Returns:
        List of DocstringSpan objects identifying genuine docstrings.
    """
    spans: List[DocstringSpan] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if not node.body:
                continue

            first_stmt = node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                val = first_stmt.value.value
                if isinstance(val, str):
                    scope_name = getattr(node, "name", "<module>")
                    is_single = len(node.body) == 1
                    # Python 3.8+ AST attributes
                    span = DocstringSpan(
                        start_line=first_stmt.lineno,
                        start_col=first_stmt.col_offset,
                        end_line=first_stmt.end_lineno,
                        end_col=first_stmt.end_col_offset,
                        docstring_text=val,
                        is_single_statement=is_single,
                        scope_type=type(node).__name__,
                        scope_name=scope_name,
                    )
                    spans.append(span)

    return spans


def remove_docstrings_from_source(
    source_code: str,
    target_doc: Optional[str] = None,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Remove docstrings from Python source code safely using AST span derivation.

    Args:
        source_code: Python function or module source code string.
        target_doc: Optional target documentation string to verify leakage elimination.

    Returns:
        Tuple of (clean_source_code_or_None, status_code, audit_metadata).
        Status codes:
        - 'SUCCESS': Docstring removed, syntax verified, zero leakage detected.
        - 'NO_DOCSTRING_PRESENT': Code is valid and already docstring-free.
        - 'REJECT_SYNTAX_ERROR': Initial code has invalid Python syntax (e.g. Python 2).
        - 'REJECT_TRANSFORM_SYNTAX_ERROR': Resulting code failed syntax verification.
        - 'REJECT_TARGET_LEAK_DETECTED': Target documentation remained in clean code.
    """
    if not source_code or not source_code.strip():
        return None, "REJECT_EMPTY_CODE", {"error": "Source code is empty"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SyntaxWarning)
        try:
            tree = ast.parse(source_code)
        except SyntaxError as syn_err:
            return None, "REJECT_SYNTAX_ERROR", {
                "error_message": syn_err.msg,
                "lineno": syn_err.lineno,
                "offset": syn_err.offset,
            }

    spans = find_docstring_spans(tree)
    lines = source_code.splitlines(keepends=True)

    if not spans:
        # Check if the target documentation text is present in the code despite no AST docstring
        if target_doc and is_target_leaked(target_doc, source_code, spans):
            return None, "REJECT_TARGET_LEAK_DETECTED", {
                "reason": "Target text found in code without AST docstring header",
            }
        return source_code, "NO_DOCSTRING_PRESENT", {
            "spans_removed": 0,
            "initial_lines": len(lines),
            "final_lines": len(lines),
        }

    # Sort spans in reverse order (bottom-to-top) so earlier line indices remain valid
    spans.sort(key=lambda s: (s.start_line, s.start_col), reverse=True)

    for span in spans:
        start_idx = span.start_line - 1  # 0-indexed
        end_idx = span.end_line - 1      # 0-indexed

        if start_idx < 0 or end_idx >= len(lines):
            continue

        if span.is_single_statement:
            # Scope has ONLY the docstring. Replace with 'pass' to preserve valid syntax.
            first_prefix = lines[start_idx][:span.start_col]
            last_suffix = lines[end_idx][span.end_col:]
            # Ensure proper line ending
            line_ending = "\n" if lines[end_idx].endswith("\n") else ""
            pass_stmt = f"{first_prefix}pass{last_suffix}"
            if not pass_stmt.endswith("\n") and line_ending:
                pass_stmt += line_ending
            lines[start_idx:end_idx + 1] = [pass_stmt]
        else:
            # Scope has other statements. Remove the docstring span cleanly.
            first_prefix = lines[start_idx][:span.start_col]
            last_suffix = lines[end_idx][span.end_col:]

            if first_prefix.strip() == "" and last_suffix.strip() == "":
                # Docstring occupies whole lines. Delete those lines.
                del lines[start_idx:end_idx + 1]
            else:
                # Docstring is inline on a shared line (e.g. `def f(): """doc"""; return 1`)
                rem = f"{first_prefix}{last_suffix}"
                if rem.strip():
                    lines[start_idx:end_idx + 1] = [rem]
                else:
                    del lines[start_idx:end_idx + 1]

    transformed_code = "".join(lines)

    # 1. Syntax Validation: Ensure transformed code parses cleanly as valid Python AST
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SyntaxWarning)
        try:
            new_tree = ast.parse(transformed_code)
        except SyntaxError as trans_err:
            return None, "REJECT_TRANSFORM_SYNTAX_ERROR", {
                "error_message": trans_err.msg,
                "lineno": trans_err.lineno,
                "offset": trans_err.offset,
            }

    # 2. Multi-tier Leakage Check
    new_spans = find_docstring_spans(new_tree)
    if target_doc and is_target_leaked(target_doc, transformed_code, new_spans):
        return None, "REJECT_TARGET_LEAK_DETECTED", {
            "reason": "Target documentation remained detectable in transformed code",
            "target_doc_preview": target_doc[:60],
        }

    metadata = {
        "spans_removed": len(spans),
        "initial_lines": len(source_code.splitlines()),
        "final_lines": len(transformed_code.splitlines()),
        "docstring_scopes": [s.scope_name for s in spans],
    }

    return transformed_code, "SUCCESS", metadata


def is_target_leaked(
    target_doc: str,
    clean_code: str,
    remaining_docstring_spans: List[DocstringSpan],
) -> bool:
    """
    Multi-tier data leakage verification:
    1. Checks if any remaining docstring in clean_code matches the target doc.
    2. Punctuation- and whitespace-insensitive phrase check for non-trivial documentation (>=20 chars, >=3 words).
    3. Verbatim substring check.
    """
    clean_target = target_doc.strip()
    if not clean_target:
        return False

    # Tier 1: Check AST docstrings in transformed code
    for span in remaining_docstring_spans:
        if clean_target in span.docstring_text or span.docstring_text in clean_target:
            return True

    # Tier 2: Normalized alphanumeric phrase check (punctuation- and line-break insensitive)
    target_words = re.sub(r"[^\w\s]", " ", clean_target).lower().split()
    if len(target_words) >= 3 and len(clean_target) >= 20:
        target_phrase = " ".join(target_words)
        code_words = re.sub(r"[^\w\s]", " ", clean_code).lower().split()
        code_phrase = " ".join(code_words)
        if target_phrase in code_phrase:
            return True

    # Tier 3: Verbatim exact check if target is distinct from code keywords
    if len(clean_target) >= 15 and clean_target in clean_code:
        return True

    return False
