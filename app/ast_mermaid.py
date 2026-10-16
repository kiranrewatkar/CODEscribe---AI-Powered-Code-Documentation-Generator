"""
app/ast_mermaid.py

Deterministic Python AST-to-Mermaid flowchart generator for CODEscribe.
Parses Python source code into abstract syntax trees and builds deterministic,
syntax-safe Mermaid flowcharts without executing user code.

Key Capabilities:
- Parses valid Python source safely using standard library `ast`.
- Constructs Control Flow Graphs (CFG) for function entry/exit, sequential statements,
  conditional branching (`if/elif/else`), loops (`for`, `while`), and terminal returns.
- Preserves loop semantics with loopback edges, `break` jump-outs, and `continue` re-evaluations.
- Handles empty functions, docstring-only functions, and nested control flow safely.
- Robust label escaping preventing Mermaid syntax breakages on quotes and angle brackets.
- Never executes Python code (`eval` and `exec` are strictly prohibited).
- Gracefully captures syntax errors and unsupported grammar without application crashes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import html
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("CODEscribe.ASTMermaid")


@dataclass
class MermaidResult:
    """Result container for AST-to-Mermaid generation."""

    success: bool
    mermaid_code: str
    error_message: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    function_names: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    supported_constructs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "success": self.success,
            "mermaid_code": self.mermaid_code,
            "error_message": self.error_message,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "function_names": self.function_names,
            "warnings": self.warnings,
            "supported_constructs": self.supported_constructs,
        }


def escape_mermaid_label(text: str, max_length: int = 60) -> str:
    """
    Escape arbitrary Python text for safe rendering inside Mermaid quoted labels.

    Replaces double quotes, brackets, and angle brackets with safe representations
    and collapses newlines to avoid breaking Mermaid syntax.
    """
    if not text:
        return "pass"

    # Normalize whitespace & newlines
    clean = re.sub(r"\s+", " ", str(text)).strip()

    # Escape angle brackets to prevent Mermaid HTML tag interpretation
    clean = clean.replace("<", "&lt;").replace(">", "&gt;")

    # Replace double quotes with single quotes
    clean = clean.replace('"', "'")

    # Replace troublesome brackets inside quoted label
    clean = clean.replace("[", "&#91;").replace("]", "&#93;")
    clean = clean.replace("{", "&#123;").replace("}", "&#125;")

    # Truncate if excessively long
    if len(clean) > max_length:
        clean = clean[: max_length - 3] + "..."

    return clean


class ASTMermaidBuilder:
    """
    Deterministic AST visitor and Control Flow Graph generator.
    Translates Python AST nodes into a collection of Mermaid nodes and directed edges.
    """

    def __init__(self, max_label_len: int = 60) -> None:
        self.max_label_len = max_label_len
        self._node_counter = 0
        self.nodes: Dict[str, str] = {}  # node_id -> rendered node string
        self.edges: List[str] = []       # rendered edge strings
        self.warnings: List[str] = []
        self.supported_constructs: Set[str] = set()
        self.function_names: List[str] = []

    def _next_id(self, prefix: str = "N") -> str:
        """Generate a stable, deterministic node identifier."""
        node_id = f"{prefix}_{self._node_counter}"
        self._node_counter += 1
        return node_id

    def _add_node(self, node_id: str, label: str, shape: str = "rect") -> str:
        """Register a node with specific Mermaid shape formatting."""
        escaped = escape_mermaid_label(label, max_length=self.max_label_len)
        if shape == "stadium":
            rendered = f'{node_id}(["{escaped}"])'
        elif shape == "diamond":
            rendered = f'{node_id}{{"{escaped}"}}'
        elif shape == "circle":
            rendered = f'{node_id}(("{escaped}"))'
        elif shape == "subroutine":
            rendered = f'{node_id}[["{escaped}"]]'
        else:  # standard rectangle
            rendered = f'{node_id}["{escaped}"]'

        self.nodes[node_id] = rendered
        return node_id

    def _add_edge(self, source_id: str, target_id: str, label: Optional[str] = None) -> None:
        """Register a directed edge between two nodes with optional edge label."""
        if label:
            clean_label = label.strip().replace("|", "/")
            edge_str = f"{source_id} -->|{clean_label}| {target_id}"
        else:
            edge_str = f"{source_id} --> {target_id}"

        # Prevent duplicate identical edges
        if edge_str not in self.edges:
            self.edges.append(edge_str)

    def _connect_active_sources(
        self,
        active_sources: List[Tuple[str, Optional[str]]],
        target_id: str,
    ) -> None:
        """Connect all active incoming points to a given target node."""
        for src_id, edge_label in active_sources:
            self._add_edge(src_id, target_id, edge_label)

    def _safe_unparse(self, node: ast.AST) -> str:
        """Safely unparse an AST node back to Python source code."""
        try:
            return ast.unparse(node)
        except Exception:
            return type(node).__name__

    def _process_statements(
        self,
        statements: List[ast.stmt],
        active_sources: List[Tuple[str, Optional[str]]],
        loop_context: Optional[Dict[str, Any]] = None,
        end_node_id: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Sequentially process a list of AST statements and update active control flow endpoints.

        Returns:
            List of (node_id, optional_label) representing open endpoints exiting this block.
        """
        current_sources = list(active_sources)

        for stmt_idx, stmt in enumerate(statements):
            # If no incoming control flow reached here (e.g. following unconditional return/raise)
            if not current_sources:
                self.warnings.append(
                    f"Dead / unreachable code detected at line {getattr(stmt, 'lineno', '?')}: "
                    f"{self._safe_unparse(stmt)[:40]}"
                )
                break

            # 1. Sequential Assignment / Variable mutation
            if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                self.supported_constructs.add("assignment")
                node_id = self._next_id("stmt")
                expr_str = self._safe_unparse(stmt)
                self._add_node(node_id, expr_str, shape="rect")
                self._connect_active_sources(current_sources, node_id)
                current_sources = [(node_id, None)]

            # 2. Expression / Function Call
            elif isinstance(stmt, ast.Expr):
                # Check if it is a lone docstring in the middle of a function
                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    # Skip docstrings from flowchart to prevent visual clutter
                    continue
                self.supported_constructs.add("expression")
                node_id = self._next_id("expr")
                expr_str = self._safe_unparse(stmt)
                self._add_node(node_id, expr_str, shape="rect")
                self._connect_active_sources(current_sources, node_id)
                current_sources = [(node_id, None)]

            # 3. Pass statement
            elif isinstance(stmt, ast.Pass):
                self.supported_constructs.add("pass")
                node_id = self._next_id("pass")
                self._add_node(node_id, "pass", shape="rect")
                self._connect_active_sources(current_sources, node_id)
                current_sources = [(node_id, None)]

            # 4. Assert statement
            elif isinstance(stmt, ast.Assert):
                self.supported_constructs.add("assert")
                node_id = self._next_id("assert")
                test_str = f"assert {self._safe_unparse(stmt.test)}"
                self._add_node(node_id, test_str, shape="rect")
                self._connect_active_sources(current_sources, node_id)
                current_sources = [(node_id, None)]

            # 5. Return statement
            elif isinstance(stmt, ast.Return):
                self.supported_constructs.add("return")
                node_id = self._next_id("ret")
                ret_val = f"return {self._safe_unparse(stmt.value)}" if stmt.value else "return"
                self._add_node(node_id, ret_val, shape="stadium")
                self._connect_active_sources(current_sources, node_id)
                if end_node_id:
                    self._add_edge(node_id, end_node_id)
                # Return terminates execution on this path
                current_sources = []

            # 6. Raise statement
            elif isinstance(stmt, ast.Raise):
                self.supported_constructs.add("raise")
                node_id = self._next_id("raise")
                exc_val = f"raise {self._safe_unparse(stmt.exc)}" if stmt.exc else "raise"
                self._add_node(node_id, exc_val, shape="stadium")
                self._connect_active_sources(current_sources, node_id)
                if end_node_id:
                    self._add_edge(node_id, end_node_id, "Exception")
                current_sources = []

            # 7. Break statement
            elif isinstance(stmt, ast.Break):
                self.supported_constructs.add("break")
                node_id = self._next_id("break")
                self._add_node(node_id, "break", shape="rect")
                self._connect_active_sources(current_sources, node_id)
                if loop_context is not None:
                    loop_context["breaks"].append((node_id, None))
                else:
                    self.warnings.append("'break' outside loop context.")
                current_sources = []

            # 8. Continue statement
            elif isinstance(stmt, ast.Continue):
                self.supported_constructs.add("continue")
                node_id = self._next_id("continue")
                self._add_node(node_id, "continue", shape="rect")
                self._connect_active_sources(current_sources, node_id)
                if loop_context is not None:
                    loop_context["continues"].append((node_id, None))
                else:
                    self.warnings.append("'continue' outside loop context.")
                current_sources = []

            # 9. Conditional branching: if / elif / else
            elif isinstance(stmt, ast.If):
                self.supported_constructs.add("if_condition")
                cond_id = self._next_id("cond")
                cond_str = f"if {self._safe_unparse(stmt.test)}"
                self._add_node(cond_id, cond_str, shape="diamond")
                self._connect_active_sources(current_sources, cond_id)

                # Process True branch (body)
                true_exits = self._process_statements(
                    stmt.body,
                    active_sources=[(cond_id, "True")],
                    loop_context=loop_context,
                    end_node_id=end_node_id,
                )

                # Process False branch (orelse)
                if stmt.orelse:
                    self.supported_constructs.add("else_branch")
                    false_exits = self._process_statements(
                        stmt.orelse,
                        active_sources=[(cond_id, "False")],
                        loop_context=loop_context,
                        end_node_id=end_node_id,
                    )
                else:
                    false_exits = [(cond_id, "False")]

                # Combine active exits from both branches
                current_sources = true_exits + false_exits

            # 10. While loop
            elif isinstance(stmt, ast.While):
                self.supported_constructs.add("while_loop")
                while_id = self._next_id("while")
                cond_str = f"while {self._safe_unparse(stmt.test)}"
                self._add_node(while_id, cond_str, shape="diamond")
                self._connect_active_sources(current_sources, while_id)

                inner_loop_context = {"breaks": [], "continues": []}

                # Process loop body
                body_exits = self._process_statements(
                    stmt.body,
                    active_sources=[(while_id, "True")],
                    loop_context=inner_loop_context,
                    end_node_id=end_node_id,
                )

                # Loop back to header for normal body exits and continue statements
                for exit_node_id, label in body_exits + inner_loop_context["continues"]:
                    self._add_edge(exit_node_id, while_id, label)

                # False exit from while test
                if stmt.orelse:
                    self.supported_constructs.add("loop_orelse")
                    orelse_exits = self._process_statements(
                        stmt.orelse,
                        active_sources=[(while_id, "False")],
                        loop_context=loop_context,
                        end_node_id=end_node_id,
                    )
                    current_sources = orelse_exits + inner_loop_context["breaks"]
                else:
                    current_sources = [(while_id, "False")] + inner_loop_context["breaks"]

            # 11. For loop (For and AsyncFor)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                self.supported_constructs.add("for_loop")
                for_id = self._next_id("for")
                prefix = "async for" if isinstance(stmt, ast.AsyncFor) else "for"
                iter_str = f"{prefix} {self._safe_unparse(stmt.target)} in {self._safe_unparse(stmt.iter)}"
                self._add_node(for_id, iter_str, shape="diamond")
                self._connect_active_sources(current_sources, for_id)

                inner_loop_context = {"breaks": [], "continues": []}

                # Process loop body
                body_exits = self._process_statements(
                    stmt.body,
                    active_sources=[(for_id, "Next")],
                    loop_context=inner_loop_context,
                    end_node_id=end_node_id,
                )

                # Loop back to header
                for exit_node_id, label in body_exits + inner_loop_context["continues"]:
                    self._add_edge(exit_node_id, for_id, label)

                # Done exit from for loop
                if stmt.orelse:
                    self.supported_constructs.add("loop_orelse")
                    orelse_exits = self._process_statements(
                        stmt.orelse,
                        active_sources=[(for_id, "Done")],
                        loop_context=loop_context,
                        end_node_id=end_node_id,
                    )
                    current_sources = orelse_exits + inner_loop_context["breaks"]
                else:
                    current_sources = [(for_id, "Done")] + inner_loop_context["breaks"]

            # 12. With statement (context manager)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                self.supported_constructs.add("with_block")
                with_id = self._next_id("with")
                items_str = ", ".join(self._safe_unparse(it) for it in stmt.items)
                prefix = "async with" if isinstance(stmt, ast.AsyncWith) else "with"
                self._add_node(with_id, f"{prefix} {items_str}", shape="rect")
                self._connect_active_sources(current_sources, with_id)

                current_sources = self._process_statements(
                    stmt.body,
                    active_sources=[(with_id, None)],
                    loop_context=loop_context,
                    end_node_id=end_node_id,
                )

            # 13. Try / Except / Finally block
            elif isinstance(stmt, ast.Try):
                self.supported_constructs.add("try_block")
                try_id = self._next_id("try")
                self._add_node(try_id, "try", shape="rect")
                self._connect_active_sources(current_sources, try_id)

                body_exits = self._process_statements(
                    stmt.body,
                    active_sources=[(try_id, None)],
                    loop_context=loop_context,
                    end_node_id=end_node_id,
                )

                handler_exits: List[Tuple[str, Optional[str]]] = []
                for handler in stmt.handlers:
                    self.supported_constructs.add("except_handler")
                    exc_name = self._safe_unparse(handler.type) if handler.type else "Exception"
                    if handler.name:
                        exc_name += f" as {handler.name}"
                    handler_id = self._next_id("except")
                    self._add_node(handler_id, f"except {exc_name}", shape="rect")
                    self._add_edge(try_id, handler_id, "on error")

                    h_exits = self._process_statements(
                        handler.body,
                        active_sources=[(handler_id, None)],
                        loop_context=loop_context,
                        end_node_id=end_node_id,
                    )
                    handler_exits.extend(h_exits)

                # Finally block (if present)
                if stmt.finalbody:
                    self.supported_constructs.add("finally_block")
                    finally_id = self._next_id("finally")
                    self._add_node(finally_id, "finally", shape="rect")
                    all_prior_exits = body_exits + handler_exits
                    self._connect_active_sources(all_prior_exits, finally_id)
                    current_sources = self._process_statements(
                        stmt.finalbody,
                        active_sources=[(finally_id, None)],
                        loop_context=loop_context,
                        end_node_id=end_node_id,
                    )
                else:
                    current_sources = body_exits + handler_exits

            # 14. Fallback for unspecialized AST statements
            else:
                stmt_name = type(stmt).__name__
                self.supported_constructs.add(stmt_name)
                node_id = self._next_id("stmt")
                expr_str = self._safe_unparse(stmt)
                self._add_node(node_id, expr_str, shape="rect")
                self._connect_active_sources(current_sources, node_id)
                current_sources = [(node_id, None)]

        return current_sources

    def build_function_subgraph(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        """Construct Mermaid flowchart sub-structure for a single Python function."""
        func_name = func_node.name
        self.function_names.append(func_name)

        # Build clean parameter signature
        args = [arg.arg for arg in func_node.args.args]
        sig = f"{func_name}({', '.join(args)})"

        start_id = self._next_id("start")
        end_id = self._next_id("end")

        start_label = f"Start: {sig}"
        self._add_node(start_id, start_label, shape="stadium")
        self._add_node(end_id, f"End ({func_name})", shape="stadium")

        # Strip initial docstring if present
        stmts = list(func_node.body)
        if stmts and isinstance(stmts[0], ast.Expr):
            first_val = getattr(stmts[0], "value", None)
            if isinstance(first_val, ast.Constant) and isinstance(first_val.value, str):
                stmts = stmts[1:]

        # If function body is completely empty (or had only docstrings)
        if not stmts:
            self._add_edge(start_id, end_id)
            return func_name

        # Process statements
        active_exits = self._process_statements(
            statements=stmts,
            active_sources=[(start_id, None)],
            end_node_id=end_id,
        )

        # Connect any non-returned active endpoints to End node
        for exit_node_id, label in active_exits:
            self._add_edge(exit_node_id, end_id, label)

        return func_name

    def build_flowchart(self, tree: ast.Module, direction: str = "TD") -> str:
        """Assemble all nodes and edges into valid Mermaid graph TD specification."""
        functions: List[ast.FunctionDef | ast.AsyncFunctionDef] = []

        # Find top-level functions and class methods
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(item)
            elif isinstance(item, ast.ClassDef):
                for sub in item.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Prefix method with class name for clarity
                        sub.name = f"{item.name}.{sub.name}"
                        functions.append(sub)

        # If functions were found, build for each function
        if functions:
            for fn in functions:
                self.build_function_subgraph(fn)
        else:
            # Plain sequential script without function defs
            start_id = self._next_id("start")
            end_id = self._next_id("end")
            self._add_node(start_id, "Start: script", shape="stadium")
            self._add_node(end_id, "End", shape="stadium")
            active_exits = self._process_statements(
                statements=tree.body,
                active_sources=[(start_id, None)],
                end_node_id=end_id,
            )
            for exit_node_id, label in active_exits:
                self._add_edge(exit_node_id, end_id, label)

        # Assemble Mermaid text
        lines = [f"graph {direction}"]

        # Add node definitions
        for node_id in sorted(self.nodes.keys()):
            lines.append(f"    {self.nodes[node_id]}")

        # Add edges
        for edge_str in self.edges:
            lines.append(f"    {edge_str}")

        return "\n".join(lines)


def generate_mermaid_flowchart(
    source_code: str,
    max_label_len: int = 60,
    direction: str = "TD",
) -> MermaidResult:
    """
    Primary API entry point:
    Deterministically transforms Python source code into a Mermaid flowchart.

    Args:
        source_code: Raw Python source code string.
        max_label_len: Maximum character length for individual node text.
        direction: Flowchart orientation ('TD' for top-down, 'LR' for left-to-right).

    Returns:
        MermaidResult containing success status, mermaid_code, and metrics.
    """
    if not source_code or not source_code.strip():
        empty_chart = f"graph {direction}\n    Empty([\"Empty Source\"])"
        return MermaidResult(
            success=True,
            mermaid_code=empty_chart,
            node_count=1,
            edge_count=0,
            warnings=["Input source code was empty or contained only whitespace."],
            supported_constructs=["empty_source"],
        )

    # 1. Parse Python source with standard library ast (no code execution)
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        line = exc.lineno or 1
        col = exc.offset or 1
        err_msg = f"SyntaxError at line {line}, col {col}: {exc.msg}"
        logger.warning("AST parsing failed on user source code: %s", err_msg)
        return MermaidResult(
            success=False,
            mermaid_code="",
            error_message=err_msg,
            warnings=["Submitted source code has Python syntax errors."],
        )
    except Exception as exc:
        err_msg = f"Parsing failed: {str(exc)}"
        logger.error("Unexpected error during ast.parse: %s", exc)
        return MermaidResult(
            success=False,
            mermaid_code="",
            error_message=err_msg,
        )

    # 2. Build Mermaid Flowchart
    builder = ASTMermaidBuilder(max_label_len=max_label_len)
    mermaid_code = builder.build_flowchart(tree, direction=direction)

    return MermaidResult(
        success=True,
        mermaid_code=mermaid_code,
        node_count=len(builder.nodes),
        edge_count=len(builder.edges),
        function_names=builder.function_names,
        warnings=builder.warnings,
        supported_constructs=sorted(list(builder.supported_constructs)),
    )
