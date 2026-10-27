"""
tests/test_optimization.py

Unit and regression test suite for CODEscribe Static Code Optimization Suggester.
Verifies all 4 optimization rules (OPT001-OPT004), scope handling, async defs,
syntax error tolerance, and absence of false positives.
"""

from __future__ import annotations

import pytest
from app.optimization import (
    OPTIMIZATION_DISCLAIMER,
    OptimizationFinding,
    OptimizationResult,
    analyze_optimizations,
)


class TestOptimizationEmptyAndSyntax:
    """Tests for baseline resilience, syntax errors, and whitespace inputs."""

    def test_empty_string_returns_success_zero_findings(self):
        res = analyze_optimizations("")
        assert res.success is True
        assert res.total_findings == 0
        assert res.findings == []
        assert res.disclaimer == OPTIMIZATION_DISCLAIMER

    def test_whitespace_and_comments_only(self):
        code = """
        # Just comments here
        # Another comment
        """
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0

    def test_invalid_input_type(self):
        res = analyze_optimizations(None)  # type: ignore
        assert res.success is False
        assert "must be a string" in (res.error_message or "")

    def test_syntax_error_handling(self):
        code = "def broken_func(:\n    pass"
        res = analyze_optimizations(code)
        assert res.success is False
        assert "Syntax error" in (res.error_message or "")
        assert res.total_findings == 0

    def test_result_to_dict_serialization(self):
        code = "def foo(a=[]): pass"
        res = analyze_optimizations(code)
        d = res.to_dict()
        assert d["success"] is True
        assert d["total_findings"] == 1
        assert len(d["findings"]) == 1
        assert d["findings"][0]["rule_id"] == "OPT001"
        assert d["findings"][0]["function_scope"] == "foo"


class TestOPT001MutableDefaults:
    """Tests for Rule OPT001: Mutable list/dict/set default arguments."""

    def test_detects_list_dict_set_in_regular_function(self):
        code = """
def process(items=[], mapping={}, seen=set()):
    pass
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 3
        rule_ids = [f.rule_id for f in res.findings]
        assert rule_ids == ["OPT001", "OPT001", "OPT001"]
        scopes = [f.function_scope for f in res.findings]
        assert all(s == "process" for s in scopes)

    def test_detects_mutable_defaults_in_async_function(self):
        code = """
async def fetch_data(cache={}):
    return cache
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        assert res.findings[0].rule_id == "OPT001"
        assert res.findings[0].function_scope == "fetch_data"

    def test_detects_kwonly_mutable_defaults(self):
        code = """
def configured(*args, options={}):
    pass
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        assert res.findings[0].rule_id == "OPT001"

    def test_ignores_immutable_defaults(self):
        code = """
def safe_func(a=None, b=0, c="default", d=(1, 2), e=True):
    pass
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0


class TestOPT002LiteralMembershipChecks:
    """Tests for Rule OPT002: Inefficient membership check against literal lists or tuples."""

    def test_detects_in_list_literal(self):
        code = """
def check(val):
    if val in [1, 2, 3]:
        return True
    return False
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        f = res.findings[0]
        assert f.rule_id == "OPT002"
        assert "literal list" in f.description
        assert f.function_scope == "check"

    def test_detects_notin_tuple_literal(self):
        code = """
def check(name):
    if name not in ("admin", "root", "superuser"):
        return False
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        f = res.findings[0]
        assert f.rule_id == "OPT002"
        assert "literal tuple" in f.description

    def test_ignores_set_literal_membership(self):
        code = """
def check(val):
    if val in {1, 2, 3}:
        return True
    if val not in {"a", "b"}:
        return False
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0

    def test_ignores_variable_or_dict_membership(self):
        code = """
def check(val, allowed_list):
    if val in allowed_list:
        return True
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0

    def test_suggestion_for_hashable_elements_recommends_set(self):
        """Hashable-constant sequence: suggestion should mention set or frozenset."""
        code = """
def check(val):
    if val in [1, 2, 3]:
        return True
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        # Must mention set / frozenset alternatives
        assert "set" in suggestion.lower() or "frozenset" in suggestion.lower()
        # Must NOT claim elements cannot be hashed (they are plain integers)
        assert "not be hashable" not in suggestion

    def test_suggestion_for_unhashable_elements_does_not_recommend_set_conversion(self):
        """Unhashable elements (nested lists): suggestion must NOT blindly say 'use a set literal'."""
        code = """
def check(val):
    if val in [[1, 2], [3, 4]]:
        return True
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        # Should warn about unhashable elements
        assert "not be hashable" in suggestion or "cannot" in suggestion.lower()
        # Must NOT recommend a plain set literal as-if it's always safe
        assert "Use a set literal" not in suggestion

    def test_suggestion_for_unhashable_dict_elements(self):
        """Dict literals as sequence elements are also unhashable."""
        code = """
def check(val):
    if val in [{"a": 1}, {"b": 2}]:
        return True
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        assert "not be hashable" in suggestion or "cannot" in suggestion.lower()

    def test_small_one_off_mention_in_hashable_suggestion(self):
        """Hashable-element suggestion should acknowledge one-off/small check trade-off."""
        code = """
def check(val):
    if val in ("a", "b"):
        return True
"""
        res = analyze_optimizations(code)
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        # Should acknowledge that for short/one-off checks the difference is negligible
        assert "one-off" in suggestion or "negligible" in suggestion


class TestOPT003RedundantListComprehension:
    """Tests for Rule OPT003: List comprehensions in reduction functions (sum, any, all, min, max)."""

    @pytest.mark.parametrize("func_name", ["sum", "any", "all", "min", "max"])
    def test_detects_list_comp_in_reductions(self, func_name):
        code = f"""
def compute(data):
    return {func_name}([x * 2 for x in data])
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        f = res.findings[0]
        assert f.rule_id == "OPT003"
        assert f"'{func_name}()'" in f.description
        assert f.function_scope == "compute"

    def test_ignores_generator_expression_in_reduction(self):
        code = """
def compute(data):
    s = sum(x * 2 for x in data)
    a = any(x > 0 for x in data)
    return s, a
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0

    def test_ignores_multi_argument_calls_to_min_max(self):
        code = """
def compute():
    # min with multiple separate arguments is not a single list comprehension reduction
    return min([1, 2], [3, 4])
"""
        res = analyze_optimizations(code)
        assert res.success is True
        # len(node.args) == 2, so it's not a single list comprehension
        opt003 = [f for f in res.findings if f.rule_id == "OPT003"]
        assert len(opt003) == 0

    def test_ignores_other_functions_taking_listcomp(self):
        code = """
def custom_worker(lst):
    return list([x for x in lst])
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0

    def test_reduction_with_kwargs_suggestion_uses_parentheses(self):
        """Reduction call with keyword args must suggest parenthesized generator expression."""
        code = """
def compute(data):
    return min([x for x in data], default=0)
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        f = res.findings[0]
        assert f.rule_id == "OPT003"
        # Suggestion must wrap generator in parentheses, not suggest bare 'min(x for x in ..., default=0)'
        suggestion = f.suggestion
        assert "parenthes" in suggestion.lower()
        # Must show the keyword argument in the hint
        assert "default=..." in suggestion

    def test_reduction_with_key_kwarg_suggestion_uses_parentheses(self):
        """max with key= keyword must be parenthesized in the suggestion."""
        code = """
def compute(data):
    return max([abs(x) for x in data], key=lambda v: -v)
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        assert "parenthes" in suggestion.lower()
        assert "key=..." in suggestion

    def test_reduction_with_start_kwarg_sum(self):
        """sum with start= keyword: should detect and use parenthesized hint."""
        code = """
def compute(data):
    return sum([x * 2 for x in data], start=0)
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        assert "parenthes" in suggestion.lower()
        assert "start=..." in suggestion

    def test_reduction_without_kwargs_suggestion_no_parentheses_required(self):
        """Reduction without keyword args should still suggest bare generator expression."""
        code = """
def compute(data):
    return sum([x * 2 for x in data])
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        suggestion = res.findings[0].suggestion
        # No keyword args, so no need for parenthesized-generator note
        assert "parenthes" not in suggestion.lower()
        # Should still describe omitting brackets
        assert "omit" in suggestion.lower() or "generator" in suggestion.lower()


class TestOPT004InLoopStringAccumulation:
    """Tests for Rule OPT004: In-loop augmented addition accumulator candidates."""

    def test_detects_augassign_in_for_loop(self):
        code = """
def build_output(chunks):
    s = ""
    for c in chunks:
        s += c
    return s
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        f = res.findings[0]
        assert f.rule_id == "OPT004"
        assert "Augmented addition" in f.description
        assert f.function_scope == "build_output"

    def test_detects_augassign_in_while_loop(self):
        code = """
def build_output(reader):
    buffer = ""
    while reader.has_next():
        buffer += reader.read()
    return buffer
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        assert res.findings[0].rule_id == "OPT004"

    def test_detects_augassign_in_async_for_loop(self):
        code = """
async def collect_stream(stream):
    out = ""
    async for packet in stream:
        out += packet
    return out
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        assert res.findings[0].rule_id == "OPT004"

    def test_ignores_augassign_outside_loops(self):
        code = """
def calculate(a, b):
    total = 0
    total += a
    total += b
    return total
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0

    def test_ignores_append_in_loop(self):
        code = """
def build_output(chunks):
    res = []
    for c in chunks:
        res.append(c)
    return "".join(res)
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 0


class TestScopeAndNestingIsolation:
    """Tests that scopes, classes, and loops do not leak across nested boundaries."""

    def test_nested_function_loop_does_not_leak_to_outer(self):
        code = """
def outer():
    for x in range(10):
        def inner():
            pass
        # loop in outer
        val += x
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        assert res.findings[0].function_scope == "outer"

    def test_outer_loop_does_not_cause_findings_inside_nested_function_body(self):
        code = """
def outer():
    for x in range(10):
        def inner():
            # This augmented assign is inside 'inner', which has NO loop of its own
            res += 1
            return res
"""
        res = analyze_optimizations(code)
        assert res.success is True
        # res += 1 is NOT in a loop within inner(), so OPT004 should NOT trigger
        assert res.total_findings == 0

    def test_class_method_scope(self):
        code = """
class DataHandler:
    def process(self, items=[]):
        pass
"""
        res = analyze_optimizations(code)
        assert res.success is True
        assert res.total_findings == 1
        assert res.findings[0].rule_id == "OPT001"
        assert res.findings[0].function_scope == "process"

    def test_deduplication_of_findings(self):
        code = """
def check(val):
    if val in [1, 2]:
        pass
"""
        res = analyze_optimizations(code)
        assert res.total_findings == 1
