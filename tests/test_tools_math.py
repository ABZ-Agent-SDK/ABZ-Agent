"""
Tests for abzagent/Tools/tools_math.py

Covers:
- Valid arithmetic expressions
- Parentheses
- Negative numbers (unary minus)
- Floating-point values
- Invalid syntax
- Division by zero
- Attempts to execute Python code via eval injection
- Attempts to import modules
- Attempts to access builtins
- Regression cases from the abzagent 0.3.0 CVE disclosure
"""
import pytest
from abzagent.Tools.tools_math import safe_eval, MathTool


# ---------------------------------------------------------------------------
# safe_eval — direct unit tests
# ---------------------------------------------------------------------------

class TestSafeEvalValidArithmetic:
    def test_addition(self):
        assert safe_eval("2 + 3") == 5

    def test_subtraction(self):
        assert safe_eval("10 - 4") == 6

    def test_multiplication(self):
        assert safe_eval("3 * 7") == 21

    def test_division(self):
        assert safe_eval("10 / 4") == 2.5

    def test_integer_division(self):
        # Floor division is not in the allowlist; should raise ValueError
        with pytest.raises((ValueError, KeyError)):
            safe_eval("10 // 3")

    def test_modulo(self):
        # Modulo is not in the allowlist; should raise ValueError
        with pytest.raises((ValueError, KeyError)):
            safe_eval("10 % 3")

    def test_exponentiation(self):
        assert safe_eval("2 ** 8") == 256

    def test_chained_operations(self):
        assert safe_eval("2 + 3 * 4") == 14  # operator precedence

    def test_float_result(self):
        assert safe_eval("1 / 3") == pytest.approx(0.3333333333333333)


class TestSafeEvalParentheses:
    def test_simple_parentheses(self):
        assert safe_eval("(2 + 3) * 4") == 20

    def test_nested_parentheses(self):
        assert safe_eval("((1 + 2) * (3 + 4))") == 21

    def test_parentheses_override_precedence(self):
        assert safe_eval("(1.5 + 2.5) * 2") == 8.0


class TestSafeEvalNegativeNumbers:
    def test_unary_minus(self):
        assert safe_eval("-5") == -5

    def test_unary_minus_in_expression(self):
        assert safe_eval("10 + -3") == 7

    def test_negative_float(self):
        assert safe_eval("-1.5 * 2") == -3.0

    def test_unary_plus(self):
        assert safe_eval("+5") == 5


class TestSafeEvalFloats:
    def test_float_literal(self):
        assert safe_eval("3.14") == pytest.approx(3.14)

    def test_float_arithmetic(self):
        assert safe_eval("1.5 + 2.5") == 4.0

    def test_float_multiplication(self):
        assert safe_eval("2.0 * 3.0") == 6.0


class TestSafeEvalInvalidSyntax:
    def test_empty_string(self):
        with pytest.raises(Exception):
            safe_eval("")

    def test_incomplete_expression(self):
        with pytest.raises(Exception):
            safe_eval("2 +")

    def test_double_operator(self):
        with pytest.raises(Exception):
            safe_eval("2 ** * 3")

    def test_non_numeric_string(self):
        with pytest.raises(Exception):
            safe_eval("hello")

    def test_string_literal(self):
        with pytest.raises(Exception):
            safe_eval('"hello"')


class TestSafeEvalDivisionByZero:
    def test_integer_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 / 0")

    def test_float_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1.0 / 0")


# ---------------------------------------------------------------------------
# Injection / code execution rejection
# ---------------------------------------------------------------------------

class TestSafeEvalRejectsCodeExecution:
    def test_rejects_import_statement(self):
        with pytest.raises(Exception):
            safe_eval("__import__('os')")

    def test_rejects_import_via_call(self):
        with pytest.raises(Exception):
            safe_eval("__import__('subprocess').run('id', shell=True)")

    def test_rejects_function_call(self):
        with pytest.raises(Exception):
            safe_eval("print('pwned')")

    def test_rejects_attribute_access(self):
        with pytest.raises(Exception):
            safe_eval("(1).__class__")

    def test_rejects_subscript(self):
        with pytest.raises(Exception):
            safe_eval("(1).__class__.__mro__[0]")

    def test_rejects_lambda(self):
        with pytest.raises(Exception):
            safe_eval("(lambda: 42)()")

    def test_rejects_list_comprehension(self):
        with pytest.raises(Exception):
            safe_eval("[x for x in range(10)]")

    def test_rejects_string_operations(self):
        with pytest.raises(Exception):
            safe_eval("'a' * 10")

    def test_rejects_boolean(self):
        with pytest.raises(Exception):
            safe_eval("True + 1")

    def test_rejects_none(self):
        with pytest.raises(Exception):
            safe_eval("None")

    def test_rejects_assignment_expression(self):
        with pytest.raises(Exception):
            safe_eval("x := 5")

    def test_rejects_comparison(self):
        with pytest.raises(Exception):
            safe_eval("1 == 1")

    def test_rejects_name_lookup(self):
        with pytest.raises(Exception):
            safe_eval("os")

    def test_rejects_getattr(self):
        with pytest.raises(Exception):
            safe_eval("getattr(int, '__subclasses__')()")


# ---------------------------------------------------------------------------
# Regression: exact payloads from the abzagent 0.3.0 disclosure
# ---------------------------------------------------------------------------

class TestRegressionDisclosurePayloads:
    """
    These are the two attack payloads proven to execute code in abzagent 0.3.0.
    safe_eval must reject both with a ValueError (not execute them).
    """

    def test_getpid_payload(self):
        # positive_builtins_eval case from BASELINE.txt
        with pytest.raises(Exception):
            safe_eval("__import__('os').getpid()")

    def test_file_write_payload(self):
        # positive_rce_side_effect case from BASELINE.txt
        import os, tempfile
        marker = os.path.join(tempfile.gettempdir(), "PWNED_RCE_MARKER.txt")
        if os.path.exists(marker):
            os.remove(marker)
        with pytest.raises(Exception):
            safe_eval(
                "open('"
                + marker.replace("\\", "/")
                + "', 'w').write('RCE_PROOF')"
            )
        assert not os.path.exists(marker), "RCE payload must not have written the marker file"


# ---------------------------------------------------------------------------
# MathTool.run() — integration tests via the Tool interface
# ---------------------------------------------------------------------------

class TestMathToolRun:
    def setup_method(self):
        self.tool = MathTool()

    def test_basic_arithmetic(self):
        result = self.tool.run(expression="2 + 3 * 4")
        assert result == "14"

    def test_float_result(self):
        result = self.tool.run(expression="(1.5 + 2.5) * 2")
        assert result == "8.0"

    def test_rejects_import_payload(self):
        result = self.tool.run(expression="__import__('os').getpid()")
        # Should return an error string, not a pid number
        assert "Error" in result or not result.isdigit()

    def test_rejects_string_expression(self):
        result = self.tool.run(expression="'hello'")
        assert "Error" in result or "Only" in result or "Invalid" in result
