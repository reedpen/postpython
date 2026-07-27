"""Compiler profile diagnostics for valid but unsupported POST features."""

import ast

from postpyc.compiler.frontend import compile_source
from postpyc.compiler.typechecker import resolve_annotation_info
from postyp import Float16


def compile_errors(source: str) -> list:
    _, errors = compile_source(source)
    return errors


def test_dataframe_annotation_reports_unsupported_profile():
    source = """\
from postyp import DataFrame

def f(df: DataFrame) -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "POST DataFrame profile" in errors[0].message


def test_series_annotation_reports_unsupported_profile():
    source = """\
from postyp import Series, Float64

def f(s: Series[Float64]) -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "POST DataFrame profile" in errors[0].message


def test_optional_annotation_reports_unsupported_valid_annotation():
    source = """\
from typing import Optional

def f(x: Optional[int]) -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "`Optional` annotations" in errors[0].message


def test_top_level_expression_reports_unsupported_instead_of_being_ignored():
    source = """\
def f() -> int:
    return 1

f()
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "top-level executable expressions" in errors[0].message


def test_module_docstring_is_allowed_at_top_level():
    source = '''\
"""module docs"""

def f() -> int:
    return 1
'''

    assert compile_errors(source) == []


def test_class_definition_reports_unsupported_instead_of_being_ignored():
    source = """\
class Point:
    x: int

def f() -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "class/dataclass definitions" in errors[0].message


# ---------------------------------------------------------------------------
# Float16 (postpython#44): the C backend maps it to uint16_t, so arithmetic,
# comparison, and casts would run on binary16 bit patterns as integers.  Every
# route that can introduce a Float16 value must diagnose rather than emit code.
# ---------------------------------------------------------------------------

def test_float16_parameter_reports_unsupported_dtype():
    source = """\
from postyp import Float16

def f(x: Float16, y: Float16) -> Float16:
    return x + y
"""

    errors = compile_errors(source)

    assert "PP900" in [error.code for error in errors]
    assert "`Float16`" in errors[0].message
    assert "binary16" in errors[0].message


def test_float16_shorthand_spelling_reports_unsupported_dtype():
    source = """\
from postyp import f16

def f(x: f16) -> f16:
    return x
"""

    errors = compile_errors(source)

    assert "PP900" in [error.code for error in errors]
    assert "`Float16`" in errors[0].message


def test_float16_array_element_reports_unsupported_dtype():
    source = """\
from postpyc import guvectorize
from postyp import Array, Float16

@guvectorize([], "(n)->()")
def g(a: Array[Float16], out: Array[Float16]) -> None:
    out[0] = a[0]
"""

    errors = compile_errors(source)

    assert "PP900" in [error.code for error in errors]
    assert "`Float16`" in errors[0].message


def test_float16_local_annotation_reports_unsupported_dtype():
    source = """\
from postyp import Float16, Float64

def f(x: Float64) -> Float64:
    t: Float16 = 1.0
    return x + 1.0
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "`Float16`" in errors[0].message


def test_float16_cast_reports_unsupported_dtype():
    source = """\
from postyp import Float16, Float64

def f(x: Float64) -> Float16:
    return Float16(x)
"""

    errors = compile_errors(source)

    assert "PP900" in [error.code for error in errors]
    assert "`Float16`" in errors[0].message


def test_float16_module_constant_is_not_compiled_into_the_artifact():
    source = """\
from postyp import Float16, Float64

HALF: Float16 = 1.0

def f(x: Float64) -> Float64:
    return x + HALF
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]


def test_float16_annotation_still_resolves_as_a_dtype():
    """Spec §4.1 vocabulary is unchanged; only lowering support is withheld."""
    node = ast.parse("Float16", mode="eval").body

    resolved = resolve_annotation_info(node)

    assert resolved.dtype is Float16
    assert resolved.is_valid
    assert not resolved.is_supported


def test_other_float_widths_still_compile():
    source = """\
from postyp import Float32, Float64

def f(x: Float32, y: Float32) -> Float32:
    return x + y

def g(x: Float64) -> Float64:
    return Float64(x) * 2.0
"""

    assert compile_errors(source) == []
