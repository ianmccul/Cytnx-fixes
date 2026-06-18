import cytnx as cy
import pytest


def test_type_helpers_are_available_from_python():
    assert cy.Type.is_complex(cy.Type.ComplexDouble)
    assert cy.Type.is_complex(cy.Type.ComplexFloat)
    assert not cy.Type.is_complex(cy.Type.Double)

    assert cy.Type.is_floating(cy.Type.ComplexDouble)
    assert cy.Type.is_floating(cy.Type.Float)
    assert not cy.Type.is_floating(cy.Type.Int64)

    assert not cy.Type.is_real(cy.Type.ComplexDouble)
    assert cy.Type.is_real(cy.Type.Double)
    assert cy.Type.is_real(cy.Type.Float)
    assert not cy.Type.is_real(cy.Type.Int64)

    assert cy.Type.is_integer(cy.Type.Int64)
    assert not cy.Type.is_integer(cy.Type.Bool)
    assert not cy.Type.is_integer(cy.Type.Double)

    assert cy.Type.is_unsigned(cy.Type.Uint64)
    assert not cy.Type.is_unsigned(cy.Type.Int64)
    assert cy.Type.is_signed(cy.Type.Int64)
    assert not cy.Type.is_signed(cy.Type.Uint64)


def test_type_conversion_helpers_are_available_from_python():
    assert cy.Type.as_complex(cy.Type.Double) == cy.Type.ComplexDouble
    assert cy.Type.as_complex(cy.Type.Float) == cy.Type.ComplexFloat
    assert cy.Type.as_complex(cy.Type.Int64) == cy.Type.ComplexDouble

    assert cy.Type.as_real(cy.Type.ComplexDouble) == cy.Type.Double
    assert cy.Type.as_real(cy.Type.ComplexFloat) == cy.Type.Float
    assert cy.Type.as_real(cy.Type.Int64) == cy.Type.Int64

    assert cy.Type.as_single_prec(cy.Type.ComplexDouble) == cy.Type.ComplexFloat
    assert cy.Type.as_single_prec(cy.Type.Double) == cy.Type.Float
    assert cy.Type.as_single_prec(cy.Type.Int64) == cy.Type.Float

    assert cy.Type.as_double_prec(cy.Type.ComplexFloat) == cy.Type.ComplexDouble
    assert cy.Type.as_double_prec(cy.Type.Float) == cy.Type.Double
    assert cy.Type.as_double_prec(cy.Type.Int32) == cy.Type.Double

    assert cy.Type.as_signed(cy.Type.Uint64) == cy.Type.Int64
    assert cy.Type.as_signed(cy.Type.Uint32) == cy.Type.Int32
    assert cy.Type.as_signed(cy.Type.Int16) == cy.Type.Int16

    assert cy.Type.as_unsigned(cy.Type.Int64) == cy.Type.Uint64
    assert cy.Type.as_unsigned(cy.Type.Int32) == cy.Type.Uint32
    assert cy.Type.as_unsigned(cy.Type.Uint16) == cy.Type.Uint16

    with pytest.raises(RuntimeError):
        cy.Type.as_signed(cy.Type.Double)
    with pytest.raises(RuntimeError):
        cy.Type.as_unsigned(cy.Type.Bool)


def test_type_promote_matches_expected_dtype_order():
    assert cy.Type.type_promote(cy.Type.Double, cy.Type.ComplexDouble) == cy.Type.ComplexDouble
    assert cy.Type.type_promote(cy.Type.Float, cy.Type.Double) == cy.Type.Double
    assert cy.Type.type_promote(cy.Type.ComplexFloat, cy.Type.Double) == cy.Type.ComplexFloat
    assert cy.Type.type_promote(cy.Type.Uint64, cy.Type.Int64) == cy.Type.Int64


def test_type_names_are_available_from_python():
    assert cy.Type.enum_name(cy.Type.ComplexDouble) == "ComplexDouble"
    assert "Double" in cy.Type.getname(cy.Type.Double)
    assert cy.Type.typeSize(cy.Type.Double) == 8
