import cytnx as cy


def test_type_helpers_are_available_from_python():
    assert cy.Type.is_complex(cy.Type.ComplexDouble)
    assert cy.Type.is_complex(cy.Type.ComplexFloat)
    assert not cy.Type.is_complex(cy.Type.Double)

    assert cy.Type.is_float(cy.Type.ComplexDouble)
    assert cy.Type.is_float(cy.Type.Float)
    assert not cy.Type.is_float(cy.Type.Int64)

    assert cy.Type.is_int(cy.Type.Int64)
    assert not cy.Type.is_int(cy.Type.Bool)
    assert not cy.Type.is_int(cy.Type.Double)

    assert cy.Type.is_unsigned(cy.Type.Uint64)
    assert not cy.Type.is_unsigned(cy.Type.Int64)


def test_type_promote_matches_expected_dtype_order():
    assert cy.Type.type_promote(cy.Type.Double, cy.Type.ComplexDouble) == cy.Type.ComplexDouble
    assert cy.Type.type_promote(cy.Type.Float, cy.Type.Double) == cy.Type.Double
    assert cy.Type.type_promote(cy.Type.ComplexFloat, cy.Type.Double) == cy.Type.ComplexFloat
    assert cy.Type.type_promote(cy.Type.Uint64, cy.Type.Int64) == cy.Type.Int64


def test_type_names_are_available_from_python():
    assert cy.Type.enum_name(cy.Type.ComplexDouble) == "ComplexDouble"
    assert "Double" in cy.Type.getname(cy.Type.Double)
    assert cy.Type.typeSize(cy.Type.Double) == 8
