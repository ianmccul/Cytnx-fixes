import cytnx as cy
import cytnx.linalg_conti as linalg_conti


def test_complex_linop_hint_warning_is_specific():
    stats = {
        "input_dtype": cy.Type.Double,
        "input_dtype_name": cy.Type.getname(cy.Type.Double),
        "op_dtype": cy.Type.ComplexDouble,
        "op_dtype_name": cy.Type.getname(cy.Type.ComplexDouble),
        "working_dtype": cy.Type.ComplexDouble,
        "working_dtype_name": cy.Type.getname(cy.Type.ComplexDouble),
    }

    warnings = linalg_conti._format_krylov_dtype_warnings(stats, [])

    assert len(warnings) == 1
    assert "LinOp dtype hint" in warnings[0]
    assert "promoted real input dtype" in warnings[0]
    assert "real operator" in warnings[0]


def test_non_linop_complex_promotion_uses_generic_warning():
    stats = {
        "input_dtype": cy.Type.Float,
        "input_dtype_name": cy.Type.getname(cy.Type.Float),
        "op_dtype": cy.Type.Double,
        "op_dtype_name": cy.Type.getname(cy.Type.Double),
        "working_dtype": cy.Type.Double,
        "working_dtype_name": cy.Type.getname(cy.Type.Double),
    }

    warnings = linalg_conti._format_krylov_dtype_warnings(stats, [])

    assert len(warnings) == 1
    assert "Lanczos working dtype changed" in warnings[0]


def test_lanczos_exp_suppresses_expected_output_dtype_change_warning():
    stats = {
        "algorithm": "Lanczos_Exp",
        "input_dtype": cy.Type.Double,
        "input_dtype_name": cy.Type.getname(cy.Type.Double),
        "op_dtype": cy.Type.Double,
        "op_dtype_name": cy.Type.getname(cy.Type.Double),
        "working_dtype": cy.Type.Double,
        "working_dtype_name": cy.Type.getname(cy.Type.Double),
    }
    output = cy.UniTensor.zeros([1, 1], [], cy.Type.ComplexDouble)

    warnings = linalg_conti._format_krylov_dtype_warnings(stats, output)

    assert warnings == []


def test_lanczos_exp_is_wrapped_for_diagnostics():
    assert hasattr(cy.linalg.Lanczos_Exp, "__cytnx_native_lanczos_exp__")


def test_lanczos_exp_diagnostics_do_not_read_result_as_eigenvalue():
    stats = {
        "algorithm": "Lanczos_Exp",
        "final_error": 1e-9,
        "cvgcrit_used": 1e-8,
        "matvec_count": 3,
        "krylov_dim": 3,
        "reason": "projected_exponential",
        "input_dtype_name": cy.Type.getname(cy.Type.Double),
    }
    result = cy.UniTensor.zeros([1, 1], [], cy.Type.Double)

    text = linalg_conti._format_krylov_diagnostics(stats, (), {}, result)

    assert "Lanczos_Exp:" in text
    assert "err=1e-09/1e-08" in text
    assert "E=" not in text
