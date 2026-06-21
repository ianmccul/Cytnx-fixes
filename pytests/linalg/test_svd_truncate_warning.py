import warnings

import cytnx as cy


def _same_line_default_cutoff(matrix):
    for _ in range(2):
        cy.linalg.Svd_truncate(matrix, 2)


def _explicit_zero_cutoff(matrix):
    cy.linalg.Svd_truncate(matrix, 2, err=0)


def _positive_cutoff(matrix):
    cy.linalg.Svd_truncate(matrix, 2, err=1e-12)


def test_svd_truncate_zero_cutoff_warns_once_per_line():
    matrix = cy.eye(2)
    cy.linalg.set_svd_truncate_warnings(True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _same_line_default_cutoff(matrix)
        _positive_cutoff(matrix)

    assert len(caught) == 1
    assert all(warning.category is RuntimeWarning for warning in caught)
    assert all(warning.filename == __file__ for warning in caught)
    assert "without an err cutoff" in str(caught[0].message)
    assert "err=1e-12" in str(caught[0].message)
    assert "numerically null singular vectors" in str(caught[0].message)


def test_svd_truncate_zero_cutoff_warning_can_be_disabled():
    matrix = cy.eye(2)
    cy.linalg.set_svd_truncate_warnings(False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cy.linalg.Svd_truncate(matrix, 2)

    assert caught == []
    cy.linalg.set_svd_truncate_warnings(True)


def test_svd_truncate_explicit_zero_cutoff_is_rejected():
    matrix = cy.eye(2)

    try:
        _explicit_zero_cutoff(matrix)
    except ValueError as exc:
        assert "err=0 is not legal" in str(exc)
        assert "negative err" in str(exc)
    else:
        raise AssertionError("Svd_truncate accepted err=0")


def test_svd_truncate_default_cutoff_removes_null_singular_vectors():
    matrix = cy.zeros([2, 2], dtype=cy.Type.Double)
    cy.linalg.set_svd_truncate_warnings(True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_out = cy.linalg.Svd_truncate(matrix, 2)
    negative_out = cy.linalg.Svd_truncate(matrix, 2, err=-1)
    positive_out = cy.linalg.Svd_truncate(matrix, 2, err=1e-12)

    assert default_out[0].shape()[0] == 1
    assert positive_out[0].shape()[0] == 1
    assert negative_out[0].shape()[0] == 2
