import warnings

import cytnx as cy


def _same_line_default_cutoff(matrix):
    for _ in range(2):
        cy.linalg.Svd_truncate(matrix, 2)


def _explicit_zero_cutoff(matrix):
    cy.linalg.Svd_truncate(matrix, 2, err=0)


def _positive_cutoff(matrix):
    cy.linalg.Svd_truncate(matrix, 2, err=1e-8)


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
    assert "without an explicit err cutoff" in str(caught[0].message)
    assert "err=1e-8" in str(caught[0].message)
    assert "numerically null or near-null singular vectors" in str(caught[0].message)


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
        assert "no longer accepts err=0" in str(exc)
        assert "err=-1" in str(exc)
    else:
        raise AssertionError("Svd_truncate accepted err=0")


def test_svd_truncate_default_cutoff_removes_near_null_singular_vectors():
    matrix = cy.zeros([2, 2], dtype=cy.Type.Double)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0e-10
    cy.linalg.set_svd_truncate_warnings(True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_out = cy.linalg.Svd_truncate(matrix, 2)
    negative_out = cy.linalg.Svd_truncate(matrix, 2, err=-1)
    loose_positive_out = cy.linalg.Svd_truncate(matrix, 2, err=1e-8)
    tight_positive_out = cy.linalg.Svd_truncate(matrix, 2, err=1e-12)

    assert default_out[0].shape()[0] == 1
    assert loose_positive_out[0].shape()[0] == 1
    assert tight_positive_out[0].shape()[0] == 2
    assert negative_out[0].shape()[0] == 2
