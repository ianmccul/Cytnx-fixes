import cytnx as cy


def _dense_mps_tensor(dtype=cy.Type.ComplexDouble):
    block = cy.zeros([2, 3, 2], dtype=dtype)
    value = 1.0
    for left in range(2):
        for physical in range(3):
            for right in range(2):
                if dtype == cy.Type.ComplexDouble:
                    block[left, physical, right] = complex(value, 0.25 * value)
                else:
                    block[left, physical, right] = value
                value += 1.0
    tensor = cy.UniTensor(block, False, 2)
    tensor.relabel_(["l", "p", "r"])
    return tensor


def _as_dense_tensor(unitensor):
    return unitensor.get_block_() if hasattr(unitensor, "get_block_") else unitensor.get_block()


def _assert_reconstructs(actual, expected, tol=1e-12):
    actual.permute_(expected.labels())
    assert (actual - expected).Norm().item() < tol


def _assert_identity(unitensor, tol=1e-12):
    block = _as_dense_tensor(unitensor)
    ident = cy.eye(block.shape()[0], dtype=block.dtype())
    assert (block - ident).Norm().item() < tol


def test_qr_left_canonicalizes_dense_mps_tensor():
    tensor = _dense_mps_tensor()

    left_iso, remainder = cy.linalg.Qr(tensor)

    assert left_iso.labels() == ["l", "p", "_aux_"]
    assert remainder.labels() == ["_aux_", "r"]
    _assert_reconstructs(cy.Contract(left_iso, remainder), tensor)

    gram = cy.Contract(left_iso.Dagger().relabel("_aux_", "_aux_dag"), left_iso)
    assert gram.labels() == ["_aux_dag", "_aux_"]
    _assert_identity(gram)


def test_lq_right_canonicalizes_dense_mps_tensor():
    tensor = _dense_mps_tensor()

    remainder, right_iso = cy.linalg.Lq(tensor)

    assert remainder.labels() == ["l", "p", "_aux_"]
    assert right_iso.labels() == ["_aux_", "r"]
    _assert_reconstructs(cy.Contract(remainder, right_iso), tensor)

    gram = cy.Contract(right_iso, right_iso.Dagger().relabel("_aux_", "_aux_dag"))
    assert gram.labels() == ["_aux_", "_aux_dag"]
    _assert_identity(gram)
