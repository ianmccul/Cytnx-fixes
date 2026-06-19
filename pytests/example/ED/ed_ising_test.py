import os
import subprocess
import sys
import textwrap

import numpy as np
sys.path.append('example/ED/')

from ed_ising import *

def test_ed_ising():
    L = 4
    J = 1
    Hx = 0.3
    H = Hising(L,J,Hx)
    v = cy.ones(16)
    res = cy.linalg.Lanczos(Hop = H, Tin = v, which = "SA")
    assert np.abs(res[0].item()-(-4.092961599426854)) < 1e-4

def test_lanczos_er_disabled_message():
    code = textwrap.dedent("""
        import cytnx as cy
        from ed_ising import Hising

        H = Hising(4, 1, 0.3)
        v = cy.ones(16)
        cy.linalg.Lanczos(Hop=H, Tin=v, method="ER", max_krydim=2)
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Lanczos_ER" in result.stderr
    assert "disabled" in result.stderr
