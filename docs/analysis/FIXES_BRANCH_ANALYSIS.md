# `Cytnx-fixes` / `fixes/general` — Branch Analysis & Upstreaming Map

*Analysis date: 2026-06-24. Companion to `TECHNICAL_DEBT.md`.*
*Subject: `ianmccul/Cytnx-fixes` branch `fixes/general`, forked from `Cytnx-dev/Cytnx` at `7b63aad8`.*

**Snapshot note.** This document analyses the branch state at the date above. Later local work on `fixes/general` may add, remove, or rewrite commits; the purpose here is to map the branch against upstream Cytnx at the time of analysis, not to serve as a live commit inventory.

This document analyses the `fixes/general` branch in the context of the upstream
technical-debt scan, and maps its 20 commits onto a concrete upstreaming plan: which are
drop-in PRs today, which carry deliberate API breaks, and which are entangled with the
numerical-core rewrite (and with upstream's in-flight `tn_algo` removal).

---

## 1. What the branch is

A focused **20-commit** branch, **≈ +4,463 / −2,163 lines**, deliberately scoped to the
**floating-point numerical-algorithm layer** — Krylov/Lanczos solvers, matrix
exponentials, `LinOp` contracts, `Svd_truncate` defaults, plus diagnostics and Python
ergonomics. It explicitly **excludes** the mixed-dtype arithmetic system (see
`TECHNICAL_DEBT.md` §2/§5), documenting it as known-broken and advising users to avoid
integer/bool and in-place arithmetic.

It is positioned as a **maintained user-facing fork for real calculations**, not a PR
pipeline — consistent with the author's stated direction in upstream #933.

**Important:** the fork point `7b63aad8` is **behind** the current upstream master
(`21cf05a4`). Upstream has since merged stride_view (#860) and removed the `tn_algo`
Python bindings (#922), and is mid-removal of the C++ `tn_algo` layer (#931). The branch's
single edit to `src/tn_algo/DMRG.cpp` is **forced API-adaptation, not an intentional
`tn_algo` change** (see §4) — but it is one concrete symptom of the wider integration risk:
the branch carries a breaking `LinOp` interface change that ripples into every `LinOp`
subclass in the tree.

---

## 2. Mapping onto the debt report

| Debt item (`TECHNICAL_DEBT.md`) | Treatment here | Verdict |
|---|---|---|
| **§4 LinOp** | Central dimension validation on **both input and output**, immutable dtype/device, removed `mv_elem` + `set_dtype`/`set_device`, `matvec` → non-virtual wrapper + protected `matvec_impl` hook | Implements #894/#898; **pragmatic subset** of #899/#930 |
| **§10 correctness cluster** | Core of the branch: Lanczos_Gnd rewrite, ER disabled, Lanczos_Exp convergence fix, ExpM → scaling-and-squaring Padé, Exp dtype preservation | Strong; covers #900–#903, #902, #914 |
| **§2/§5 dtype dispatch + Scalar** | Out of scope, documented as unfixed | Honest, correct prioritisation |
| **§6 dead subsystems** | Not addressed; one **forced** `tn_algo/DMRG.cpp` edit (LinOp API fallout, §4) | Evaporates if #931 lands |
| **§1 cruft** | One "clean up test temp files" commit; `boostraps/` untouched | Marginal |

The LinOp redesign implements the **checked-wrapper / protected-virtual-hook** pattern
from #898/#930 and validates the *output* dimension too (closing the "callback output can
violate the contract" gap). It stops short of the full #899/#930 split: it remains **one
class with two overloads**, and `matvec_impl` is **virtual-but-not-pure** (the "you forgot
to override" contract is still a runtime throw, not compile-time). A measured ~70% of the
proposal, chosen for source-compatibility.

---

## 3. Code quality

Genuinely good, and better-tested than the code it replaces.

- **Krylov core** (`Lanczos_Gnd_core.hpp`): dtype-aware epsilon, a principled
  `beta_breakdown_tolerance` (roundoff × eps × scale × √dim), a 100-vector ceiling on
  non-restarted Lanczos, and explicit steering toward the ARPACK path.
- **Diagnostics** (`KrylovStats`, `last_krylov_stats()` / `krylov_stats()` / stderr
  one-liners): a capability upstream lacks entirely, and exactly the instrumentation that
  would have surfaced several open Krylov bugs earlier.
- **Tests**: new `Dtype_smoke_test`, `Exp_test`, `ExpM_test`, `LinOp_test`, large
  `Lanczos_Gnd_test` additions, and four new pytest files.

**Minor nits:**
- `capped_nonrestarted_maxiter` uses a function-local `static bool warning_issued` (not
  `thread_local`), inconsistent with the thread-local stats elsewhere — benign race,
  "warn once globally."
- `KrylovStats::add_to_total` mixes semantics: it **sums** counts (matvec, iterations) but
  **overwrites** scalar fields (`final_error`, `cvgcrit_used`) with the last call's value,
  so "cumulative" stats are part-aggregate, part-last-value.
- `DenseMatrix_internal.hpp` adds another small in-house dense-matrix abstraction — watch
  it doesn't become a third parallel matrix type.
- The `LinOp` constructor dropped the `dtype = Type.Double` default, so `LinOp("mv", n)`
  no longer compiles — a deliberate but real source break for C++ subclasses.

---

## 4. The strategic risk: drift

The branch's liability is organisational, not technical:

- forked **behind** upstream master, which has since moved on;
- per its own README, **not** being upstreamed systematically, so it accumulates rebase
  debt against a moving target.

**On the `tn_algo/DMRG.cpp` edit specifically** — it is *not* an intentional `tn_algo`
change and *not* the main risk. It is mechanical fallout of the `LinOp` redesign, two
hunks in `e61a2744`:

1. `matvec` → `matvec_impl` rename in both `Hxx_new` and `Hxx`, forced because the branch
   turned `matvec` into a non-virtual checked wrapper and moved the override to
   `matvec_impl`.
2. `LinOp("mv", 0 /*doesn't matter*/, …)` → `LinOp("mv", EffectiveDim(functArgs), …)`,
   forced because the new mandatory dimension check rejects the old bogus `nx=0`. This is
   itself a concrete instance of the #894 bug class: the DMRG operator had been
   constructed with a meaningless dimension all along, and the new validation simply
   surfaced it.

If #931 lands and `DMRG.cpp` is deleted, this hunk **evaporates** on rebase (nothing to
conflict with). So the file edit is benign.

The `matvec` → `matvec_impl` rename is breaking, but **only on the C++ side, and the blast
radius is small in practice**:

- **Python is unaffected.** The pybind trampoline (`pybind/linop_py.cpp`) uses
  `PYBIND11_OVERLOAD_NAME` to keep the Python-facing method name as `"matvec"` while the
  C++ hook becomes `matvec_impl`. Python users subclass `LinOp` and override `matvec`
  exactly as before. Everything Python-visible in this branch is either a bug-fix or a
  compatibility stub (`set_dtype`/`set_device` retained but now raise informative errors;
  the broken `set_elem`/`mv_elem` path removed; `dtype` now a required constructor arg).
- **C++ subclasses must be updated**, but the only ones that exist are *in this tree*
  (DMRG's `Hxx`/`Hxx_new`, test fixtures, examples) and are updated by the same commit.
  Cytnx's C++ API has, realistically, no external downstream users, so the "migration note
  for downstream subclasses" is close to theoretical — worth a line in the PR, not a
  blocker.

The value (correct, instrumented numerics for actual DMRG/TDVP work) is immediate; the
cost is a maintained fork accumulating rebase debt. The mitigation is to land the
**independently valuable pieces** upstream now, before further divergence widens the gap —
see §5.

---

## 5. Upstreaming map (cherry-pick assessment)

Based on the actual per-commit file footprint. Commits listed oldest→newest within tier.
"Hotspots" `include/linalg.hpp`, `pybind/linalg_py.cpp`, and `cytnx/linalg_conti.py` are
touched **additively** by many commits, so they rarely block a cherry-pick on their own.

### Tier A — drop-in standalone PRs (isolated files, additive or pure fix, no buy-in needed)

| Commit | Title | Footprint | Upstream value |
|---|---|---|---|
| `d5ef6d29` | Fix CUDA Pow dtype-changing input path | `src/linalg/Pow.cpp` only | **Cleanest PR.** Pure correctness fix |
| `589c9053` | Gate error stack traces on debug symbols | `cytnx_error.hpp` + CMake + test | Self-contained infra/UX |
| `c0fd54a3` | Expose Python Type helpers | `pybind/cytnx.cpp` + test | Purely additive; useful API |
| `9cc8322a` | Add Lq canonicalization | `Qr.cpp` + additive decls + tests | Additive feature; low controversy |
| `0fd96560` | Replace ExpM eigensolver with Padé | `ExpM.cpp` + test | High-value correctness (#902)* |

*\*`ExpM.cpp` is also touched later by `c204a82c` (smoke-test wiring) and `3e87198a`
(diagnostics); the Padé commit is the substance and rebases cleanly ahead of them.*

### Tier B — standalone but carry a deliberate API/behaviour change (need a policy nod; trivial to split)

| Commit | Title | Footprint | Note |
|---|---|---|---|
| `34aa230a` | Fix Exp dtype paths + remove Expf | `Exp.cpp`, `Expf*.cpp`, linalg.hpp, test | **Split it:** the dtype fix (#914) is uncontroversial; the `Expf`/`Expf_` *removal* is the API break |
| `d8c22bc4` | Warn on unsafe Svd_truncate cutoffs | `cytnx/linalg_conti.py` only | Python-only; changes default `err` 0→1e-8. Easy PR, needs default-policy agreement |
| `52817e55` | Simplify Python import path | `cytnx/__init__.py` only | Trivial, standalone |

### Tier C — valuable cluster, internally coupled (PR as one unit)

| Commits | Title | Why coupled |
|---|---|---|
| `00eb1105` + `d7676974` (+ `KrylovStats.cpp`) | Krylov diagnostics + stats | Woven through **every** solver (Arnoldi, Lanczos, Lanczos_Exp, Lanczos_Gnd, Lanczos_ER); depends on solver-signature changes introduced in `e61a2744`. Upstream has nothing equivalent — high value, but ships as a cluster, not commit-by-commit |

### Tier D — entangled numerical core (hardest; carries the `LinOp` interface break)

| Commit | Title | Entanglement |
|---|---|---|
| `e61a2744` | Tighten LinOp & Krylov dtype handling | **Keystone & omnibus.** Bundles the LinOp restructure/validation **with** Krylov dtype changes **and** the forced `matvec`→`matvec_impl` updates to all `LinOp` subclasses (incl. `DMRG.cpp`). Most later commits build on it |
| `bc72113d` | Make LinOp metadata immutable | `LinOp.hpp` + pybind — isolated, but semantically depends on `e61a2744`'s restructure |
| `40871a7c` | Disable Lanczos ER, harden Gnd | API break (removes `method="ER"`); touches examples + solvers |
| `3e87198a` | Rewrite Lanczos_Gnd residual solver | The big rewrite; adds `Lanczos_Gnd_core.hpp` + `DenseMatrix_internal.hpp`; depends on `e61a2744` + KrylovStats |
| `c4ac3e6a`, `442ad264`, `99d80986` | Lanczos_Exp: ExpH path, precision warning, convergence fix | Interdependent chain on `Lanczos_Exp.cpp`, rooted in `e61a2744` |

**Key nuance:** the LinOp central-validation work I rate highly in §2 is **not** cleanly
isolated — it lives inside the omnibus `e61a2744` commit alongside the Krylov dtype changes
and the forced subclass renames. To upstream "just the LinOp contract fix" (#894/#898),
`e61a2744` must first be **split** into (a) LinOp restructure + validation **plus the
`matvec`→`matvec_impl` updates to all in-tree subclasses** (DMRG, test fixtures, examples),
and (b) the Krylov dtype handling. Part (a) + `bc72113d` + the `dce9a8f9` test would then
form a self-contained LinOp PR. Note (a) spans several files: the override-hook rename
touches every in-tree C++ `LinOp` subclass (DMRG, fixtures, examples). It is **C++-only** —
Python overriders are unaffected (the trampoline preserves the `"matvec"` name) — and there
is no known external C++ consumer, so it's a wide-but-shallow diff, not a compatibility
hazard.

---

## 6. Recommended upstreaming sequence

1. **Ship Tier A as individual PRs now** — `d5ef6d29`, `589c9053`, `c0fd54a3`, `9cc8322a`,
   `0fd96560`. All are isolated, low-controversy, and several are strict correctness wins
   (#902, #914-adjacent). These reduce fork-debt immediately at near-zero negotiation cost.
2. **Split `34aa230a`** and PR the Exp dtype fix (#914) separately from the `Expf` removal.
3. **Split the keystone `e61a2744`**, then PR the LinOp half (+ `bc72113d` + `dce9a8f9`) as
   the #894/#898 fix. The C++ override-hook rename touches every in-tree `LinOp` subclass,
   so the PR should land those updates with it; Python is unaffected. It reads more cleanly
   if it goes in *before* #931 removes `DMRG.cpp`, simply so the diff matches the tree
   reviewers see.
4. **Propose the Krylov diagnostics cluster (Tier C)** as a single feature PR — it's
   additive and upstream lacks it; it does not require the solver *rewrites* to land first
   if the stats hooks are introduced as no-ops where the rewrites aren't present.
5. **Hold Tier D's solver rewrites** (ER removal, Gnd rewrite, Lanczos_Exp chain) as a
   coordinated proposal tied to issues #900–#903 — these are the most valuable but also the
   most API-breaking and the most entangled, so they warrant explicit maintainer buy-in
   rather than opportunistic cherry-picks.

*Net:* roughly **8 of 20 commits** (all of Tier A, plus the splittable halves of `34aa230a`
and `e61a2744`) can become clean upstream PRs without depending on the contentious rewrites
or on `tn_algo`. Landing those alone would substantially shrink the fork's rebase surface.
