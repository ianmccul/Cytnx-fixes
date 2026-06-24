# Cytnx — Technical-Debt & Maintenance-Burden Scan

*Scan date: 2026-06-24. Branch: `master` @ `21cf05a4`. Upstream: `Cytnx-dev/Cytnx`.*

**Method.** Static scan of the tree (LOC, file sizes, churn, tracked-but-non-source
files, duplication) cross-referenced against the last ~4 months of merged/open PRs
and open issues on `Cytnx-dev/Cytnx`. Where a finding is already tracked, the issue/PR
number is cited so this document stays a *map* rather than a competing backlog.

The notable result of the cross-reference: **most of the deep structural debt has already
been written up in the issue tracker** (UniTensor typing, LinOp contracts, Scalar
implementation, tn_algo removal) — largely in issues and PRs filed by the author of this
scan as an outside contributor, *not* as accepted maintainer priorities. They reflect one
contributor's diagnosis; whether the maintainers act on them is a separate question. This
document consolidates those threads and adds the mechanical cruft that isn't yet ticketed.

---

## Executive summary

| # | Area | Severity | Nature | Tracked as |
|---|------|----------|--------|------------|
| 1 | Committed junk files (`boostraps/`, stray `log`/`.tmp`) | Low effort / high signal | Cruft | — (untracked) |
| 2 | Hand-maintained per-dtype × per-device dispatch | High | Maintenance burden | — |
| 3 | `UniTensor` as an untyped god-object over `Tensor` | High | Design debt + correctness | #933, #934 |
| 4 | `LinOp` dual-purpose string-typed class, no central contract | High | Design debt + latent bugs | #930, #899, #898, #894 |
| 5 | `Scalar` raw-pointer PIMPL polymorphism | High | Design debt + memory bugs | #935, #847, #937, #906 |
| 6 | Parallel/dead subsystems (`Gncon` vs `Network`, `tn_algo`) | Medium | Dead weight | #931, #924, #927 |
| 7 | Oversized headers driving churn & compile time | Medium | Maintenance burden | (part of #933) |
| 8 | Duplicated GPU test-data trees | Low | Cruft | — |
| 9 | Growing issue backlog (~+1/week for 3 yrs) | Meta | Process | #933 |
| 10 | Numerical-correctness cluster (Krylov, LAPACK layout, Exp) | High | Correctness | #900–#903, #907, #914, #902 |

---

## 1. Committed junk files (delete — zero risk)

A number of stray, non-source files are tracked in git:

- **`src/backend/boostraps/`** (note the misspelling of "bootstraps"). Holds the
  Kron/Outer codegen scripts (`gen_*_internal.py`, `cy_type.py`) **plus four captured
  stdout logs** — `log`, `logout`, `logkron`, `culog` — totalling ~220 KB. These are
  generator output left next to the generators.
- **`src/backend/linalg_internal_gpu/log`** — 967 lines of generated C++ captured to a file named `log`.
- **`src/backend/linalg_internal_gpu/bet.py`** — a 20-line throwaway that opens `cuArithmetic_internal.cu` and munges it.
- **`cytnx/log`**, **`cytnx/cxxflags.tmp`**, **`cytnx/linkflags.tmp`**, **`cytnx/vinfo.tmp`**,
  **`cytnx/Inspector.py.dev`** — build-artifact / scratch files committed into the Python package dir.

None affect runtime; all hurt navigability and signal that generator output was
hand-committed rather than produced by the build. **Recommendation:** remove from
version control.

---

## 2. Hand-maintained per-dtype × per-device dispatch (dominant burden)

Every elementwise op is implemented and dispatched **by hand for each dtype pair on each
device**. This is the single largest mechanical maintenance cost in the tree.

- `iAdd/iSub/iMul/iDiv_internal.cpp` are **2,206 lines each** and largely clones of one
  another (only ~910 of 2,206 lines differ between `iAdd` and `iSub`).
- GPU equivalents are enormous and mostly mechanical dtype-pair expansion:
  `cuSub_internal.cu` **10,918 lines**, `cuAdd` 6,462, `cuDiv` 4,606, `cuMul` 2,992,
  plus `cuMod`, `cuCpr`, `cuArithmetic`.
- The dispatch tables (`linalg_internal_interface.cpp`, `utils_internal_interface.cpp`)
  are **882 lines each** of function-pointer matrices.

Symptoms of the pattern: the identical comment
`// TODO: Investigate why _Rin is a host pointer instead of device pointer` is
copy-pasted ~15× across `cuMod`/`cuCpr`; the same `cuDet` TODO appears 4×.

There is already a partial codegen approach (the `gen_*_internal.py` scripts), but it
only covers Kron/Outer and its output was committed once and abandoned (§1).

**Recommendation:** pick one direction and commit to it — either (a) generate these
files at build time from a single template and stop tracking the output, or (b) collapse
the dtype matrix with C++ templates / `concept`-constrained functions. Issue #937 already
demonstrates the template direction (`template <CytnxType T>`) and notes the blocker:
*virtual functions can't be templates*, so this is entangled with the `Scalar`/UniTensor
virtual-dispatch redesign (§3, §5).

---

## 3. `UniTensor` is an untyped god-object layered over `Tensor` (the Tensor/UniTensor split)

**Already diagnosed in #933 and #934 — this is the headline design-debt item.**

`Tensor` is a general ndarray container; `UniTensor` is meant to be a tensor-network
object (bonds, labels, rowrank, symmetry sectors, fermionic signs). The current design
conflates them:

- **`include/UniTensor.hpp` is 6,094 lines** and the most-churned file in the repo
  (79 commits in 2 yrs). The polymorphic hierarchy (`UniTensor_base` →
  `Dense`/`Sparse`/`Block`/`BlockFermionic`) carries a very large virtual interface,
  much of it the raw element-wise `Tensor` API surfaced verbatim on `UniTensor`.
- Per #934, several of those exposed ops are **not well-defined** on symmetric/block/
  fermionic tensors and are a *correctness* bug, not just API bloat. The clearest example:
  `DenseUniTensor::Add_(const Scalar&)` adds a scalar to every stored element, while the
  block/fermionic overrides (correctly) throw — so the same API is silently wrong for one
  subtype and refused for another.
- Per #933, `UniTensor` also accepts integer/bool dtypes that have no tensor-network
  meaning and are effectively untested.

**Direction (from #933):** restrict `UniTensor` to `{float, double, complex<float>,
complex<double>}`; make it `template <UniTensorScalar Scalar> class UniTensor`; and prune
the element-wise API down to operations that are meaningful on a structured
tensor-network object. #933 also notes this overlaps with the separate `uni20`
effort and may ultimately be a Python-compatibility layer rather than an in-place rewrite
— so any large refactor here should be weighed against that.

`BlockFermionicUniTensor.cpp` (2,841 lines, 400 comment lines, 59 commits) is the
sharpest edge of this hierarchy and carries unresolved *semantic* TODOs, notably
`// TODOfermion: signflips need to be included!!!` — a fermionic-sign correctness flag,
not cleanup.

---

## 4. `LinOp`: one string-typed class doing two jobs, with no central contract

**Tracked across #930, #899, #898, #894 — a coherent four-issue thread.**

`LinOp` (`include/LinOp.hpp` 178 lines, `src/LinOp.cpp` 68 lines) branches at runtime on
a `_type` string:

- `"mv"` — a thin metadata holder whose `matvec(Tensor)` / `matvec(UniTensor)` are
  `virtual` but **not pure**; the base versions exist only to throw "required overload…".
  The contract is enforced at first call, not compile time.
- `"mv_elem"` — a precomputed sparse matrix; `matvec(UniTensor)` always throws here.
  Only exercised by tests/docs/benchmarks, never by production (`src/tn_algo`).

Consequences:
- **No central validation.** `nx()` is checked once against the initial vector in each
  solver, never re-validated against the tensor a subclass's `matvec` hands back inside
  the loop (#894). The dtype check in `Lanczos.cpp` is *ineffective* — it casts to
  `Hop->dtype()` and then asserts the dtype matches (#898).
- **Two unrelated interfaces fused into one** — most operators implement only one of the
  two `matvec` overloads (#899).

**Direction:** split `"mv"`/`"mv_elem"` into distinct classes; make the multiply hook
pure-virtual; move validation into a non-virtual checked `matvec` wrapper on the base; and
clean abbreviation-heavy public names (`nx`, `mv`) with a deprecation path (the class is
exposed in both C++ and Python). #930 contains a concrete proposed `LinOpBase` design.

---

## 5. `Scalar`: hand-rolled `std::variant` with manual memory management

**Tracked across #935, #847, #937, #906, #916.**

`cytnx::Scalar` is a raw owning pointer to a runtime-polymorphic `Scalar_base` hierarchy
(`include/backend/Scalar.hpp`, **3,472 lines**), with deep-copy copy/assign, manual
`delete` in the destructor, a dtype-indexed factory table, and duplicated conversion
methods per dtype. It is, as #935 puts it, "effectively a hand-written `std::variant`, but
with heap allocation, virtual dispatch … and manual memory management."

This isn't only architectural: #937 found that several `Scalar` arithmetic combinations
(mixed signed/unsigned, mixed int/float, in-place `/=`) **silently return wrong results**,
with "a bit of memory corruption," and that **`Scalar` had no tests at all** before that
PR. PR #937 disables the unsafe ops with a throwing error as a stopgap; #935/#847 propose
the real fix (replace the PIMPL hierarchy with `std::variant`). #906 notes a related view-
detaching bug in `Scalar` in-place Tensor arithmetic.

Because the per-dtype overloads here are `virtual`, they can't be replaced by the
`concept`-constrained templates #937 wants to use — so **§2, §3, and §5 share a root cause**:
runtime virtual dispatch over dtype, where compile-time templates would eliminate the
boilerplate.

---

## 6. Parallel & dead subsystems

- **`Gncon` vs `Network`.** Two full contraction-network stacks exist:
  `Network*.cpp` / `RegularNetwork.cpp` / `FermionNetwork.cpp` **and**
  `Gncon*.cpp` / `RegularGncon.cpp`. `Gncon` is exported via `include/cytnx.hpp` but is
  **referenced by no pybind binding and no test** — an unfinished/abandoned rewrite that
  is pure dead weight until retired or marked WIP. `RegularGncon.cpp` also carries 221
  commented-out lines.
- **`tn_algo` (C++ MPS/MPO/DMRG)** is mid-removal: PR #922 (merged) dropped its Python
  bindings as "untested … non-functional"; PR #931 (open) removes the C++ layer; #924/#927
  add tests and fix the bugs found. Finish or fully revert this — a half-removed subsystem
  is worse than either end state.

---

## 7. Oversized headers concentrate churn and compile cost

- `include/UniTensor.hpp` — 6,094 lines (see §3).
- `include/backend/Scalar.hpp` — 3,472 lines (see §5).
- `include/linalg.hpp` — 3,361 lines; `include/Tensor.hpp` — 1,738 lines.

These heavy headers dominate incremental compile times and make the most-churned files
merge-conflict-prone. Splitting `UniTensor.hpp` along the subtype boundary (or behind the
templated redesign of §3) is the highest-leverage compile-time win.

---

## 8. Duplicated GPU test-data trees

`tests/test_data_base/` (132 files) and `tests/gpu/test_data_base/` (114 files) are
largely byte-identical copies of the same `.cytnx` binary fixtures (only 17 paths differ).
The GPU suite should reference the CPU fixtures rather than maintaining a divergent copy.
(Related: #851 flags the GPU elementwise tests as dominating ctest runtime.)

---

## 9. Process-level signal: the issue backlog

Per #933, open issues have grown by roughly **one net-unresolved issue per week for ~3
years**, many representing real, still-present bugs whose original reports went stale. The
recent burst of well-scoped refactor issues (#894–#939) is the right response, but the
trend says incremental triage alone won't reverse it — hence the #933 proposal to shrink
the API surface to the core tensor-network operations. Worth treating "reduce API surface"
as an explicit, measurable goal rather than a by-product.

---

## 10. Open correctness cluster (not cruft, but the real risk)

Several confirmed correctness bugs are open and worth grouping because they share roots in
the numerical/linalg layer:

- **Krylov solvers** — #900 (audit non-ARPACK Krylov for numerical bugs), #901, #903; the
  TDVP/`Lanczos_Exp` series (#887–#892) shows how fragile these paths are. *(Matches the
  standing note: prefer the ARPACK paths; `method="ER"/"Gnd"` are unreliable.)*
- **Matrix exponential** — #902 (use a robust algorithm for `ExpM`/`ExpH`), #914
  (`linalg.Exp` wrong for complex64).
- **LAPACK layout** — #907 (row-major Tensor storage fed to mostly column-major LAPACKE).
- **Memory safety** — #852, #853 (ASan heap-buffer-overflows in `_Load` and
  `OptimalTreeSolver::solve`).
- **Invariants** — #846 (in-place `Bond` mutation breaks the block-tensor symmetry
  invariant; make `Bond` immutable), #864 (`reshape` always copies), #858 (dtype promotion
  via type ordering).
- **Test hygiene** — #857 (GoogleTest underscore naming → ~39 reserved-identifier UB cases).

---

## Suggested priority order

1. **Delete tracked junk (§1)** and finish the **`tn_algo` removal / `Gncon` decision (§6)**
   — pure subtraction, immediate clarity, no design risk.
2. **Land the `LinOp` refactor (§4)** — small, self-contained, closes four issues and
   removes a class of latent solver bugs.
3. **Replace `Scalar` with `std::variant` (§5)** — unblocks the template direction and
   removes a memory-safety hazard.
4. **De-duplicate the dtype dispatch (§2)** via codegen-at-build or templates — highest
   long-term leverage; depends on (3).
5. **Tackle the `UniTensor` typing/API-surface reduction (§3, §7)** — largest and most
   strategic; weigh in-place refactor vs. the `uni20`/Python-compat path per #933.
6. **Burn down the correctness cluster (§10)** continuously alongside the above.

*Root-cause note:* §2, §3, and §5 are the same underlying problem — runtime virtual
dispatch over dtype. A single decision to move dtype to compile-time templates would
shrink all three.
