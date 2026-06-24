# Cytnx — Maintenance Strategy and Project Direction

*A proposal for discussion. Draft, for distribution to Cytnx maintainers and stakeholders.*

## Purpose

This document proposes a realistic maintenance strategy for Cytnx given the project's actual constraints. It is written after a review of the codebase and the open issue tracker. Its central argument is uncomfortable but, I believe, correct: **the project does not have the C++ capacity to rewrite its core components, so it should stop planning as if it does, and adopt a strategy that the available people can actually execute.** That strategy is achievable and would leave Cytnx materially safer and more sustainable than it is today.

## The situation

A few facts, stated plainly so the rest follows from them.

- **The core has deep, structural design debt.** The `Scalar` type is a hand-written variant built on raw-pointer polymorphism with manual memory management; `UniTensor` is a 6,000-line polymorphic god-object that exposes the raw element-wise array API of `Tensor`; and every arithmetic operation is implemented and dispatched by hand per dtype-pair per device, producing multi-thousand-line near-duplicate files. These are not cosmetic problems; they are the source of recurring bugs.
- **There are active correctness and memory-safety bugs in that core**, not just ugliness. Mixed integer/floating-point and signed/unsigned arithmetic can silently return wrong results. At least one in-place GPU path writes a `double` result through a pointer cast to an integer buffer — i.e. genuine memory corruption. These are documented in the tracker (e.g. #937, #935, #906, #914).
- **The fixes are beyond the project's current C++ capacity.** Repairing the core properly requires confident use of templates, `std::variant`, concepts, and a typed redesign of `UniTensor`. The project does not presently have maintainers able to carry out and review work at that level. This is the binding constraint, and no amount of strategy changes it.
- **The backlog is growing, not shrinking.** Open issues have increased by roughly one net-unresolved issue per week for about three years. Many old issues are real bugs that are simply stale. The current development model is not reversing this trend.
- **The library's users are almost entirely Python users.** The C++ API has, realistically, no external downstream consumers. This is important: it widens what can be changed or removed without breaking real users.

A concrete illustration of all of the above at once: PR #937 disables the unsafe `Scalar` arithmetic and adds the first-ever tests for `Scalar`. The only response it received was an automated report that it breaks 16 existing `DenseUniTensor` tests — and no human review. Those 16 failures are `Add_/Sub_/Mul_/Div_` on `UniTensor`: the dangerous `Scalar` code is not dead, it underpins UniTensor element-wise operations that are themselves of dubious validity (#934), and there are tests pinning the broken behaviour in place. A serious safety fix, blocked by tests defending the bug, with nobody to adjudicate. That is the project's situation in miniature.

## The reframe: aim for *safe*, not *correct-by-construction*

The mistake is to keep treating "rewrite the core" as the plan. It is a wish, not a plan, because the capacity to execute it does not exist. The achievable and worthwhile goal is different:

> **A library that fails loudly instead of silently corrupting data or returning wrong answers.**

That bar is reachable with moderate C++ skill, because *guarding and disabling* a dangerous path is far easier than *redesigning* it. We do not need to fix `Scalar` to make it throw a clear error instead of corrupting memory. This is exactly the philosophy already applied to the numerical-algorithm layer in the `fixes/general` work, and it should become explicit project policy.

## Strategy

### 1. Make the core safe

Adopt a standing policy: where a core component cannot be fixed, it must fail loudly and never silently produce wrong results or corrupt memory. Treat memory-corruption paths (e.g. the GPU in-place mixed-dtype kernels) as **release-blockers**, even when the only available remedy is to throw. Merge holding-action PRs like #937 on that basis. Add tests that pin *correct* behaviour and that assert the dangerous combinations are refused.

### 2. Shrink the surface to fit the maintainers we have

The core is unmaintainable largely because it is too large and too general for the team. The highest-leverage action per unit of skill is therefore **subtraction**:

- **Restrict `UniTensor` to floating-point scalar types** (`float`, `double`, `complex<float>`, `complex<double>`), per #933. Integer and boolean tensors have no clear tensor-network meaning, are effectively untested, and are a direct source of the broken arithmetic. Removing them deletes whole classes of bugs rather than fixing them.
- **Remove the element-wise array operations that are not meaningful tensor-network operations** (#934). For symmetric, block, and fermionic tensors many of these are not merely unusual but ill-defined.
- **Remove dead and half-removed subsystems** — the unused `Gncon` contraction stack, and the C++ `tn_algo` layer already mid-removal.

This is deletion and domain-tightening work. It is within the team's reach, and a smaller, sharply-scoped library is one that a non-specialist team can realistically keep correct.

### 3. Treat `Scalar` as the one core rewrite that is actually in reach

Of all the core debt, replacing `Scalar`'s raw-pointer polymorphism with a `std::variant`-based value type (#935 / #847) is the **most tractable**: `std::variant` is well-understood, the semantics are unambiguous, and the task is well-bounded. It is a fundamentally different order of difficulty from templating `UniTensor`. It is therefore the single core refactor worth attempting even with thin C++ capacity — **but only with a competent reviewer in the loop.** Until then, #937's "disable the unsafe operations" is the correct interim state.

### 4. Recognise that capacity is the binding constraint

Everything above is harm reduction. It makes Cytnx *safe*; it does not make it *good*. The #937 episode shows the real limit: the memory-corruption bug was found by a human carefully reading the source, not produced automatically. AI assistance genuinely lowers the skill floor for deletion, test-writing, and mechanical refactoring — but it does not substitute for the judgement that finds these bugs or designs their replacement, and AI-assisted patches to the core still require a skilled reviewer.

The implication is direct: **if the project wants more than safety, the only lever is acquiring C++ capacity** — a funded student or research assistant with real C++ experience, a targeted collaboration, or a sponsored refactor. That is a resourcing decision, not a coding decision, and it should be put plainly to whoever can act on it. If that capacity is not coming, the project should say so and set expectations accordingly.

### 5. Name the successor honestly

If the "do it properly" effort is going into a successor library (uni20), with Cytnx's long-term future as a Python-compatibility layer over it, that direction should be **stated publicly**. Users and would-be contributors can then plan, and effort stops being poured into a core rewrite that has no one to land it. An honest public position — "Cytnx is stable but frozen at the core; new architectural work targets its successor; here is the migration story when it is ready" — serves users far better than the current implicit drift.

## The one decision that unblocks the most

If only one thing is decided, decide this:

> **Rule the dubious `UniTensor` element-wise array operations out of scope.**

That single call:

- lets #937 merge (the 16 failing tests defend behaviour we have decided is invalid, so they are removed rather than preserved);
- shrinks the `Scalar` blast radius, making both the interim guards and the eventual `variant` rewrite smaller;
- resolves #934 directly.

Three knots cut at once. The test failures are not an obstacle to this — they are a map of exactly which operations are affected.

## Recommended actions, in priority order

1. **Adopt the safety policy** (§1) and merge the outstanding holding-action PRs (#937), after taking the scope decision above so the broken tests can be removed rather than kept green.
2. **Take the scope decision** (§"unblock"): floating-point-only `UniTensor`, remove the ill-defined element-wise ops.
3. **Triage the backlog**: close stale/unreproducible issues; label the remainder as either "needs core rewrite" or "fixable now". Stop the count from being meaningless.
4. **Document the safe operating envelope** for users: floating-point tensors are supported; mixed-dtype and in-place integer/unsigned arithmetic are known-unsafe. (The `fixes/general` README already states this; promote it upstream.)
5. **Delete dead subsystems** (`Gncon`, finish the `tn_algo` removal).
6. **Attempt the `Scalar` → `std::variant` rewrite** (§3), only with a competent reviewer.
7. **Make the resourcing ask** (§4) and **publish the successor position** (§5).

## What this does and does not promise

This plan does not make Cytnx architecturally good, and it is honest about that. What it does is stop the library from silently producing wrong answers, shrink it to something the available team can maintain, and tell users the truth about where it is safe and where it is going. Given the constraints, that is both the responsible outcome and an achievable one. The alternative — continuing to plan around core rewrites the project cannot staff — guarantees only that the backlog keeps growing and the corruption bugs stay shipped.

---

### Appendix: issue cross-reference

- **Scalar safety / redesign:** #937 (disable unsafe ops; memory corruption), #935 / #847 (replace with `std::variant`), #906 (view-detaching in-place arithmetic).
- **UniTensor scope:** #933 (typed, floating-point-only `UniTensor`; smaller API), #934 (element-wise ops that should not exist).
- **Arithmetic correctness:** #914 (`Exp` wrong for complex64), #858 (dtype promotion via type ordering), #907 (row/column-major LAPACK mismatch).
- **Memory safety:** #852, #853 (ASan heap overflows).
- **LinOp contracts:** #894, #898, #899, #930.
- **Numerical algorithms:** #900–#903 (Krylov), #902 (robust matrix exponential).
