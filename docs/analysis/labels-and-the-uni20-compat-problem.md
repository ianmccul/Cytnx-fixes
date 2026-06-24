# UniTensor labels and why a Cytnx → uni20 Python compatibility shim is hard

*Analysis note. Companion to `cytnx-maintenance-strategy.md`.*

## Purpose

The maintenance-strategy note assumed that, whatever happens to Cytnx's core, a Python-compatibility layer over a principled successor (uni20) remains a plausible long-term path — letting existing Cytnx Python code run with little or no change. This note argues that the `UniTensor` **label / relabel mechanism** is a serious obstacle to that plan, and explains precisely why. It is the one part of Cytnx where the debt is not in the *implementation* but in the *intended semantics*, which is what makes it resistant to the "shrink, restrict, define a safe envelope" approach that works elsewhere.

## TL;DR

A compatibility shim works when the **surface API** differs but the **semantics** are reproducible on the new backend. Cytnx labels fail this on three axes at once: leg identity is modelled as a mutable, uniqueness-constrained string namespace; `relabel()` shares storage but forks metadata (a *partial* view); and label propagation through ordinary operations is ad-hoc, undefined, and in places buggy. A drop-in "scripts run unchanged" shim is therefore not feasible without making uni20 adopt Cytnx's incoherent identity-and-aliasing model. What is feasible is a **porting/translation layer that requires edits at every label-dependent site** — which is an honest offering, but is a migration tool, not a compatibility shim.

## Background: in Cytnx, labels *are* the leg identity

Labels are not decoration. Contraction matches legs by label — this is the basis of `Contract`, `ncon`, and `Network` (see #310 "Network output label inheritance", #337 "index order after Contract"). The label string is the mathematical identity that decides which legs join. That single design choice is the root of everything below.

## What the design paper says

The Cytnx design paper (Wu et al., "The Cytnx Library for Tensor Networks", SciPost Phys. Codebases, 2023) documents the label model. Read against the operations Cytnx later grew, it exposes five structural problems:

1. **Labels are a separate component, not a property of `Bond`.** §5 / Fig. 5 defines a `UniTensor` as three parts — `Block(s)`, `Bond(s)`, **and** `Label(s)`. §5.3: "two important attributes are defined for each index: its position and label." So a leg's identity is split across its position, a `Bond` (which carries the physics — dimension, direction, quantum numbers), and a free-floating label string that the `Bond` knows nothing about. The code agrees: `_labels` is a standalone `std::vector<std::string>` on `UniTensor_base`, parallel to the bonds, not a field inside `Bond`.

2. **The stated design goal is "labels let you ignore index order."** §1.1: *"Once meaningful index labels are set, the ordering and permutation of the indices in tensors can be handled by the library automatically."* §5.3: *"the same labels can still be used to find the correct index even after a permutation. Therefore, all methods that use index labels do not need to be changed after a permutation."* This is the founding premise, and it conflicts with later parts of the library.

3. **The paper itself documents the order/identity contradiction as a manual requirement.** §9.1: before any decomposition the user must *"1) permute the indices such that all the row indices are in front of all the column indices, and 2) set rowrank."* So SVD / Eig / QR depend on **position + rowrank**, never on labels. §7 also warns, of `ncon`, that *"the index order matters ... This can be error prone."* The library knows order matters for linear algebra; it just places that responsibility on the user rather than recognising that it now has two incompatible notions of leg identity. This is exactly `A['l','s','r']` ≡ `A['r','s','l']` to `Contract` but ≠ to `Svd`.

4. **The #920 crash follows the documented idiom.** §9.2 / Listing 60: SVD output legs get auto-generated labels (*"U inherits the row labels ... the other labels are automatically generated"*), then are manually `relabels_`'d to `"_aux_L"` / `"_aux_R"`. That auto-name-then-relabel pattern is precisely what collides across successive SVDs and aborts MPS construction (#920).

5. **Braiding is invisible to the design by construction.** The paper does not describe braiding or fermionic signs; its first principle ("order is ignorable") conflicts with the thing fermions require — that permuting legs can carry a sign. `BlockFermionicUniTensor` was added later onto a model whose label semantics do not encode the information needed to make this reliable.

## Three structural facts that form the trap

**1. Leg identity is a mutable, uniqueness-constrained string namespace.**
`set_label` hard-errors if the new label collides with any existing one (`include/UniTensor.hpp:193`, "already has a label that is the same as the input label"). So the contraction key is a mutable set of strings that must remain unique, and any operation that produces a collision is fatal. This is not merely a user hazard: Cytnx's own MPS code triggers it. In #920, `RegularMPS::Into_Lortho` runs an SVD that reuses the bond label `"_aux_L"`, producing duplicate labels and aborting construction — "the MPS class is effectively unusable from Python for the regular case." The model is not even internally self-consistent.

**2. `relabel()` shares data but forks metadata — a partial view.**
`DenseUniTensor::relabel` does `clone_meta()` and then `out_raw->_block = this->_block` (`src/DenseUniTensor.cpp:210-217`). So `B = A.relabel(...)` returns a handle that **shares A's underlying storage but carries an independent label vector**. Mutating B's *data* mutates A; relabeling B leaves A untouched. This is the NumPy view/copy confusion, made worse: data and metadata have *different* sharing semantics inside the same object, so there is no clean "is this a view?" answer. (The in-place `relabel_` mutates labels on the existing object; the value-returning `relabel` is the partial-view form.)

**3. Label propagation through operations is ad-hoc and partly buggy.**
The contraction key is silently dropped or reset by common operations:

- `A = B + C` resets A's labels to `['0','1',...]` regardless of B and C (#753).
- Elementwise arithmetic does not copy labels (#675, still open).
- Slicing does not inherit labels (#408).
- Network output label inheritance is undefined (#310).

So the one piece of state that determines contraction is treated inconsistently by the operations that produce new tensors. There is no coherent "label algebra"; there is a pile of per-operation behaviours, some of which are acknowledged bugs.

## Why this defeats a compatibility shim specifically

- **Identity-model mismatch.** Any principled successor models leg identity as something stable — typed bonds, positional indices, or explicit index handles — not "mutable strings matched by value at contraction time." The shim therefore cannot pass labels straight through; it must emulate a string→leg map and reimplement Cytnx's match-by-label contraction rule on top of uni20's real identity. That is not a thin adapter; it is re-implementing the naming semantics of contraction.
- **The partial-view aliasing cannot be expressed over value semantics.** If uni20 is value-semantic (or copy-on-write done correctly), Cytnx's "share storage, fork labels" can only be reproduced by reintroducing reference-counted shared storage with detachable metadata — i.e. rebuilding the mechanism uni20 exists to avoid — or by silently breaking every program that relies on `relabel()` returning a data-sharing alias.
- **There is no well-defined behaviour to be faithful to.** Because propagation is ad-hoc and buggy (#753, #675, #408, #310), "compatible" is not a coherent target. Bug-for-bug fidelity means porting the inconsistencies; correctness means diverging and breaking the very code the shim exists to support. You cannot shim semantics that were never specified — only accidents.

## A secondary problem: the API is also a redundancy swamp

Even setting the semantics aside, the spelling is a mess, and has been for the project's whole history: `set_label`, `set_labels` (deprecated), `relabel`, `relabels` (deprecated), `relabel_`, `relabels_`, `change_label`, each with `char*`/`std::string`/`int` overloads. The open issues to rationalise this span years — #33, #184, #228, #232, #286 (closed, historical churn), and #481 (remove `relabels`), #421 (merge `relabel`/`relabels`), #335 (underscore consistency), #465 (`labels=` → `label=`) still open. This is the *easy* part — it is ordinary API-surface reduction and could be done with the project's current capacity. It is worth separating from the semantic problem above, which cannot.

## The through-line

This is exactly the point made in #933: "labels should not be treated as the canonical mathematical identity of a leg ... Cytnx labels are too easy to relabel accidentally." Everywhere else in Cytnx the debt is implementation quality — the intended semantics are fine and one can shrink toward them. The label system is the exception: the **intended semantics are themselves the problem**, and they are woven into both contraction and defensive user code (scripts are full of relabels precisely because labels are fragile). That defensive code is the most quirk-coupled code in any Cytnx program, and it is exactly what a shim would most need to reproduce faithfully.

## Conclusion

A drop-in "your scripts run unchanged" shim is not feasible — not for lack of effort, but because it would require uni20 to adopt Cytnx's incoherent identity-and-aliasing model, defeating uni20's purpose. A **translation layer that requires edits at every label-dependent site** is feasible and honest: contraction-by-label rewritten to uni20's index model, `relabel()`-aliasing made explicit, arithmetic/slice label-propagation pinned down. That is a porting tool, not a compatibility shim, and any migration story should say so plainly.

Put bluntly: the dtype/Scalar debt means Cytnx sometimes computes the wrong number — repairable in principle. The label debt means **Cytnx and uni20 do not agree on what a tensor leg is**, and a `pip install` cannot paper over an ontology difference.

## Turning it around: what uni20 must decide for a *porting* layer to be writable

The constructive version of this note is a list of semantic decisions uni20 has to settle before even a translation layer can be specified:

1. **Leg identity.** Positional, typed-bond, or explicit index-handle? Whatever it is, define the deterministic mapping from a Cytnx label string to a uni20 leg.
2. **Contraction matching.** Does uni20 ever match by name, or always by explicit index/handle? The porting layer's contraction rewrite depends entirely on this.
3. **Aliasing.** What are uni20's view/copy rules, and how does a ported `relabel()` express "I want a renamed handle" vs "I want a renamed copy"? Cytnx conflates these; the port must split them.
4. **Label propagation.** Define, once, what happens to leg identity under `+`, `-`, elementwise ops, slicing, SVD/QR output bonds, and network contraction. Cytnx never did; uni20 must, because the port has to translate each case.
5. **Uniqueness and auto-naming.** Decide whether duplicate leg names are an error, allowed, or impossible by construction — and how internal/auto-generated bond names (the `"_aux_L"` class of bug, #920) are made collision-free.

If those five are settled in uni20's own terms, a Cytnx-label porting layer becomes a well-defined (if laborious) translation. Until they are, "Python compatibility" is not specifiable.

## Caveat

This rests on an inference about uni20's leg-identity model, which I have not seen. If uni20 deliberately retains a string-label notion of identity, axis one of the shim problem softens — but the partial-view aliasing (fact 2) and the undefined propagation (fact 3) remain obstacles regardless.

---

### Appendix: references

**Code:** `include/UniTensor.hpp:193` (uniqueness error in `set_label`); `src/DenseUniTensor.cpp:210-217` (`relabel` shares `_block`, forks labels); the `relabel`/`relabels`/`set_label*` overload cluster in `include/UniTensor.hpp` (lines ~156–214, ~2900–2995); `_labels` is a `std::vector<std::string>` on `UniTensor_base` (separate from bonds). `include/Bond.hpp`: `Bond_impl` holds `_dim`, `_type` (`BD_REG`/`BD_KET`/`BD_BRA`), `_qnums`, `_degs`, `_syms`, plus `get_fermion_parity()`.

**Design paper:** Wu et al., *The Cytnx Library for Tensor Networks*, SciPost Phys. Codebases (2023): §1.1 (labels make ordering automatic), §4 (Bond = dim + direction + qnums), §5/Fig. 5 (UniTensor = Block/Bond/Label), §5.3 (position+label; labels survive permute), §7 (ncon order matters), §9.1 (permute + set rowrank before decomposition), §9.2/Listing 60 (auto-generated `_aux` labels).

**Issues — semantics:** #753 (arithmetic resets labels), #675 (copy labels in elementwise arithmetic, open), #408 (slicing doesn't inherit labels), #310 (network output label inheritance), #920 (duplicate `_aux_L` crashes MPS), #337 (index order after Contract), #933 (labels are not canonical leg identity).

**Issues — API redundancy:** #481, #421, #335, #465 (open); #33, #184, #228, #232, #286 (closed, historical).
