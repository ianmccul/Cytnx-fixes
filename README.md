# Cytnx-fixes

This is a standalone repository used for local Cytnx fixes, experiments, and notes. It is not the upstream Cytnx project repository.

The default `main` branch intentionally contains only this README and, later, lightweight documentation. It is not a Cytnx source branch.

The `master` branch is kept as a clean Cytnx source branch aligned with `Cytnx-dev/Cytnx` master at commit `7b63aad8`, the fork point used by the fixes work.

The branch `fixes/general` contains the main material on this repository. This branch makes many correctness, usability and numerical stability fixes for Cytnx-based Python applications. This includes rewritten and more robust Krylov/Lanczos algorithms, stricter `LinOp` dimension checks, fixed dtype-changing elementwise exponentials, safer Python `Svd_truncate()` defaults, added `Lq()` function so that regauging MPS does not need to go via SVD, and diagnostics that show what the linear algebra functions are doing and convergence information. Some functions that were buggy, difficult to use, or numerically unsound have been removed. Existing scripts may need small updates, but failures should now be explicit instead of silently producing unreliable results.

See [Cytnx-fixes issue #1](https://github.com/ianmccul/Cytnx-fixes/issues/1) for a survey of upstream Cytnx issues that this branch either fixes directly, probably fixes, or makes easier to diagnose.

## Known unresolved arithmetic problems

This repository does not yet fix Cytnx's general mixed-dtype arithmetic system. In particular, mixed integer/floating-point, signed/unsigned, bool, and mixed precision real/complex arithmetic can still produce incorrect results. Some in-place arithmetic paths are especially dangerous because they preserve the destination dtype, so operations such as integer tensor divided by double tensor can silently truncate the result. Some GPU paths are worse and appear capable of writing through pointers cast to the wrong output type.

The `fixes/general` branch should therefore be used for floating-point tensor-network calculations, especially the Krylov, TDVP, DMRG, and SVD-related fixes described below. It should not be treated as a general validation of Cytnx integer, bool, or mixed-dtype arithmetic. Until this is fixed, avoid integer/bool tensor arithmetic and avoid mixed-dtype in-place arithmetic in user code.

Branch `fixes/general` adds:

* `34aa230a` Fix Exp dtype-changing paths and remove Expf
* `0fd96560` Replace ExpM eigensolver path with Pade exponential
* `c204a82c` Add linalg dtype smoke tests
* `c4ac3e6a` Use ExpH for Lanczos_Exp projected exponentials
* `c0fd54a3` Expose Python Type helpers
* `e61a2744` Tighten LinOp and Krylov dtype handling
* `d5ef6d29` Fix CUDA Pow dtype-changing input path
* `40871a7c` Disable Lanczos ER and harden Gnd
* `589c9053` Gate cytnx error stack traces on debug symbols
* `442ad264` Warn on higher precision input for Lanczos_Exp
* `00eb1105` Track and print Krylov diagnostics
* `27b933bb` Clean up test temp files and local coverage
* `d9fba69e` Rewrite Lanczos_Gnd residual solver
* `0272ee4f` Add Lq canonicalization support
* `427a44a6` Warn on unsafe Python Svd_truncate cutoffs
* `ef8189e4` Simplify Python package import path
* `d3a4a93c` Extend Krylov diagnostics to Lanczos_Exp
* `bf1e95f4` Fix Lanczos_Exp convergence estimate
* `b08bc731` Test LinOp matvec dimension checks
* `c39dde5a` Make LinOp dtype and device metadata immutable

Visible changes for Python users:

* `cytnx.linalg.Exp()` now preserves the correct dtype for Float/Double and ComplexFloat/ComplexDouble parameters. Integer inputs are promoted to Double.
* `cytnx.linalg.Expf()` and `cytnx.linalg.Expf_()` have been removed. Use `cytnx.linalg.Exp()` / `cytnx.linalg.Exp_()` instead.
* `cytnx.linalg.Lanczos_Exp()` uses `cytnx.linalg.ExpH()` for Hermitian projected Krylov exponentials, and preserves precision. Real Hermitian operators with real prefactors remain real and are no longer converted to complex.
* `cytnx.linalg.Lanczos_Exp()` uses more reliable error estimate instead of the old projected-matrix last-component heuristic. The default tolerance is now `1e-8`, and requests below the useful dtype-dependent numerical scale are raised with a warning.
* `cytnx.linalg.ExpM()` uses a scaling-and-squaring Pade implementation instead of diagonalizing the matrix and inverting the eigenvector matrix. This is substantially more robust for non-normal matrices. `ExpM()` for Float and Double matrices with real scale coefficients no longer converts to ComplexFloat/ComplexDouble.
* `cytnx.linalg.Lq()` is available for dense Tensor and dense UniTensor inputs. Together with `cytnx.linalg.Qr()`, this gives a QR/LQ canonicalization path for tensor-network code that does not need singular values.
* Python `cytnx.linalg.Svd_truncate()` now treats missing or zero cutoffs as potentially dangerous:
  * omitting `err` warns once per Python call site and uses `err=1e-8`;
  * explicit `err=0` raises an error because keeping null singular vectors is ambiguous and unsafe;
  * explicit `err>0` uses the requested singular-value cutoff;
  * explicit `err<0` can be used for code that needs to deliberately keep all singular vectors, including numerical zeros.
* The `Svd_truncate()` warning points users toward `Qr()` / `Lq()` for canonicalization when singular values are not needed, and toward `Svd()` / `Gesvd()` when singular values are needed without truncation.
* Python dtype helper functions. These are useful in Python code that needs to construct dtype-stable tensors, choose a `LinOp` dtype, or avoid hard-coded dtype tables:
  * `cytnx.Type.is_complex(dtype)`
  * `cytnx.Type.is_floating(dtype)`
  * `cytnx.Type.is_real(dtype)`
  * `cytnx.Type.is_integer(dtype)`
  * `cytnx.Type.is_signed(dtype)`
  * `cytnx.Type.is_unsigned(dtype)`
  * `cytnx.Type.as_complex(dtype)`
  * `cytnx.Type.as_real(dtype)`
  * `cytnx.Type.as_single_prec(dtype)`
  * `cytnx.Type.as_double_prec(dtype)`
  * `cytnx.Type.as_signed(dtype)`
  * `cytnx.Type.as_unsigned(dtype)`
  * `cytnx.Type.type_promote(dtype_a, dtype_b)`
  * `cytnx.Type.typeSize(dtype)`
  * `cytnx.Type.getname(dtype)`
  * `cytnx.Type.enum_name(dtype)`
* Krylov routines have stricter `LinOp` size checking. Calls through Cytnx linalg routines check that `LinOp.matvec()` input and output vectors match the declared `nx()` dimension. Python code that set `nx()` incorrectly will now cause an error. This can be fixed by setting the `nx` parameter correctly when constructing the `LinOp`.
* It is no longer necessary to set the dtype of a real-valued `LinOp` to `ComplexDouble` or `ComplexFloat` just because the input vector might be complex. This particularly affects TDVP code where the Hamiltonian is real but the timestep might be real or complex. Previous Cytnx would require the `LinOp` dtype of a real Hamiltonian to be complex in order to use it with a complex timestep. This is no longer the case. The dtype is instead treated as a type promotion hint.
* `LinOp` dtype and device metadata are fixed at construction time. The C++ `set_dtype()` and `set_device()` mutators have been removed. The Python methods remain as compatibility stubs, but now raise explicit errors explaining that changing metadata would not change the actual matrix-free `matvec()` implementation.
* `cytnx.linalg.Lanczos_Exp()` now warns if the input tensor has higher precision than the `LinOp` dtype hint, for example a Double input tensor with a Float `LinOp`. This usually means the Krylov basis will use the promoted dtype, but the operator action may still only carry single-precision information.
* The old `cytnx.linalg.Lanczos(..., method="ER")` function has been removed. An attempt to call `method=ER` raises an error explaining what to use instead. For ordinary Hermitian eigenvalue problems, use the ARPACK-backed interface such as `cytnx.linalg.Lanczos(Hop, Tin, which="SA")` for the smallest algebraic eigenvalue. If you *really* want a Lanczos function with explicit restarts, then call `cytnx.linalg.Lanczos(..., method="Gnd")` in a loop. This will work better than the old `"ER"` method, but not as well as ARPACK.
* `cytnx.linalg.Lanczos(..., method="Gnd")` has been reworked to be a simple but numerically stable Lanczos solver. It is intended for iterative algorithms where convergence is achieved through multiple passes of environment updates and the number of eigensolver iterations in each pass is kept deliberately low. It is expected and normal with this mode that during the early stages of the calculation the convergence criteria might not be met before `Maxiter` is reached. The default `Maxiter` is now `20`; requests that would build more than `100` Krylov vectors are capped and warned, since a non-restarted Lanczos will be numerically unstable for a large number of iterations. For DMRG applications, 4 or 5 iterations is probably appropriate.
* Krylov routines record basic convergence and work statistics. Use `cytnx.linalg.last_krylov_stats()` to inspect the most recent Krylov call, `cytnx.linalg.krylov_stats()` to inspect cumulative stats for the current thread, and `cytnx.linalg.clear_krylov_stats()` to reset them. The reported fields include the algorithm path, convergence flag, reason, matvec count, iteration count, effective Krylov dimension, requested/used tolerance and iteration limits, final residual/error data where available, and input/operator/working dtypes.
* Python calls to `cytnx.linalg.Lanczos()` and `cytnx.linalg.Lanczos_Exp()` now print a one-line diagnostic summary to stderr by default. This reports the algorithm path, `LinOp.nx()`, dtype, matvec count, Krylov dimension, stopping reason, and final residual/error data where available. It also warns when a very large absolute energy-difference threshold causes the solver to stop before `Maxiter` is used. Disable this with `cytnx.linalg.set_krylov_diagnostics(False)` or by setting `CYTNX_KRYLOV_DIAGNOSTICS=0`.
* Krylov diagnostics now warn when a complex `LinOp` dtype hint promotes a real input vector to complex working dtype. If the operator is actually real, construct the `LinOp` with a real dtype hint to avoid unnecessary complex arithmetic.
* Error messages no longer print raw stack traces by default in builds without debug symbols. Set `CYTNX_SHOW_STACKTRACE=1` to force the previous raw backtrace behavior, or `CYTNX_SHOW_STACKTRACE=0` to suppress stack traces explicitly.
* The Python package imports NumPy before loading the Cytnx extension module. This avoids BLAS/LAPACK symbol load-order crashes in environments where importing Cytnx first breaks NumPy's own runtime checks.
* The dead Python torch-backend import branch has been removed. This branch uses the normal Cytnx backend only.

# Getting existing code working with `fixes/general`

The easiest way to migrate existing Cytnx-based scripts is to install this branch and run the code. In many cases it will run immediately, but with warnings that point to code paths worth improving. If the branch detects behavior that was probably invalid before, it should fail with a more useful error message rather than silently continuing.

Start with an editable install from a checkout:

```bash
git clone git@github.com:ianmccul/Cytnx-fixes.git
cd Cytnx-fixes
git switch fixes/general
python -m pip install -v -e .
```

Then run the existing calculation. The most common follow-up fixes are:

1. If a `LinOp` dimension error appears, set `nx` to the flattened dimension of the vector space acted on by `matvec()`. Do not use `nx=0`; this now fails because it was never a valid Krylov operator dimension.

2. If diagnostics warn that a complex `LinOp` dtype is promoting real vectors to complex, use a real dtype hint for real operators. For example, a real Hamiltonian should normally use `dtype=cytnx.Type.Double`, not `ComplexDouble`, even if some later timestep or state is complex.

3. If an old Gnd call fails because it used `CvgCrit=...`, replace it with `residual_tol=...`. If unsure, start with:

   ```python
   cytnx.linalg.Lanczos(Hop, Tin, method="Gnd", residual_tol=1e-14, Maxiter=20)
   ```
   For DMRG-type solvers, it is generally better to use fewer iterations (perhaps 4 or 5), but make sure that there are enough sweeps to converge.

4. If `method="ER"` fails, replace it with an ARPACK selector for standalone eigenvalue problems, usually:

   ```python
   cytnx.linalg.Lanczos(Hop, Tin, which="SA")
   ```

   For local DMRG-style solves where a fixed small number of matvecs is intended, use `method="Gnd"` with a sensible `Maxiter`.

5. If `Svd_truncate()` warns about a missing cutoff, inspect that call site. If it is being used only to canonicalize an MPS and singular values are not needed, use `Qr()` or `Lq()` instead. If it is really truncating a bond, pass an explicit positive cutoff such as `err=1e-8`. If you deliberately want to keep numerical null vectors, pass `err=-1`.

6. Read the Krylov diagnostic lines printed to stderr. They show `nx`, dtype, matvec count, Krylov dimension, stopping reason, and residual/error information. You may prefer to incorporate some or all of this information into your own diagnostic output instead. The same information is available through:

   ```python
   cytnx.linalg.last_krylov_stats()
   cytnx.linalg.krylov_stats()
   ```

   Once the code has been checked, the automatic stderr diagnostics can be disabled with:

   ```python
   cytnx.linalg.set_krylov_diagnostics(False)
   ```

## Example workflow

Some of the Cytnx example scripts contain serious flaws that silently produce incorrect or poorly converged results. These have not been updated, since they can be used to demonstrate the improved robustness of the `fixes/general` branch.

### `example/TDVP/tdvp1_dense.py`

In Cytnx, this example program has very poor numerical properties. If you run this example using the `fixes/general` branch it will run and produce good results. It will however show some warnings about changed behavior and some suggestions to improve the code. The first thing that you will notice is that it produces a lot more output. By default, `fixes/general` will display diagnostic output from linear algebra functions. The first new line should look something like:
```text
[cytnx] Lanczos_Exp: err=1.09339e-16/1e-08, matvecs=4, k=4, nx=4, dtype=Double (Float64), stop=eigenvector, maxiter=4/100
```
This is from the `Lanczos_Exp` function for calculating the matrix exponential. This shows that the obtained error measure was `1e-16` (numerically exact) from a requested error tolerance of `1e-8`. `matvecs=4` shows the number of matrix-vector muliply operations that were performed by the `LinOp`. `k=4` is internal information, indicating that the `Lanczos_Exp` function built a Krylov subspace of dimension 4. Normally this will be equal to the number of `matvec` operations. `nx=4` is the `LinOp::nx()` parameter, of the flattened size of the domain and codomain of the operator. The `dtype` is the input tensor type `Double`. `stop=eigenvector` shows the stopping criteria of `Lanczos_Exp`; `stop=eigenvector` indicates that it finished because it found a numerically exact eigenvector (which aligns with the final tolerance `1e-16` being smaller than numerical precision). The final term is the `maxiter` parameter, showing that the function performed 4 out of a possible maximum of 100 iterations.

The next new line is a warning:
```text
[cytnx] WARNING: LinOp dtype hint Complex Double (Complex Float64) promoted real input dtype Double (Float64) to complex working dtype Complex Double (Complex Float64). If this LinOp represents a real operator, construct it with a real dtype hint to avoid unnecessary complex arithmetic.
```
This warning means that the `dtype` parameter specified by the `LinOp` is too strict, and resulted in a conversion of the tensor from `Double` to `Complex Double`. In the `fixes/general` branch, `LinOp` operators that are purely real are able to advertise a `dtype` of `Double`, even if they are used in real-time evolution or other applications where the input tensors are complex. Cytnx requires the `LinOp` in these cases to have `dtype` of `Complex Double`, which has the side-effect that if you use the same operator for *imaginary time* evolution, where all of the prefactors are real valued, the `LinOp` will convert what would have been a purely real calculation to using complex arithmetic. However this is just a warning, and we can fix that later.

The next line is another warning, and a much longer one:
```text
git/cytnx-fixes/example/TDVP/tdvp1_dense.py:173: RuntimeWarning: cytnx.linalg.Svd_truncate was called without an explicit err cutoff.

This fixes branch changes the Python default to:

    err=1e-8
...
```
This warning is quite long, but you should read it. In short, this warning occurs when you call `Svd_truncate(t, maxstates)` with no `err=` parameter. In Cytnx, this would default to `err=0`, which is numerically unsound for TDVP and a major cause for the numerical instability of this example code. On the `fixes/general` branch the default has changed to `1e-8`, which is a good general-purpose value that is fine here, and gives good results for TDVP as well as DMRG, TEBD, and other algorithms. If you want to silence this warning, there are three ways to do it:

- Change the `Svd_truncate()` call to set the `err=` value.
- Call the function `cytnx.linalg.set_svd_truncate_warnings(False)` in your Python script. This silences the warning for the duration of the script.
- Set the environment variable `CYTNX_SVD_TRUNCATE_WARNINGS=0`

The warning also contains a suggestion that `Svd_truncate()` might not the best function to use, depending on the purpose. You might be better to use a function such as `Svd`, `Qr`, `Lq`, or `Gesvd`.

This warning is only shown *once* for each time `Svd_truncate` appears in the script, so we will not see this warning about `tdvp1_dense.py:173` for the rest of the calculation.

The calculation now continues, continuing to print a diagnostic line for each call to `Lanczos_Exp`. The next line is notable:
```text
[cytnx] Lanczos_Exp: err=4.58211e-19/1e-08, matvecs=4, k=4, nx=4, dtype=Complex Double (Complex Float64), stop=eigenvector, maxiter=4/100
```
This is fairly similar to the previous diagnostic line, except we can see now the `dtype` is `Complex Double`. This came from the `LinOp` at the previous step promoting the tensor from real to complex arithmetic, and the remainder of the calculation will be done using complex arithmetic.

The calculation now continues, showing a diagnostic line each step, until we get a second warning about `Svd_truncate()`, this time originating from `tdvp1_dense.py:214`. This is the second time this function is used in the TDVP example script. Again we can ignore this warning if we like; the script will still work. Or we could take one of the suggested actions to stop the warning.

No more warnings are shown for this script, but it continues to produce a diagnostic line for each step. These diagnostics show that the convergence of the calculation is very good, which is reflected in the energy of the state: decreasing rapidly to the groundstate energy of `-7.0` in the first section of the calculation (imaginary time evolution), and approximately constant at `~24.0` in the second section of the calculation (real-time evolution).

Earlier, we ignored the warning about the `LinOp::dtype` hint that caused the unwanted promotion of the real tensor to complex. We can fix that by removing from the script the 5 conversions to `Complex Double` (lines 52,53,54 and 62,63). With this change, the warning about promotion of real input to complex will not appear, and the first half of the calculation, imaginary time evolution, will happen purely in real `Double` arithmetic with no unnecessary complex components.
