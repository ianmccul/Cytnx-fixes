from .utils import *
from cytnx import *

import os
import sys
import warnings

from . import cytnx as _cytnx

_KRYLOV_DIAGNOSTICS_ENV = "CYTNX_KRYLOV_DIAGNOSTICS"
_KRYLOV_DIAGNOSTICS_FALSE = {"0", "false", "off", "no"}
_SVD_TRUNCATE_WARNINGS_ENV = "CYTNX_SVD_TRUNCATE_WARNINGS"
_SVD_TRUNCATE_WARNINGS_FALSE = {"0", "false", "off", "no"}
_SVD_TRUNCATE_DEFAULT_ERR = 1e-8
_SVD_TRUNCATE_DEFAULT_ERR_TEXT = "1e-8"
_CYTNX_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_SVD_TRUNCATE_DEFAULT_WARNING = """
cytnx.linalg.Svd_truncate was called without an explicit err cutoff.

This fixes branch changes the Python default to:

    err={err}

This removes numerically null or near-null singular vectors. For typical
double-precision MPS/TDVP calculations this corresponds to discarded state
weight around 1e-16, while removing basis directions that are not numerically
meaningful. Removing near-null singular vectors is especially important for
TDVP and other tangent-space algorithms, where such directions can corrupt the
tangent-space projector and give wrong results.

If you are only canonicalizing an MPS and do not need singular values, prefer:

    cytnx.linalg.Qr(...)    # left-canonicalization
    cytnx.linalg.Lq(...)    # right-canonicalization

If you want singular values but do not want truncation, use:

    cytnx.linalg.Svd(...)
    cytnx.linalg.Gesvd(...)

If you deliberately want to keep null singular vectors, pass a negative cutoff:

    cytnx.linalg.Svd_truncate(T, keepdim, err=-1)

This warning is shown once per Python call site. You can silence it globally:

    cytnx.linalg.set_svd_truncate_warnings(False)

or with the environment variable:

    export CYTNX_SVD_TRUNCATE_WARNINGS=0

The best long-term fix is to pass an explicit cutoff at every call site, for
example:

    cytnx.linalg.Svd_truncate(T, keepdim, err={err})
""".strip()
_SVD_TRUNCATE_ZERO_ERROR = """
cytnx.linalg.Svd_truncate no longer accepts err=0 from Python.

The value err=0 is ambiguous and unsafe. In most tensor-network code, a call to
Svd_truncate means either:

1. truncate an MPS bond using a singular-value cutoff; or
2. canonicalize a tensor network bond.

For case 1, keeping numerically null singular vectors is usually wrong. It is
particularly dangerous in TDVP and other tangent-space algorithms, because the
null-space basis enters the projector. An arbitrary numerical null-space basis
can therefore change the algorithm.

For case 2, Svd_truncate is usually the wrong tool. If you only want
canonicalization and do not need singular values, prefer:

    cytnx.linalg.Qr(...)    # left-canonicalization
    cytnx.linalg.Lq(...)    # right-canonicalization

If you want singular values but do not want truncation, use:

    cytnx.linalg.Svd(...)
    cytnx.linalg.Gesvd(...)

If you are using Svd_truncate to truncate an MPS bond, the MPS must already be
canonical at that bond. Then choose one of these explicit meanings:

    err=1e-8
        Remove numerical zero and near-zero singular values. This corresponds
        to discarded state weight around 1e-16 and is the recommended default
        for double-precision MPS/TDVP-style calculations.

    err=<positive cutoff>
        Apply your own positive singular-value cutoff.

    err=-1
        Keep every singular vector, including numerically zero vectors. Use
        this only if you have checked that the null-space basis cannot affect
        later calculations.
""".strip()


def _env_krylov_diagnostics_enabled():
    value = os.environ.get(_KRYLOV_DIAGNOSTICS_ENV, "1")
    return value.strip().lower() not in _KRYLOV_DIAGNOSTICS_FALSE


def _env_svd_truncate_warnings_enabled():
    value = os.environ.get(_SVD_TRUNCATE_WARNINGS_ENV, "1")
    return value.strip().lower() not in _SVD_TRUNCATE_WARNINGS_FALSE


_krylov_diagnostics_enabled = _env_krylov_diagnostics_enabled()
_svd_truncate_warnings_enabled = _env_svd_truncate_warnings_enabled()
_svd_truncate_warning_sites = set()


def set_krylov_diagnostics(enabled=True):
    """Enable or disable automatic Lanczos/Krylov diagnostics."""
    global _krylov_diagnostics_enabled
    _krylov_diagnostics_enabled = bool(enabled)


def get_krylov_diagnostics():
    """Return whether automatic Lanczos/Krylov diagnostics are enabled."""
    return _krylov_diagnostics_enabled


def set_svd_truncate_warnings(enabled=True):
    """Enable or disable Svd_truncate zero-cutoff call-site warnings."""
    global _svd_truncate_warnings_enabled
    _svd_truncate_warnings_enabled = bool(enabled)


def get_svd_truncate_warnings():
    """Return whether Svd_truncate zero-cutoff call-site warnings are enabled."""
    return _svd_truncate_warnings_enabled


def _call_zero_arg(obj, name):
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    if callable(attr):
        return attr()
    return attr


def _dtype_name_from_obj(obj):
    dtype = _dtype_id_from_obj(obj)
    if dtype is None:
        return None
    return _dtype_name_from_id(dtype)


def _dtype_name_from_id(dtype):
    if dtype is None:
        return None
    try:
        return Type.getname(dtype)
    except Exception:
        try:
            return Type.enum_name(dtype)
        except Exception:
            return str(dtype)


def _dtype_id_from_obj(obj):
    dtype = _call_zero_arg(obj, "dtype")
    if dtype is None:
        return None
    try:
        return int(dtype)
    except Exception:
        return dtype


def _result_sequence(result):
    if isinstance(result, (list, tuple)):
        return result
    try:
        return list(result)
    except Exception:
        return [result]


def _result_object(result, index):
    values = _result_sequence(result)
    if index >= len(values):
        return None
    return values[index]


def _result_object_dtype(obj):
    dtype = _result_object_dtype_id(obj)
    return _dtype_name_from_id(dtype)


def _result_object_dtype_id(obj):
    if obj is None:
        return None
    dtype = _dtype_id_from_obj(obj)
    if dtype is not None:
        return dtype
    try:
        block = obj.get_block_()
    except Exception:
        return None
    return _dtype_id_from_obj(block)


def _eigenvalue_object(result):
    return _result_object(result, 0)


def _output_vector_object(result):
    vector = _result_object(result, 1)
    if vector is not None:
        return vector
    return _result_object(result, 0)


def _first_scalar_text(obj):
    if obj is None:
        return None
    candidates = [obj]
    try:
        candidates.append(obj.get_block_())
    except Exception:
        pass
    for candidate in candidates:
        try:
            return str(candidate.item())
        except Exception:
            pass
        try:
            return str(candidate[0].item())
        except Exception:
            pass
        try:
            return str(candidate.storage().at(0))
        except Exception:
            pass
    return None


def _format_stat_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_float_like(value):
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _append_stat_field(fields, name, value):
    if value is not None:
        fields.append(f"{name}={_format_stat_value(value)}")


def _valid_dtype_name(dtype):
    return dtype not in {None, "Void", "void", ""}


def _valid_dtype_id(dtype):
    return dtype is not None and dtype != getattr(Type, "Void", None)


def _dtype_is_complex(dtype):
    if not _valid_dtype_id(dtype):
        return False
    try:
        return bool(Type.is_complex(dtype))
    except Exception:
        return False


def _krylov_diagnostic_label(stats, args, kwargs):
    method = _method_arg(args, kwargs)
    if isinstance(method, str):
        upper_method = method.upper()
        if upper_method == "GND":
            return "GND"
        if upper_method == "ER":
            return "ER"
        if upper_method in {"LM", "LA", "SA"}:
            return f"ARPACK({upper_method})"
    algorithm = stats.get("algorithm", "Lanczos")
    if algorithm in {"Lanczos_Gnd", "Lanczos_Gnd_Ut"}:
        return "GND"
    if algorithm in {"Lanczos", "Lanczos_Ut"}:
        return "ARPACK"
    return algorithm


def _user_stop_reason(reason):
    if reason == "breakdown":
        return "eigenvector"
    if reason == "full_krylov_dimension":
        return "full_space"
    return reason


def _format_krylov_diagnostics(stats, args, kwargs, result):
    fields = [f"{_krylov_diagnostic_label(stats, args, kwargs)}:"]
    algorithm = stats.get("algorithm")
    if algorithm == "Lanczos_Exp":
        final_error = stats.get("final_error")
        cvgcrit = stats.get("cvgcrit_used")
        if final_error is not None and cvgcrit is not None:
            fields.append(f"err={_format_float_like(final_error)}/{_format_float_like(cvgcrit)}")
        else:
            _append_stat_field(fields, "err", final_error)
    else:
        _append_stat_field(fields, "E", _first_scalar_text(_eigenvalue_object(result)))
        residual = stats.get("final_residual")
        residual_tol = stats.get("residual_tol_used")
        if residual is not None and residual_tol is not None:
            fields.append(f"res={_format_float_like(residual)}/{_format_float_like(residual_tol)}")
        else:
            _append_stat_field(fields, "res", residual)
    _append_stat_field(fields, "matvecs", stats.get("matvec_count"))
    _append_stat_field(fields, "k", stats.get("krylov_dim"))
    _append_stat_field(fields, "dtype", stats.get("input_dtype_name"))
    stop_reason = _user_stop_reason(stats.get("reason"))
    _append_stat_field(fields, "stop", stop_reason)
    maxiter_requested = stats.get("maxiter_requested")
    maxiter_used = stats.get("maxiter_used")
    if maxiter_requested is not None and maxiter_used is not None and maxiter_requested != maxiter_used:
        fields.append(f"maxiter={maxiter_used}/{maxiter_requested}")
    if len(fields) == 1:
        return "[cytnx] " + fields[0]
    return "[cytnx] " + fields[0] + " " + ", ".join(fields[1:])


def _format_krylov_dtype_warnings(stats, result):
    warnings = []
    algorithm = stats.get("algorithm")
    input_dtype = stats.get("input_dtype")
    working_dtype = stats.get("working_dtype")
    output_dtype = None if algorithm == "Lanczos_Exp" else _result_object_dtype_id(
        _output_vector_object(result)
    )
    input_dtype_name = stats.get("input_dtype_name")
    working_dtype_name = stats.get("working_dtype_name")
    output_dtype_name = _dtype_name_from_id(output_dtype)
    if _valid_dtype_id(input_dtype) and _valid_dtype_id(working_dtype) and working_dtype != input_dtype:
        warnings.append(
            f"[cytnx] WARNING: Lanczos working dtype changed: {input_dtype_name} -> "
            f"{working_dtype_name}."
        )
    if (
        algorithm != "Lanczos_Exp"
        and _valid_dtype_id(input_dtype)
        and _valid_dtype_id(output_dtype)
        and output_dtype != input_dtype
    ):
        warnings.append(
            f"[cytnx] WARNING: Lanczos returned vector dtype {output_dtype_name} from input dtype "
            f"{input_dtype_name}."
        )
    return warnings


def _format_krylov_warning(stats):
    try:
        reason = stats.get("reason")
        cvgcrit = float(stats.get("cvgcrit_requested"))
        matvecs = int(stats.get("matvec_count"))
        maxiter = int(stats.get("maxiter_requested"))
    except Exception:
        return None
    if reason == "energy_diff" and cvgcrit >= 1.0 and matvecs < maxiter:
        return (
            "[cytnx] WARNING: Lanczos stopped by an absolute energy-difference "
            f"threshold before Maxiter was used; CvgCrit={cvgcrit:g} is very "
            "large and can make Maxiter ineffective."
        )
    return None


def _method_arg(args, kwargs):
    if "method" in kwargs:
        return kwargs["method"]
    if len(args) >= 3:
        return args[2]
    return None


def _krylov_input_arg(args, kwargs):
    if len(args) >= 2:
        return args[1]
    for name in ("Tin", "v", "UT_init"):
        if name in kwargs:
            return kwargs[name]
    return None


def _reject_tensor_krylov_input(function_name, args, kwargs):
    input_arg = _krylov_input_arg(args, kwargs)
    if isinstance(input_arg, _cytnx.Tensor):
        raise TypeError(
            f"cytnx.linalg.{function_name} no longer accepts Tensor initial vectors. "
            "Use a UniTensor initial vector instead. If you really want to run a Krylov "
            "algorithm on dense Tensor data, wrap it explicitly as a UniTensor first."
        )


def _print_krylov_diagnostics(args, kwargs, result):
    stats = _cytnx.linalg.last_krylov_stats()
    print(_format_krylov_diagnostics(stats, args, kwargs, result), file=sys.stderr, flush=True)
    for warning in _format_krylov_dtype_warnings(stats, result):
        print(warning, file=sys.stderr, flush=True)
    warning = _format_krylov_warning(stats)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)


_native_Lanczos = getattr(_cytnx.linalg.Lanczos, "__cytnx_native_lanczos__", _cytnx.linalg.Lanczos)
_native_Lanczos_Exp = getattr(
    _cytnx.linalg.Lanczos_Exp, "__cytnx_native_lanczos_exp__", _cytnx.linalg.Lanczos_Exp
)


def Lanczos(*args, **kwargs):
    method = _method_arg(args, kwargs)
    method_upper = method.upper() if isinstance(method, str) else method
    _reject_tensor_krylov_input("Lanczos", args, kwargs)
    if method_upper == "ER":
        raise TypeError(
            "cytnx.linalg.Lanczos(..., method='ER') is disabled. Use one of the "
            "ARPACK solvers instead; for ground states use method='SA' "
            "(smallest algebraic eigenvalue)."
        )
    if len(args) >= 4 and method_upper == "GND":
        raise TypeError(
            "cytnx.linalg.Lanczos(..., method='Gnd') no longer accepts a fourth "
            "positional convergence parameter. Use residual_tol=... by keyword "
            "for the Ritz residual tolerance. If you are not sure, try "
            "residual_tol=1e-14."
        )
    if "CvgCrit" in kwargs and method_upper == "GND":
        raise TypeError(
            "cytnx.linalg.Lanczos(..., method='Gnd') no longer accepts CvgCrit. "
            "Use residual_tol=... for the Ritz residual tolerance. If you are "
            "not sure, try residual_tol=1e-14."
        )
    result = _native_Lanczos(*args, **kwargs)
    if _krylov_diagnostics_enabled:
        try:
            _print_krylov_diagnostics(args, kwargs, result)
        except Exception as exc:
            warnings.warn(f"Failed to print Cytnx Krylov diagnostics: {exc}", RuntimeWarning)
    return result


Lanczos.__cytnx_native_lanczos__ = _native_Lanczos
_cytnx.linalg.Lanczos = Lanczos


def Lanczos_Exp(*args, **kwargs):
    _reject_tensor_krylov_input("Lanczos_Exp", args, kwargs)
    result = _native_Lanczos_Exp(*args, **kwargs)
    if _krylov_diagnostics_enabled:
        try:
            _print_krylov_diagnostics(args, kwargs, result)
        except Exception as exc:
            warnings.warn(f"Failed to print Cytnx Krylov diagnostics: {exc}", RuntimeWarning)
    return result


Lanczos_Exp.__cytnx_native_lanczos_exp__ = _native_Lanczos_Exp
_cytnx.linalg.Lanczos_Exp = Lanczos_Exp
_cytnx.linalg.set_krylov_diagnostics = set_krylov_diagnostics
_cytnx.linalg.get_krylov_diagnostics = get_krylov_diagnostics


def _is_cytnx_python_file(filename):
    try:
        path = os.path.abspath(filename)
        return os.path.commonpath([path, _CYTNX_PACKAGE_DIR]) == _CYTNX_PACKAGE_DIR
    except Exception:
        return False


def _python_user_call_site():
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not _is_cytnx_python_file(filename):
            return filename, frame.f_lineno
        frame = frame.f_back
    return "<unknown>", 0


def _is_min_blockdim_argument(value):
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (list, tuple))


def _svd_truncate_explicit_err(args, kwargs):
    if "err" in kwargs:
        return True, kwargs["err"]
    if len(args) < 3:
        return False, None
    if _is_min_blockdim_argument(args[2]):
        if len(args) < 4:
            return False, None
        return True, args[3]
    return True, args[2]


def _is_zero_cutoff(value):
    try:
        return float(value) == 0.0
    except Exception:
        return False


def _is_negative_cutoff(value):
    try:
        return float(value) < 0.0
    except Exception:
        return False


def _warn_svd_truncate_default_cutoff(args, kwargs):
    if not _svd_truncate_warnings_enabled:
        return
    filename, lineno = _python_user_call_site()
    site = (filename, lineno)
    if site in _svd_truncate_warning_sites:
        return
    _svd_truncate_warning_sites.add(site)
    message = _SVD_TRUNCATE_DEFAULT_WARNING.format(err=_SVD_TRUNCATE_DEFAULT_ERR_TEXT)
    warnings.warn_explicit(message, RuntimeWarning, filename, lineno)


def _svd_truncate_checked_args(args, kwargs):
    has_err, err = _svd_truncate_explicit_err(args, kwargs)
    if not has_err:
        _warn_svd_truncate_default_cutoff(args, kwargs)
        kwargs = dict(kwargs)
        kwargs["err"] = _SVD_TRUNCATE_DEFAULT_ERR
        return args, kwargs
    if _is_zero_cutoff(err):
        raise ValueError(_SVD_TRUNCATE_ZERO_ERROR)
    if _is_negative_cutoff(err):
        return args, kwargs
    return args, kwargs


_native_Svd_truncate = getattr(
    _cytnx.linalg.Svd_truncate, "__cytnx_native_svd_truncate__", _cytnx.linalg.Svd_truncate
)


def Svd_truncate(*args, **kwargs):
    args, kwargs = _svd_truncate_checked_args(args, kwargs)
    return _native_Svd_truncate(*args, **kwargs)


Svd_truncate.__cytnx_native_svd_truncate__ = _native_Svd_truncate
_cytnx.linalg.Svd_truncate = Svd_truncate
_cytnx.linalg.set_svd_truncate_warnings = set_svd_truncate_warnings
_cytnx.linalg.get_svd_truncate_warnings = get_svd_truncate_warnings
