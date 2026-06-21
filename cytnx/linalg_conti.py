from .utils import *
from cytnx import *

import os
import sys
import warnings

from . import cytnx as _cytnx

_KRYLOV_DIAGNOSTICS_ENV = "CYTNX_KRYLOV_DIAGNOSTICS"
_KRYLOV_DIAGNOSTICS_FALSE = {"0", "false", "off", "no"}


def _env_krylov_diagnostics_enabled():
    value = os.environ.get(_KRYLOV_DIAGNOSTICS_ENV, "1")
    return value.strip().lower() not in _KRYLOV_DIAGNOSTICS_FALSE


_krylov_diagnostics_enabled = _env_krylov_diagnostics_enabled()


def set_krylov_diagnostics(enabled=True):
    """Enable or disable automatic Lanczos/Krylov diagnostics."""
    global _krylov_diagnostics_enabled
    _krylov_diagnostics_enabled = bool(enabled)


def get_krylov_diagnostics():
    """Return whether automatic Lanczos/Krylov diagnostics are enabled."""
    return _krylov_diagnostics_enabled


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


def _hop_nx(args, kwargs):
    hop = kwargs.get("Hop")
    if hop is None and args:
        hop = args[0]
    if hop is None:
        return None
    try:
        return hop.nx()
    except Exception:
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
    _append_stat_field(fields, "E", _first_scalar_text(_eigenvalue_object(result)))
    residual = stats.get("final_residual")
    residual_tol = stats.get("residual_tol_used")
    if residual is not None and residual_tol is not None:
        fields.append(f"res={_format_float_like(residual)}/{_format_float_like(residual_tol)}")
    else:
        _append_stat_field(fields, "res", residual)
    _append_stat_field(fields, "matvecs", stats.get("matvec_count"))
    _append_stat_field(fields, "k", stats.get("krylov_dim"))
    _append_stat_field(fields, "nx", _hop_nx(args, kwargs))
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
    input_dtype = stats.get("input_dtype")
    working_dtype = stats.get("working_dtype")
    output_dtype = _result_object_dtype_id(_output_vector_object(result))
    input_dtype_name = stats.get("input_dtype_name")
    working_dtype_name = stats.get("working_dtype_name")
    output_dtype_name = _dtype_name_from_id(output_dtype)
    if _valid_dtype_id(input_dtype) and _valid_dtype_id(working_dtype) and working_dtype != input_dtype:
        warnings.append(
            f"[cytnx] WARNING: Lanczos working dtype changed: {input_dtype_name} -> "
            f"{working_dtype_name}."
        )
    if _valid_dtype_id(input_dtype) and _valid_dtype_id(output_dtype) and output_dtype != input_dtype:
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


def _print_krylov_diagnostics(args, kwargs, result):
    stats = _cytnx.linalg.last_krylov_stats()
    print(_format_krylov_diagnostics(stats, args, kwargs, result), file=sys.stderr, flush=True)
    for warning in _format_krylov_dtype_warnings(stats, result):
        print(warning, file=sys.stderr, flush=True)
    warning = _format_krylov_warning(stats)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)


_native_Lanczos = getattr(_cytnx.linalg.Lanczos, "__cytnx_native_lanczos__", _cytnx.linalg.Lanczos)


def Lanczos(*args, **kwargs):
    method = _method_arg(args, kwargs)
    method_upper = method.upper() if isinstance(method, str) else method
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
_cytnx.linalg.set_krylov_diagnostics = set_krylov_diagnostics
_cytnx.linalg.get_krylov_diagnostics = get_krylov_diagnostics
