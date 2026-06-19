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
    dtype = _call_zero_arg(obj, "dtype")
    if dtype is None:
        return None
    try:
        return Type.enum_name(dtype)
    except Exception:
        try:
            return Type.getname(dtype)
        except Exception:
            return str(dtype)


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
    if obj is None:
        return None
    dtype = _dtype_name_from_obj(obj)
    if dtype is not None:
        return dtype
    try:
        block = obj.get_block_()
    except Exception:
        return None
    return _dtype_name_from_obj(block)


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


def _append_stat_field(fields, name, value):
    if value is not None:
        fields.append(f"{name}={_format_stat_value(value)}")


def _format_krylov_diagnostics(stats, args, kwargs, result):
    algorithm = stats.get("algorithm", "Lanczos")
    fields = [f"Finished eigensolver {algorithm}"]
    _append_stat_field(fields, "nx", _hop_nx(args, kwargs))
    _append_stat_field(fields, "input_dtype", stats.get("input_dtype_name"))
    _append_stat_field(fields, "working_dtype", stats.get("working_dtype_name"))
    _append_stat_field(fields, "output_dtype", _result_object_dtype(_output_vector_object(result)))
    _append_stat_field(fields, "eigenvalue_dtype", _result_object_dtype(_eigenvalue_object(result)))
    _append_stat_field(fields, "eigenvalue", _first_scalar_text(_eigenvalue_object(result)))
    _append_stat_field(fields, "matvecs", stats.get("matvec_count"))
    _append_stat_field(fields, "iterations", stats.get("iterations"))
    _append_stat_field(fields, "krylov_dim", stats.get("krylov_dim"))
    _append_stat_field(fields, "cvgcrit", stats.get("cvgcrit_requested"))
    _append_stat_field(fields, "cvgcrit_used", stats.get("cvgcrit_used"))
    _append_stat_field(fields, "maxiter", stats.get("maxiter_requested"))
    _append_stat_field(fields, "maxiter_used", stats.get("maxiter_used"))
    _append_stat_field(fields, "stop", stats.get("reason"))
    _append_stat_field(fields, "final_error", stats.get("final_error"))
    _append_stat_field(fields, "final_beta", stats.get("final_beta"))
    return "[cytnx] " + ", ".join(fields)


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


def _print_krylov_diagnostics(args, kwargs, result):
    stats = _cytnx.linalg.last_krylov_stats()
    print(_format_krylov_diagnostics(stats, args, kwargs, result), file=sys.stderr, flush=True)
    warning = _format_krylov_warning(stats)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)


_native_Lanczos = getattr(_cytnx.linalg.Lanczos, "__cytnx_native_lanczos__", _cytnx.linalg.Lanczos)


def Lanczos(*args, **kwargs):
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
