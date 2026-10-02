"""Engines implementing the term contract."""

from .array_backend import ArrayEngine  # noqa: F401

_DEFAULT = {"engine": None, "name": None}


def available_backends():
    """Return the names of the backends usable on this machine."""
    names = ["numpy"]
    try:
        import jax  # noqa: F401

        names.append("jax")
    except ImportError:
        pass
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            names.append("cuda")
    except Exception:
        pass
    return names


def make_engine(name, **kwargs):
    """Create an engine by backend name."""
    if name == "numpy":
        import numpy

        return ArrayEngine(numpy, **kwargs)
    if name == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        return ArrayEngine(jnp, **kwargs)
    if name == "cuda":
        from .cuda_backend import CudaEngine

        return CudaEngine(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Available: {available_backends()}")


def set_backend(name, **kwargs):
    """Select the default backend."""
    _DEFAULT["engine"] = make_engine(name, **kwargs)
    _DEFAULT["name"] = name
    return _DEFAULT["engine"]


def get_engine():
    """Return the default engine, choosing cuda > jax > numpy on first use."""
    if _DEFAULT["engine"] is None:
        avail = available_backends()
        for name in ("cuda", "jax", "numpy"):
            if name in avail:
                try:
                    set_backend(name)
                    break
                except Exception:
                    continue
    return _DEFAULT["engine"]


def backend_name():
    """Name of the default backend."""
    get_engine()
    return _DEFAULT["name"]
