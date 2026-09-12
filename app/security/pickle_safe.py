"""Trusted-directory pickle loading (Slice C Wave 1)."""
from __future__ import annotations

import os
import pickle
from typing import List


class SecurityError(Exception):
    """Raised when a security policy blocks an operation."""


_TRUSTED_ROOTS: List[str] = []


def configure_trusted_pickle_roots(*paths: str) -> None:
    """Register absolute/relative roots allowed for pickle.load."""
    global _TRUSTED_ROOTS
    roots: List[str] = []
    for path in paths:
        if not path:
            continue
        try:
            roots.append(os.path.realpath(path))
        except OSError:
            continue
    try:
        roots.append(os.path.realpath(os.getcwd()))
    except OSError:
        pass
    _TRUSTED_ROOTS = roots


def _default_roots() -> List[str]:
    roots: List[str] = []
    for path in ("data", "models", "reports", os.getcwd()):
        try:
            roots.append(os.path.realpath(path))
        except OSError:
            continue
    return roots


def trusted_pickle_roots() -> List[str]:
    return list(_TRUSTED_ROOTS) if _TRUSTED_ROOTS else _default_roots()


def safe_pickle_load(filepath):
    """Load pickle only from trusted application directories."""
    real_path = os.path.realpath(filepath)
    trusted = any(
        real_path == root or real_path.startswith(root + os.sep)
        for root in trusted_pickle_roots()
    )
    if not trusted:
        raise SecurityError(
            f"Refusing to load pickle outside trusted directories: {filepath}"
        )
    with open(real_path, "rb") as f:
        return pickle.load(f)
