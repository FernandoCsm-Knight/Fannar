"""Utilidades de relatorio: JSON sem NaN, numeros com virgula e tabelas em Markdown."""

from __future__ import annotations

import numpy as np


def clean(obj):
    """numpy -> tipos do json; nan -> null."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    return obj


def fmt(v, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}".replace(".", ",")


def table(title: str, rows: dict[str, dict[str, float]], columns: list[str], digits: int = 3) -> str:
    lines = [f"### {title}", "", "| | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for name, values in rows.items():
        lines.append(f"| {name} | " + " | ".join(fmt(values.get(c), digits) for c in columns) + " |")
    return "\n".join(lines) + "\n"
