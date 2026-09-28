"""A standard, exportable table of readings: rows are layers, columns are readings per space."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

LABELS = {
    "knn_prediction": "k-NN vs. prediction",
    "correctness_auc": "margin AUC for correctness",
    "confusion_agreement": "agreement with confusion",
    "containment": "class directions in V_r",
    "e_ratio_error_auc": "E_ratio AUC for errors",
    "rank": "participation rank",
}


@dataclass
class Report:
    """Readings per layer and per representation (``"joint"``, baselines, ``"floor"``).

    ``values[representation][layer][reading] -> float``
    """

    layers: list[str]
    values: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def representations(self) -> list[str]:
        return list(self.values)

    @property
    def readings(self) -> list[str]:
        first = next(iter(self.values.values()))
        return list(next(iter(first.values())))

    def series(self, reading: str, representation: str = "joint") -> np.ndarray:
        per = self.values.get(representation, {})
        return np.array([per.get(layer, {}).get(reading, np.nan) for layer in self.layers])

    def above_floor(self, reading: str, representation: str = "joint") -> np.ndarray | None:
        """Boolean per layer: strictly above the floor (None when no floor was computed)."""
        if "floor" not in self.values:
            return None
        return self.series(reading, representation) > self.series(reading, "floor")

    def to_dict(self) -> dict:
        return {"layers": self.layers, "values": self.values, "meta": self.meta}

    def to_json(self, path=None, **kw) -> str:
        text = json.dumps(self.to_dict(), default=_jsonable, indent=2, **kw)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_frame(self):
        """Long-format ``pandas.DataFrame`` (requires pandas): representation, layer, reading, value."""
        import pandas as pd
        rows = [{"representation": rep, "layer": layer, "reading": r, "value": v}
                for rep, per in self.values.items() for layer, vals in per.items() for r, v in vals.items()]
        return pd.DataFrame(rows)

    def to_markdown(self, digits: int = 3) -> str:
        out = []
        for reading in self.readings:
            out += [f"### {LABELS.get(reading, reading)}", "",
                    "| | " + " | ".join(self.layers) + " |", "|---|" + "---|" * len(self.layers)]
            for rep in self.representations:
                vals = self.series(reading, rep)
                fmt = (lambda v: f"{v:.0f}") if reading == "rank" else (lambda v: f"{v:.{digits}f}")
                out.append(f"| {rep} | " + " | ".join("—" if not np.isfinite(v) else fmt(v) for v in vals) + " |")
            out.append("")
        return "\n".join(out)

    def __repr__(self) -> str:
        return f"Report(layers={self.layers}, representations={self.representations})"

    def plot(self, readings=None, ax=None):
        from .plots import plot_report
        return plot_report(self, readings)


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
