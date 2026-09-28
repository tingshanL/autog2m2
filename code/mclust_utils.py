#!/usr/bin/env python3
"""Small rpy2 wrapper for the mclust fits used in Figures 6 and 8."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MclustResult:
    labels: np.ndarray
    elapsed_seconds: float
    selected_components: int
    selected_model: str
    bic_max: float


class MclustRunner:
    """Load R/mclust once and fit multiple matrices in one Python process."""

    def __init__(self) -> None:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, pandas2ri
        from rpy2.robjects.packages import importr

        self.ro = ro
        self.converter = default_converter + pandas2ri.converter
        self.mclust = importr("mclust")

    def fit(self, X: np.ndarray, components: Iterable[int]) -> MclustResult:
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.vectors import IntVector

        component_values = [int(value) for value in components]
        if not component_values:
            raise ValueError("components must contain at least one value")

        frame = pd.DataFrame(np.asarray(X, dtype=float))
        with localconverter(self.converter):
            r_frame = self.ro.conversion.py2rpy(frame)

        started = time.perf_counter()
        fitted = self.mclust.Mclust(
            r_frame,
            G=IntVector(component_values),
            verbose=False,
        )
        elapsed = time.perf_counter() - started

        labels = np.asarray(fitted.rx2("classification"), dtype=int)
        selected_components = int(np.asarray(fitted.rx2("G"), dtype=int)[0])
        model_value = fitted.rx2("modelName")
        selected_model = str(model_value[0])
        bic_values = np.asarray(fitted.rx2("bic"), dtype=float)
        finite = bic_values[np.isfinite(bic_values)]
        bic_max = float(finite.max()) if finite.size else float("nan")
        return MclustResult(
            labels=labels,
            elapsed_seconds=float(elapsed),
            selected_components=selected_components,
            selected_model=selected_model,
            bic_max=bic_max,
        )

