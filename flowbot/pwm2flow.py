"""Single-axis flow models: pwm_incr_module1 ↔ flow_Lmin.

Pools all (module2, module3) combinations, averages flow per module1 PWM
increment, then fits a polynomial:  flow = poly(pwm_incr_module1).
The inverse model (Flow2PwmModel) is derived by fitting flow → pwm on
synthetic sweep points from the forward model.

Simpler and faster than FlowModel (3-input) — use this when only module1
drive level is known or when a quick 1-D lookup is sufficient.

--- Train and save (both forward + inverse) ---
    python pwm2flow.py --save pwm2flow.pkl

--- Predict ---
    python pwm2flow.py --load pwm2flow.pkl --pwm 15

--- Import in controller ---
    from flowbot.pwm2flow import Pwm2FlowModel, Flow2PwmModel
    fwd = Pwm2FlowModel.load("flowbot/pwm2flow.pkl")
    inv = Flow2PwmModel.load("flowbot/flow2pwm.pkl")
    flow = fwd.predict(15)        # pwm_incr → L/min
    pwm  = inv.predict(flow)      # L/min    → pwm_incr
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from sklearn.metrics import r2_score, mean_absolute_error
    _sklearn_ok = True
except ModuleNotFoundError:
    _sklearn_ok = False

try:
    import matplotlib.pyplot as plt
    _plt_ok = True
except ModuleNotFoundError:
    _plt_ok = False

_PWM1_COL = "pwm_incr_module1"
_FLOW_COL  = "flow_Lmin_mean"

# Remap classes that were pickled while running this file directly as __main__
class _Unpickler(pickle.Unpickler):
    _NAMES = {"Pwm2FlowModel", "Flow2PwmModel"}
    def find_class(self, module, name):
        if module == "__main__" and name in self._NAMES:
            module = "flowbot.pwm2flow"
        return super().find_class(module, name)


class Pwm2FlowModel:
    """Polynomial model: pwm_incr_module1 → flow_Lmin.

    All module2 / module3 combinations are pooled and averaged so the
    model captures the module-1-only drive characteristic.
    """

    def __init__(self, coeffs: object, degree: int, pwm_range: tuple,
                 equation_str: str):
        self._coeffs = coeffs          # np.poly1d or ndarray
        self._poly   = np.poly1d(coeffs)
        self.degree  = degree
        self.pwm_range    = pwm_range  # (min_pwm1, max_pwm1)
        self.equation_str = equation_str

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        csv_path: str | Path,
        max_degree: int = 2,
    ) -> "Pwm2FlowModel":
        """Load the average-summary CSV and fit poly(pwm1) → flow.

        Tries degrees 1 … max_degree and keeps the one with highest R².

        Parameters
        ----------
        csv_path   : Average_*.csv produced by build_average_summary.py
        max_degree : highest polynomial degree to try (default 2)
        """
        if np is None or pd is None or not _sklearn_ok:
            raise ImportError(
                "numpy, pandas and scikit-learn are required.\n"
                "  pip install numpy pandas scikit-learn"
            )

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in [_PWM1_COL, _FLOW_COL]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {csv_path}")

        df = df[[_PWM1_COL, _FLOW_COL]].dropna()

        # Average flow over all (module2, module3) combinations per pwm1
        df_avg = (
            df.groupby(_PWM1_COL, as_index=False)
            .agg(flow_mean=(_FLOW_COL, "mean"))
            .sort_values(_PWM1_COL)
        )
        x = df_avg[_PWM1_COL].to_numpy()
        y = df_avg["flow_mean"].to_numpy()

        # Fit each degree, pick best R²
        best_deg, best_coeffs, best_r2 = 1, None, -1e9
        for deg in range(1, max_degree + 1):
            coeffs = np.polyfit(x, y, deg)
            y_pred = np.poly1d(coeffs)(x)
            r2 = r2_score(y, y_pred)
            print(f"  degree {deg}  R²={r2:.6f}")
            if r2 > best_r2:
                best_deg, best_coeffs, best_r2 = deg, coeffs, r2

        # Build equation string
        degree = best_deg
        terms = []
        for i, c in enumerate(best_coeffs):
            power = degree - i
            if power > 1:
                terms.append(f"{c:.8f}·pwm1^{power}")
            elif power == 1:
                terms.append(f"{c:.8f}·pwm1")
            else:
                terms.append(f"{c:.8f}")
        eq = "flow_Lmin = " + " + ".join(terms)

        pwm_range = (float(x.min()), float(x.max()))
        print(f"  Best degree={best_deg}  R²={best_r2:.6f}")
        print(f"  {eq}")
        return cls(best_coeffs, best_deg, pwm_range, eq)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, pwm_incr: float) -> float:
        """Return estimated flow (L/min) for a single PWM increment."""
        return float(self._poly(pwm_incr))

    def predict_batch(self, pwm_incrs: Sequence[float]) -> List[float]:
        """Return estimated flows for a list of PWM increments."""
        return [float(self._poly(p)) for p in pwm_incrs]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Pwm2FlowModel":
        with open(path, "rb") as f:
            obj = _Unpickler(f).load()
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Pwm2FlowModel: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Inverse: flow → pwm  (exact root-finding, preferred)
    # ------------------------------------------------------------------
    def predict_inverse(self, flow_target: float,
                        pwm_min: float = None, pwm_max: float = None) -> float:
        """Return the PWM increment that produces *flow_target* (L/min).

        Uses scipy.optimize.brentq — numerically exact and always consistent
        with the forward model.  Requires the forward model to be monotonically
        increasing over [pwm_min, pwm_max].

        Parameters
        ----------
        flow_target : desired flow in L/min
        pwm_min     : lower PWM bound (default: training minimum)
        pwm_max     : upper PWM bound (default: training maximum)
        """
        try:
            from scipy.optimize import brentq
        except ModuleNotFoundError:
            raise ImportError("scipy is required for predict_inverse. "
                              "  pip install scipy")
        lo = pwm_min if pwm_min is not None else self.pwm_range[0]
        hi = pwm_max if pwm_max is not None else self.pwm_range[1]
        f_lo = self.predict(lo) - flow_target
        f_hi = self.predict(hi) - flow_target
        if f_lo * f_hi > 0:
            # flow_target outside the reachable range — clamp to nearest bound
            return lo if abs(f_lo) < abs(f_hi) else hi
        return float(brentq(lambda pwm: self.predict(pwm) - flow_target, lo, hi))

    def predict_inverse_batch(self, flows: Sequence[float],
                              pwm_min: float = None, pwm_max: float = None) -> List[float]:
        """Return PWM increments for a list of flow targets."""
        return [self.predict_inverse(f, pwm_min, pwm_max) for f in flows]

    # ------------------------------------------------------------------
    # Inverse model: flow → pwm  (polynomial fit, scipy-free fallback)
    # ------------------------------------------------------------------
    def train_inverse(self, degree: int = 3, n_points: int = 300) -> "Flow2PwmModel":
        """Derive an inverse model: flow_Lmin → pwm_incr_module1.

        Sweeps pwm over the training range, evaluates the forward model to
        generate synthetic (flow, pwm) pairs, then fits pwm = poly(flow).
        """
        pwm_vals  = np.linspace(self.pwm_range[0], self.pwm_range[1], n_points)
        flow_vals = np.array([self.predict(p) for p in pwm_vals])

        coeffs = np.polyfit(flow_vals, pwm_vals, degree)
        y_pred = np.poly1d(coeffs)(flow_vals)
        r2 = r2_score(pwm_vals, y_pred)

        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power > 1:
                terms.append(f"{c:.8f}·flow^{power}")
            elif power == 1:
                terms.append(f"{c:.8f}·flow")
            else:
                terms.append(f"{c:.8f}")
        eq = "pwm_incr = " + " + ".join(terms)
        flow_range = (float(flow_vals.min()), float(flow_vals.max()))
        print(f"  Inverse  degree={degree}  R²={r2:.6f}")
        print(f"  {eq}")
        return Flow2PwmModel(coeffs, degree, flow_range, eq)

    def __repr__(self) -> str:
        return (
            f"Pwm2FlowModel(degree={self.degree}, "
            f"pwm1=[{self.pwm_range[0]:.0f},{self.pwm_range[1]:.0f}])\n"
            f"  {self.equation_str}"
        )


# ---------------------------------------------------------------------------
# Inverse model class
# ---------------------------------------------------------------------------
class Flow2PwmModel:
    """Inverse model: flow_Lmin → pwm_incr_module1.

    Derived from Pwm2FlowModel by fitting a polynomial on synthetic
    (flow, pwm) pairs generated by sweeping the forward model.

    Usage::
        fwd = Pwm2FlowModel.load("pwm2flow.pkl")
        inv = fwd.train_inverse()
        pwm = inv.predict(3.5)      # L/min → pwm increment
        inv.save("flow2pwm.pkl")

        # Later in the controller:
        inv = Flow2PwmModel.load("flow2pwm.pkl")
        pwm = inv.predict(flow)
    """

    def __init__(self, coeffs: object, degree: int, flow_range: tuple,
                 equation_str: str):
        self._coeffs = coeffs
        self._poly   = np.poly1d(coeffs)
        self.degree  = degree
        self.flow_range   = flow_range   # (min_flow, max_flow)
        self.equation_str = equation_str

    def predict(self, flow_lmin: float) -> float:
        """Return estimated PWM increment for a given flow (L/min)."""
        return float(self._poly(flow_lmin))

    def predict_batch(self, flows: Sequence[float]) -> List[float]:
        """Return estimated PWM increments for a list of flow values."""
        return [float(self._poly(f)) for f in flows]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Inverse model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Flow2PwmModel":
        with open(path, "rb") as f:
            obj = _Unpickler(f).load()
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Flow2PwmModel: {type(obj)}")
        return obj

    def __repr__(self) -> str:
        return (
            f"Flow2PwmModel(degree={self.degree}, "
            f"flow=[{self.flow_range[0]:.2f},{self.flow_range[1]:.2f}] L/min)\n"
            f"  {self.equation_str}"
        )


# ---------------------------------------------------------------------------
# Evaluation + plot
# ---------------------------------------------------------------------------
def evaluate_and_plot(
    model: Pwm2FlowModel,
    csv_path: str | Path,
    output_png: str = "",
) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[[_PWM1_COL, _FLOW_COL]].dropna()

    df_avg = (
        df.groupby(_PWM1_COL, as_index=False)
        .agg(flow_mean=(_FLOW_COL, "mean"), flow_std=(_FLOW_COL, "std"))
        .sort_values(_PWM1_COL)
    )
    x_avg = df_avg[_PWM1_COL].to_numpy()
    y_avg = df_avg["flow_mean"].to_numpy()
    y_pred = np.array(model.predict_batch(x_avg.tolist()))

    mae = mean_absolute_error(y_avg, y_pred)
    r2  = r2_score(y_avg, y_pred)
    print(f"  MAE : {mae:.4f} L/min   R² : {r2:.4f}")

    if not _plt_ok:
        return

    x_fit = np.linspace(model.pwm_range[0], model.pwm_range[1], 300)
    y_fit = np.array(model.predict_batch(x_fit.tolist()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[_PWM1_COL], df[_FLOW_COL],
               alpha=0.15, s=15, color="steelblue", label="Raw pooled data")
    ax.errorbar(x_avg, y_avg, yerr=df_avg["flow_std"],
                fmt="o", capsize=4, color="navy",
                label="Mean ± std per PWM1")
    ax.plot(x_fit, y_fit, "r-", linewidth=2,
            label=f"Poly deg={model.degree}  R²={r2:.3f}")
    ax.set_xlabel("pwm_incr_module1")
    ax.set_ylabel("Flow rate (L/min)")
    ax.set_title("Flow vs PWM1 increment (module2/3 pooled)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if output_png:
        fig.savefig(output_png, dpi=150)
        print(f"  Plot saved: {output_png}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or use the single-axis flow model (pwm1 → flow)."
    )
    parser.add_argument("--save", default="pwm2flow.pkl",
                        help="Path to save trained model (default: pwm2flow.pkl)")
    parser.add_argument("--load", default="",
                        help="Load existing model instead of training")
    parser.add_argument("--max-degree", type=int, default=2,
                        help="Max polynomial degree to try (default: 2)")
    parser.add_argument("--pwm", type=float, default=None, metavar="PWM1",
                        help="Predict flow for this PWM1 increment")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot display")
    parser.add_argument("--plot-output", default="",
                        help="Save plot to this PNG path")
    return parser


def main() -> None:
    FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(FILE_DIR)
    sys.path.insert(0, PARENT_DIR)
    args = _build_parser().parse_args()

    csv_file = (Path(PARENT_DIR) / "data" / "pwm_vs_pressure_calibration"
                / "average" / "Average_merged_11files.csv")

    if args.load:
        print(f"Loading model from {args.load} ...")
        model = Pwm2FlowModel.load(args.load)
    else:
        print(f"Training on {csv_file} ...")
        model = Pwm2FlowModel.train(csv_file, max_degree=args.max_degree)
        model.save(Path(FILE_DIR) / args.save)

        print("\nTraining inverse model (flow → pwm) ...")
        inv_model = model.train_inverse()
        print(inv_model)
        inv_save = Path(FILE_DIR) / args.save.replace("pwm2flow", "flow2pwm")
        inv_model.save(inv_save)

    print(model)

    if not args.no_plot:
        print("\nEvaluation:")
        evaluate_and_plot(model, csv_file, output_png=args.plot_output)

    if args.pwm is not None:
        flow = model.predict(args.pwm)
        print(f"\npredict(pwm1={args.pwm:.0f}) = {flow:.4f} L/min")


if __name__ == "__main__":
    main()
