"""Pressure-flow relationship model for a single suction module (module 1).

Fits a polynomial equation:  press_MPa = f(flow_Lmin)
from the paired (flow_Lmin_mean, press_MPa_mean) values in the average summary CSV.
The data used belongs to PWM1 only (single-module calibration).

--- Fit and save ---
    python pressure_flow_model.py \\
        --csv data/pwm_vs_pressure_calibration/average/Average_merged_Xfiles.csv \\
        --save flow2press.pkl

--- Predict ---
    python pressure_flow_model.py --load flow2press.pkl --flow 2.5

--- Import in controller ---
    from pressure_flow_model import Flow2PressModel
    model = Flow2PressModel.load("flow2press.pkl")
    press = model.predict(2.5)   # flow in L/min → pressure in MPa
"""

from __future__ import annotations

import argparse
import pickle
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
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error, r2_score
    _sklearn_ok = True
except ModuleNotFoundError:
    _sklearn_ok = False

try:
    import matplotlib.pyplot as plt
    _plt_ok = True
except ModuleNotFoundError:
    _plt_ok = False

_FLOW_COL  = "flow_Lmin_mean"
_PRESS_COL = "press_MPa_mean"


def _filter_single_module(df: object, pwm2_max: int, pwm3_max: int) -> object:
    """Keep only rows where modules 2 and 3 are at or below their idle PWM.

    Negative values mean "no filter" (use all rows).
    pwm2_max == 0  → keep rows where pwm2_cur is at its minimum in the dataset.
    pwm2_max  > 0  → keep rows where pwm2_cur <= pwm2_max.
    pwm2_max  < 0  → no filter applied.
    """
    if pwm2_max == 0 and "pwm2_cur" in df.columns:
        df = df[df["pwm2_cur"] <= df["pwm2_cur"].min()]
    elif pwm2_max > 0 and "pwm2_cur" in df.columns:
        df = df[df["pwm2_cur"] <= pwm2_max]
    if pwm3_max == 0 and "pwm3_cur" in df.columns:
        df = df[df["pwm3_cur"] <= df["pwm3_cur"].min()]
    elif pwm3_max > 0 and "pwm3_cur" in df.columns:
        df = df[df["pwm3_cur"] <= pwm3_max]
    return df


def _aggregate_by_flow(df: object, flow_bin: float) -> object:
    """Group rows by rounded flow value and return mean pressure per flow level.

    For each distinct flow level (binned to *flow_bin* L/min resolution),
    all corresponding pressure values are averaged into a single data point.
    This gives the characteristic curve: one (flow, avg_press) per flow level.
    """
    df = df.copy()
    df["_flow_bin"] = (df[_FLOW_COL] / flow_bin).round() * flow_bin
    grouped = (
        df.groupby("_flow_bin")[_PRESS_COL]
        .mean()
        .reset_index()
        .rename(columns={"_flow_bin": _FLOW_COL})
    )
    return grouped.sort_values(_FLOW_COL).reset_index(drop=True)


class Flow2PressModel:
    """Polynomial model: flow_Lmin → press_MPa for a single suction module."""

    def __init__(self, pipeline: object, degree: int, equation_str: str,
                 flow_range: tuple = (0.0, 10.0)):
        self._pipeline = pipeline
        self.degree = degree
        self.equation_str = equation_str
        self.flow_range = flow_range   # (min_flow, max_flow) seen during training

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        csv_path: str | Path,
        degree: int = 2,
        alpha: float = 1e-4,
        pwm2_max: int = 0,
        pwm3_max: int = 0,
        flow_bin: float = 0.1,
        aggregate: bool = True,
    ) -> "Flow2PressModel":
        """Fit press = poly(flow) from the average summary CSV.

        Parameters
        ----------
        csv_path  : Average_*.csv produced by build_average_summary.py
        degree    : Polynomial degree (default 2)
        alpha     : Ridge regularisation (default 1e-4)
        pwm2_max  : Keep only rows where pwm2_cur <= this value (0 = idle minimum)
        pwm3_max  : Keep only rows where pwm3_cur <= this value (0 = idle minimum)
        flow_bin  : Resolution for grouping flow values (L/min). All rows whose
                    flow falls in the same bin are averaged into one data point,
                    giving a single mean pressure per flow level (default 0.1).
        """
        if not _sklearn_ok or np is None or pd is None:
            raise ImportError(
                "numpy, pandas and scikit-learn are required.\n"
                "  pip install numpy pandas scikit-learn"
            )

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in [_FLOW_COL, _PRESS_COL]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Re-run build_average_summary.py "
                    f"to regenerate the CSV with pressure data."
                )

        df = _filter_single_module(df, pwm2_max, pwm3_max)
        df = df[(df[_FLOW_COL] > 0) & (df[_PRESS_COL] > 0)].copy()
        if df.empty:
            raise ValueError("No rows remain after filtering. Check your data or filter settings.")
        if aggregate:
            # Group by flow level → one averaged pressure per flow level
            n_raw = len(df)
            df = _aggregate_by_flow(df, flow_bin)
            print(f"  Raw rows: {n_raw}  →  flow-level groups: {len(df)}  (bin={flow_bin} L/min)")
        else:
            print(f"  Raw rows (no aggregation): {len(df)}")

        X = df[[_FLOW_COL]].values.astype(float)
        y = df[_PRESS_COL].values.astype(float)

        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipeline.fit(X, y)

        # Build human-readable equation string
        coef = pipeline.named_steps["ridge"].coef_
        intercept = pipeline.named_steps["ridge"].intercept_
        terms = [f"{intercept:.6f}"]
        for i in range(1, degree + 1):
            c = coef[i]
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {abs(c):.6f}·flow^{i}")
        eq = "press_MPa = " + " ".join(terms)
        flow_range = (float(X[:, 0].min()), float(X[:, 0].max()))

        return cls(pipeline, degree, eq, flow_range)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, flow_lmin: float) -> float:
        """Estimate pressure (MPa) for a given flow rate (L/min)."""
        X = np.array([[flow_lmin]], dtype=float)
        return float(self._pipeline.predict(X)[0])

    def predict_batch(self, flows: Sequence[float]) -> List[float]:
        """Estimate pressures for a list of flow values."""
        X = np.array(flows, dtype=float).reshape(-1, 1)
        return self._pipeline.predict(X).tolist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Flow2PressModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Flow2PressModel: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Inverse model: pressure → flow
    # ------------------------------------------------------------------
    def train_inverse(self, degree: int = 3, alpha: float = 1e-4,
                      n_points: int = 300) -> "Press2FlowModel":
        """Derive an inverse model: press_MPa → flow_Lmin.

        Sweeps flow over the training range, evaluates the forward model to
        generate synthetic (press, flow) pairs, then fits flow = poly(press).
        Using synthetic data guarantees the inverse is fully consistent with
        the forward model without needing access to the original CSV.

        Parameters
        ----------
        degree   : polynomial degree for the inverse fit (default 3)
        alpha    : Ridge regularisation (default 1e-4)
        n_points : number of synthetic sweep points (default 300)
        """
        flow_vals = np.linspace(self.flow_range[0], self.flow_range[1], n_points)
        press_vals = np.array([self.predict(f) for f in flow_vals])

        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipeline.fit(press_vals.reshape(-1, 1), flow_vals)

        # Build equation string
        coef = pipeline.named_steps["ridge"].coef_
        intercept = pipeline.named_steps["ridge"].intercept_
        terms = [f"{intercept:.6f}"]
        for i in range(1, degree + 1):
            c = coef[i]
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {abs(c):.6f}·press^{i}")
        eq = "flow_Lmin = " + " ".join(terms)

        press_range = (float(press_vals.min()), float(press_vals.max()))
        return Press2FlowModel(pipeline, degree, eq, press_range)

    def __repr__(self) -> str:
        return f"Flow2PressModel(degree={self.degree})\n  {self.equation_str}"


# ---------------------------------------------------------------------------
# Inverse model class
# ---------------------------------------------------------------------------
class Press2FlowModel:
    """Inverse model: press_MPa → flow_Lmin.

    Derived from the forward Flow2PressModel by fitting a polynomial on
    synthetic (press, flow) pairs generated by sweeping the forward model.

    Usage::
        forward = Flow2PressModel.load("flow2press.pkl")
        inv = forward.train_inverse()
        flow = inv.predict(0.05)          # pressure MPa → flow L/min
        inv.save("press2flow.pkl")

        # Later in the controller:
        inv = Press2FlowModel.load("press2flow.pkl")
        flow = inv.predict(0.05)
    """

    def __init__(self, pipeline: object, degree: int, equation_str: str,
                 press_range: tuple = (0.0, 0.2)):
        self._pipeline = pipeline
        self.degree = degree
        self.equation_str = equation_str
        self.press_range = press_range   # (min_press, max_press) valid input range

    def predict(self, press_mpa: float) -> float:
        """Return estimated flow (L/min) for a given pressure (MPa)."""
        X = np.array([[press_mpa]], dtype=float)
        return float(self._pipeline.predict(X)[0])

    def predict_batch(self, pressures: Sequence[float]) -> List[float]:
        """Return estimated flows for a list of pressure values."""
        X = np.array(pressures, dtype=float).reshape(-1, 1)
        return self._pipeline.predict(X).tolist()

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Inverse model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Press2FlowModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Press2FlowModel: {type(obj)}")
        return obj

    def __repr__(self) -> str:
        return (
            f"Press2FlowModel(degree={self.degree}, "
            f"press_range=[{self.press_range[0]:.4f}, {self.press_range[1]:.4f}] MPa)\n"
            f"  {self.equation_str}"
        )


# ---------------------------------------------------------------------------
# Evaluation + plot  (forward + inverse in one window)
# ---------------------------------------------------------------------------
def evaluate_and_plot(
    model: Flow2PressModel,
    inv_model: Press2FlowModel,
    csv_path: str | Path,
    flow_bin: float = 0.1,
    press_bin: float = 0.0005,
    output_png: str = "",
    pwm2_max: int = -1,
    pwm3_max: int = -1,
) -> None:
    """Plot forward (flow→press) and inverse (press→flow) side-by-side."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _filter_single_module(df, pwm2_max, pwm3_max)
    df = df[(df[_FLOW_COL] > 0) & (df[_PRESS_COL] > 0)].copy()

    flow_raw  = df[_FLOW_COL].values.astype(float)
    press_raw = df[_PRESS_COL].values.astype(float)

    # --- Forward: flow-binned averages ---
    df_fagg   = _aggregate_by_flow(df, flow_bin)
    flow_agg  = df_fagg[_FLOW_COL].values.astype(float)
    press_agg = df_fagg[_PRESS_COL].values.astype(float)
    press_pred = np.array(model.predict_batch(flow_agg.tolist()))
    mae_f = mean_absolute_error(press_agg, press_pred)
    r2_f  = r2_score(press_agg, press_pred)

    # --- Inverse: pressure-binned averages ---
    df2 = df.copy()
    df2["_press_bin"] = (df2[_PRESS_COL] / press_bin).round() * press_bin
    df_pagg = (
        df2.groupby("_press_bin")[_FLOW_COL]
        .mean().reset_index()
        .rename(columns={"_press_bin": _PRESS_COL})
        .sort_values(_PRESS_COL).reset_index(drop=True)
    )
    press_agg2 = df_pagg[_PRESS_COL].values.astype(float)
    flow_agg2  = df_pagg[_FLOW_COL].values.astype(float)
    flow_pred  = np.array(inv_model.predict_batch(press_agg2.tolist()))
    mae_i = mean_absolute_error(flow_agg2, flow_pred)
    r2_i  = r2_score(flow_agg2, flow_pred)

    print(f"  Forward  — flow-level groups : {len(flow_agg)}")
    print(f"             MAE  : {mae_f:.6f} MPa   R² : {r2_f:.4f}")
    print(f"             Eq   : {model.equation_str}")
    print(f"  Inverse  — press-level groups: {len(press_agg2)}")
    print(f"             MAE  : {mae_i:.4f} L/min  R² : {r2_i:.4f}")
    print(f"             Eq   : {inv_model.equation_str}")

    if not _plt_ok:
        return

    flow_line  = np.linspace(flow_raw.min(),  flow_raw.max(),  200)
    press_line = np.linspace(press_raw.min(), press_raw.max(), 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: forward model
    ax1.scatter(flow_raw, press_raw, s=12, alpha=0.3, color="steelblue", label="Raw per-PWM avg")
    ax1.scatter(flow_agg, press_agg, s=40, alpha=0.9, color="navy",
                label=f"Mean per flow bin ({flow_bin} L/min)", zorder=3)
    ax1.plot(flow_line, np.array(model.predict_batch(flow_line.tolist())), "r-", linewidth=2,
             label=f"Poly deg={model.degree}  R²={r2_f:.3f}")
    ax1.set_xlabel("Flow rate (L/min)")
    ax1.set_ylabel("Pressure (MPa)")
    ax1.set_title("Forward: Pressure vs Flow")
    ax1.legend()

    # Right: inverse model
    ax2.scatter(press_raw, flow_raw, s=12, alpha=0.3, color="steelblue", label="Raw per-PWM avg")
    ax2.scatter(press_agg2, flow_agg2, s=40, alpha=0.9, color="navy",
                label=f"Mean per press bin ({press_bin} MPa)", zorder=3)
    ax2.plot(press_line, np.array(inv_model.predict_batch(press_line.tolist())), "r-", linewidth=2,
             label=f"Poly deg={inv_model.degree}  R²={r2_i:.3f}")
    ax2.set_xlabel("Pressure (MPa)")
    ax2.set_ylabel("Flow rate (L/min)")
    ax2.set_title("Inverse: Flow vs Pressure")
    ax2.legend()

    fig.suptitle("Pressure–Flow calibration — module 1", fontsize=13)
    fig.tight_layout()

    if output_png:
        fig.savefig(output_png, dpi=150)
        print(f"  Plot saved  : {output_png}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and use the pressure-flow model (press = f(flow))."
    )
    parser.add_argument("--train", "-t", default=True, help="train model")
    parser.add_argument("--save", default="flow2press.pkl", help="Path to save model")
    parser.add_argument("--load", default="", help="Path to load existing model")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
    parser.add_argument("--alpha",  type=float, default=1e-4, help="Ridge alpha (default: 1e-4)")
    parser.add_argument(
        "--flow-bin", type=float, default=0.1,
        help="Flow bin size for grouping (L/min). All rows in the same bin are averaged "
             "into one pressure value before fitting. Default: 0.1",
    )
    parser.add_argument(
        "--single-module", action="store_true",
        help="Restrict to single-module rows (modules 2 and 3 at idle). Default: use all rows.",
    )
    parser.add_argument(
        "--flow", type=float, default=None, metavar="FLOW_LMIN",
        help="Predict pressure for this flow value (L/min)",
    )
    parser.add_argument("--plot-output", default="", help="Save plot to this PNG path")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot display") 
    parser.add_argument(
        "--no-aggregate", "-no-agg",  dest="aggregate", action="store_false",
        help="Fit on raw per-PWM data points instead of averaging by flow level",
    )
    parser.set_defaults(aggregate=True)
    
    return parser


def main() -> None:
    import os
    import sys
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(FILE_DIR)
    sys.path.insert(0, PARENT_DIR)
    args = build_arg_parser().parse_args()

    if np is None or pd is None or not _sklearn_ok:
        raise SystemExit(
            "Required packages missing.\n  pip install numpy pandas scikit-learn matplotlib"
        )

    pwm2_max = 0 if args.single_module else -1
    pwm3_max = 0 if args.single_module else -1

    if args.load:
        print(f"Loading model from {args.load} ...")
        model = Flow2PressModel.load(args.load)
        print(model)
    elif args.train:
        csv_file = Path(PARENT_DIR) / "data" / "pwm_vs_pressure_calibration" / "average" / "Average_merged_11files.csv"
        print(f"Training on {csv_file}  (degree={args.degree}) ...")
        model = Flow2PressModel.train(
            csv_file,
            degree=args.degree,
            alpha=args.alpha,
            pwm2_max=pwm2_max,
            pwm3_max=pwm3_max,
            flow_bin=args.flow_bin,
            aggregate=args.aggregate,
        )
        print(model)

        # Train inverse model (press → flow)
        print("\nTraining inverse model (press → flow) ...")
        inv_model = model.train_inverse(degree=args.degree)
        print(inv_model)

        print("\nFit metrics:")
        if not args.no_plot:
            evaluate_and_plot(
                model, inv_model, csv_file,
                flow_bin=args.flow_bin,
                output_png=args.plot_output,
                pwm2_max=pwm2_max,
                pwm3_max=pwm3_max,
            )

        model.save(Path(FILE_DIR) / args.save)
        inv_save = Path(FILE_DIR) / args.save.replace("flow2press", "press2flow")
        inv_model.save(inv_save)
        
    else:
        raise SystemExit("Provide --csv to train or --load to load an existing model.")

    if args.flow is not None:
        press = model.predict(args.flow)
        print(f"\nforward: predict({args.flow:.3f} L/min) = {press:.6f} MPa")


if __name__ == "__main__":
    main()
