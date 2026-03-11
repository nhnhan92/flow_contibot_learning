"""Flow estimation model trained from the average summary CSV.

Estimates total flow (L/min) from three module PWM increments using a
polynomial regression model.  The fitted model is fast to evaluate and
suitable for use inside a real-time controller.

--- Training (command line) ---
    python flow_model.py --csv data/pwm_vs_pressure_calibration/average/Average_merged_Xfiles.csv \\
                         --save flow_model.pkl

--- Prediction (command line) ---
    python flow_model.py --load flow_model.pkl --pwm 10 5 8

--- Importing in the controller ---
    from flow_model import FlowModel
    model = FlowModel.load("flow_model.pkl")
    flow = model.predict(10, 5, 8)           # scalar
    flows = model.predict_batch([[10,5,8], [15,12,14]])  # list of scalars
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple
import os   
import sys
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
except ModuleNotFoundError:
    Ridge = PolynomialFeatures = Pipeline = None
    mean_absolute_error = r2_score = None


# Column names expected in the summary CSV
_INCR_COLS = ["pwm_incr_module1", "pwm_incr_module2", "pwm_incr_module3"]
_FLOW_COL = "flow_Lmin_mean"


class FlowModel:
    """Polynomial regression model: (incr1, incr2, incr3) -> flow_Lmin."""

    def __init__(self, pipeline: object, degree: int, feature_range: Tuple[float, float, float, float, float, float]):
        self._pipeline = pipeline
        self.degree = degree
        # (min1, max1, min2, max2, min3, max3) for reference / sanity checks
        self.feature_range = feature_range

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        csv_path: str | Path,
        degree: int = 2,
        alpha: float = 1e-3,
    ) -> "FlowModel":
        """Load a summary CSV and fit a polynomial regression model.

        Parameters
        ----------
        csv_path : path to the Average_*.csv file produced by build_average_summary.py
        degree   : polynomial degree (default 2; try 3 for more flexibility)
        alpha    : Ridge regularisation strength (default 1e-3)
        """
        if np is None or pd is None or Ridge is None:
            raise ImportError("numpy, pandas, and scikit-learn are required. "
                              "Install with: pip install numpy pandas scikit-learn")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        missing = [c for c in _INCR_COLS + [_FLOW_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        X = df[_INCR_COLS].values.astype(float)
        y = df[_FLOW_COL].values.astype(float)

        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipeline.fit(X, y)

        feature_range = (
            float(X[:, 0].min()), float(X[:, 0].max()),
            float(X[:, 1].min()), float(X[:, 1].max()),
            float(X[:, 2].min()), float(X[:, 2].max()),
        )
        return cls(pipeline, degree, feature_range)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, incr1: float, incr2: float, incr3: float) -> float:
        """Return estimated flow in L/min for a single PWM increment triple."""
        X = np.array([[incr1, incr2, incr3]], dtype=float)
        return float(self._pipeline.predict(X)[0])

    def predict_batch(self, pwm_triples: Sequence[Sequence[float]]) -> List[float]:
        """Return estimated flows for a list of (incr1, incr2, incr3) triples."""
        X = np.array(pwm_triples, dtype=float)
        return self._pipeline.predict(X).tolist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle the model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FlowModel":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a FlowModel: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Convenience: train from CSV without saving
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, csv_path: str | Path, degree: int = 2, alpha: float = 1e-3) -> "FlowModel":
        """Shortcut for controllers that want to train on the fly."""
        return cls.train(csv_path, degree=degree, alpha=alpha)

    def __repr__(self) -> str:
        mn = self.feature_range
        return (
            f"FlowModel(degree={self.degree}, "
            f"incr1=[{mn[0]:.0f},{mn[1]:.0f}], "
            f"incr2=[{mn[2]:.0f},{mn[3]:.0f}], "
            f"incr3=[{mn[4]:.0f},{mn[5]:.0f}])"
        )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(model: FlowModel, csv_path: str | Path) -> None:
    """Print train-set metrics and a sample of predictions vs actual."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    X = df[_INCR_COLS].values.astype(float)
    y_true = df[_FLOW_COL].values.astype(float)
    y_pred = np.array(model.predict_batch(X.tolist()))

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  MAE  : {mae:.4f} L/min")
    print(f"  R²   : {r2:.4f}")
    print(f"  Rows : {len(y_true)}")

    print("\n  Sample predictions (first 10 rows):")
    print(f"  {'incr1':>6} {'incr2':>6} {'incr3':>6}  {'actual':>8}  {'predicted':>10}  {'error':>8}")
    for i in range(min(10, len(y_true))):
        err = y_pred[i] - y_true[i]
        print(f"  {X[i,0]:>6.0f} {X[i,1]:>6.0f} {X[i,2]:>6.0f}  "
              f"{y_true[i]:>8.3f}  {y_pred[i]:>10.3f}  {err:>+8.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or use the flow estimation model."
    )
    parser.add_argument("--train", "-t", default=True, help="train model")
    parser.add_argument("--save", default="flow_model.pkl", help="Path to save trained model (default: flow_model.pkl)")
    parser.add_argument("--load", default="", help="Path to load an existing model (skips training)")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Ridge regularisation alpha (default: 1e-3)")
    parser.add_argument(
        "--pwm", type=float, nargs=3, metavar=("INCR1", "INCR2", "INCR3"),
        help="Predict flow for a single PWM increment triple",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    return parser


def main() -> None: 
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(FILE_DIR)
    sys.path.insert(0, PARENT_DIR)
    args = build_arg_parser().parse_args()

    # ---- Load or train ----
    if args.load:
        print(f"Loading model from {args.load} ...")
        model = FlowModel.load(args.load)
        print(f"  {model}")
    elif args.train:
        csv_file = Path(PARENT_DIR) / "data" / "pwm_vs_pressure_calibration" / "average" / "Average_merged_11files.csv"
        print(f"Training on {csv_file}  (degree={args.degree}, alpha={args.alpha}) ...")
        model = FlowModel.train(csv_file, degree=args.degree, alpha=args.alpha)
        print(f"  {model}")
        if not args.no_eval:
            print("\nTrain-set metrics:")
            evaluate(model, csv_file)
        model.save(Path(FILE_DIR) / args.save)
    else:
        raise SystemExit("Provide --csv to train or --load to load an existing model.")

    # ---- Optional single prediction ----
    if args.pwm:
        flow = model.predict(*args.pwm)
        print(f"\nflow({args.pwm[0]:.0f}, {args.pwm[1]:.0f}, {args.pwm[2]:.0f}) = {flow:.4f} L/min")


if __name__ == "__main__":
    main()
