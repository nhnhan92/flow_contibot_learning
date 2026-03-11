import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


DEFAULT_INPUTS_FALLBACK = ["data/pwm_vs_pressure_calibration/average"]
DEFAULT_AVERAGE_DIR = Path("data/pwm_vs_pressure_calibration/average")
DEFAULT_MERGED_PATTERN = "Average_*.csv"

MODULE_COLS = ["module1", "module2", "module3"]


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    index = {normalize_name(col): col for col in columns}
    for cand in candidates:
        if cand in index:
            return index[cand]
    return ""


def resolve_default_inputs() -> List[str]:
    if DEFAULT_AVERAGE_DIR.is_dir():
        candidates = [p for p in DEFAULT_AVERAGE_DIR.glob(DEFAULT_MERGED_PATTERN) if p.is_file()]
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            return [str(latest)]
    # Fall back to the average directory itself so the user gets a clear error
    # rather than a silent empty result when no Average_*.csv exists yet.
    return list(DEFAULT_INPUTS_FALLBACK)


def collect_input_csv_files(
    inputs: Sequence[str],
    pattern: str,
    recursive: bool,
    include_timestamp: bool,
    include_derived: bool,
) -> List[Path]:
    files: List[Path] = []
    seen = set()

    for item in inputs:
        p = Path(item)

        if p.is_file() and p.suffix.lower() == ".csv":
            name_lower = p.name.lower()
            if (not include_derived) and name_lower.startswith("average_"):
                continue
            if include_timestamp or ("timestamp" not in name_lower):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(p)
            continue

        if p.is_dir():
            iterator = p.rglob(pattern) if recursive else p.glob(pattern)
            for f in iterator:
                if (not f.is_file()) or f.suffix.lower() != ".csv":
                    continue
                name_lower = f.name.lower()
                if (not include_derived) and name_lower.startswith("average_"):
                    continue
                if (not include_timestamp) and ("timestamp" in name_lower):
                    continue
                key = str(f.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(f)

    files.sort()
    return files


def load_contributions(
    file_path: Path,
    positive_threshold: float,
    keep_all_zero_pwm: bool,
    base1: int,
    base2: int,
    base3: int,
) -> Tuple[object, str, int]:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame(columns=MODULE_COLS + ["sum_flow", "n_used"]), "unknown", 0

    pwm1_col = find_column(df.columns, ["pwm1cur", "pwm1"])
    pwm2_col = find_column(df.columns, ["pwm2cur", "pwm2"])
    pwm3_col = find_column(df.columns, ["pwm3cur", "pwm3"])
    if not pwm1_col or not pwm2_col or not pwm3_col:
        raise ValueError("Required PWM columns (pwm1_cur / pwm2_cur / pwm3_cur) not found.")

    d = pd.DataFrame(
        {
            "pwm1_cur": pd.to_numeric(df[pwm1_col], errors="coerce"),
            "pwm2_cur": pd.to_numeric(df[pwm2_col], errors="coerce"),
            "pwm3_cur": pd.to_numeric(df[pwm3_col], errors="coerce"),
        }
    )
    d = d.dropna(subset=["pwm1_cur", "pwm2_cur", "pwm3_cur"]).copy()
    d["pwm1_cur"] = d["pwm1_cur"].round().astype(int)
    d["pwm2_cur"] = d["pwm2_cur"].round().astype(int)
    d["pwm3_cur"] = d["pwm3_cur"].round().astype(int)

    d = d[(d["pwm1_cur"] >= 0) & (d["pwm2_cur"] >= 0) & (d["pwm3_cur"] >= 0)]
    if not keep_all_zero_pwm:
        d = d[~((d["pwm1_cur"] == 0) & (d["pwm2_cur"] == 0) & (d["pwm3_cur"] == 0))]

    # Module increment = absolute PWM - hardware base.
    # Works for both summary and raw CSVs: pwm1_cur always holds the absolute value.
    d["module1"] = d["pwm1_cur"] - base1
    d["module2"] = d["pwm2_cur"] - base2
    d["module3"] = d["pwm3_cur"] - base3

    summary_flow_col = find_column(df.columns, ["flowlminmean"])
    if summary_flow_col:
        mode = "summary"
        d["flow"] = pd.to_numeric(df.loc[d.index, summary_flow_col], errors="coerce").fillna(0.0)
        n_used_col = find_column(df.columns, ["nused"])
        if n_used_col:
            d["n_used"] = pd.to_numeric(df.loc[d.index, n_used_col], errors="coerce").fillna(0.0)
        else:
            d["n_used"] = 1.0
        d = d[d["n_used"] > 0]
        d["sum_flow"] = d["flow"] * d["n_used"]
    else:
        raw_flow_col = find_column(df.columns, ["flowlmin", "flowlpm"])
        if not raw_flow_col:
            raise ValueError("Flow column not found. Expected flow_Lmin or flow_Lmin_mean.")
        mode = "raw"
        d["flow"] = pd.to_numeric(df.loc[d.index, raw_flow_col], errors="coerce").fillna(0.0)
        d = d[d["flow"] > positive_threshold]
        d["n_used"] = 1.0
        d["sum_flow"] = d["flow"]

    grouped = (
        d.groupby(MODULE_COLS, as_index=False)[["sum_flow", "n_used"]]
        .sum()
        .reset_index(drop=True)
    )
    return grouped, mode, int(len(d))


def map_sizes(values: List[int], size_min: float, size_max: float) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return [(size_min + size_max) * 0.5 for _ in values]
    return [size_min + (v - vmin) * (size_max - size_min) / (vmax - vmin) for v in values]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot 3D scatter from summary/raw CSV files.")
    parser.add_argument("inputs", nargs="*", help="Input CSV file(s) or directory path(s).")
    parser.add_argument("--pattern", default="*.csv", help="Glob for directory inputs (default: *.csv)")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    parser.add_argument("--include-timestamp", action="store_true", help="Include *_timestamp.csv")
    parser.add_argument("--include-derived", action="store_true", help="Include Average_*.csv (default: excluded)")
    parser.add_argument("--output", default="", help="Output PNG path")
    parser.add_argument("--min-used", type=int, default=3, help="Keep only points with n_used >= this value")
    parser.add_argument("--positive-threshold", type=float, default=0.0, help="Raw mode: keep flow > threshold")
    parser.add_argument(
        "--base1", type=int, default=148,
        help="Hardware base PWM for module 1; subtracted from pwm1_cur to give the command increment (default: 148)",
    )
    parser.add_argument("--base2", type=int, default=151, help="Hardware base PWM for module 2 (default: 151)")
    parser.add_argument("--base3", type=int, default=148, help="Hardware base PWM for module 3 (default: 148)")
    parser.add_argument(
        "--allow-negative-increments",
        action="store_true",
        help="Allow points where module increment is negative (default: filtered out)",
    )
    parser.add_argument("--keep-all-zero-pwm", action="store_true", help="Keep rows where pwm1=pwm2=pwm3=0")
    parser.add_argument("--size-min", type=float, default=10.0, help="Minimum marker size")
    parser.add_argument("--size-max", type=float, default=60.0, help="Maximum marker size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Marker alpha")
    parser.add_argument("--dpi", type=int, default=160, help="Save DPI")
    parser.add_argument("--title", default="", help="Custom plot title")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if pd is None:
        raise SystemExit("pandas is required. Install it with: pip install pandas")

    if not args.inputs:
        args.inputs = resolve_default_inputs()
        if len(args.inputs) == 1 and Path(args.inputs[0]).name.startswith("Average_"):
            args.include_derived = True
            print(f"[INFO] Using default summary: {args.inputs[0]}")
        else:
            print("[INFO] No Average_*.csv found in default directory. Falling back to default input path.")

    # Auto-enable include_derived when the user explicitly points at an Average_*
    # file or a directory whose CSVs are all derived (average) outputs.
    if not args.include_derived:
        for item in args.inputs:
            p = Path(item)
            if p.is_file() and p.name.lower().startswith("average_"):
                args.include_derived = True
                break
            if p.is_dir() and any(f.name.lower().startswith("average_") for f in p.glob("*.csv")):
                args.include_derived = True
                break

    input_files = collect_input_csv_files(
        inputs=args.inputs,
        pattern=args.pattern,
        recursive=args.recursive,
        include_timestamp=args.include_timestamp,
        include_derived=args.include_derived,
    )
    if not input_files:
        raise SystemExit("No input CSV files found.")

    contributions: List[object] = []
    for fpath in input_files:
        try:
            grouped, mode, used_rows = load_contributions(
                file_path=fpath,
                positive_threshold=args.positive_threshold,
                keep_all_zero_pwm=args.keep_all_zero_pwm,
                base1=args.base1,
                base2=args.base2,
                base3=args.base3,
            )
            if not grouped.empty:
                contributions.append(grouped)
            print(f"[OK] {fpath} ({mode}) -> used rows: {used_rows}")
        except Exception as e:
            print(f"[WARN] Failed to process {fpath}: {e}")

    if not contributions:
        raise SystemExit("No rows to plot after filtering.")

    merged = pd.concat(contributions, ignore_index=True)
    merged = merged.groupby(MODULE_COLS, as_index=False)[["sum_flow", "n_used"]].sum()
    merged = merged[merged["n_used"] > 0].copy()
    merged["flow_Lmin_mean"] = merged["sum_flow"] / merged["n_used"]

    if not args.allow_negative_increments:
        merged = merged[
            (merged["module1"] >= 0)
            & (merged["module2"] >= 0)
            & (merged["module3"] >= 0)
        ]

    merged["n_used_int"] = merged["n_used"].round().astype(int)
    merged = merged[merged["n_used_int"] >= args.min_used]
    if merged.empty:
        raise SystemExit("No rows to plot after filtering.")

    x = merged["module1"].tolist()
    y = merged["module2"].tolist()
    z = merged["module3"].tolist()
    color = merged["flow_Lmin_mean"].tolist()
    sizes = map_sizes(merged["n_used_int"].tolist(), args.size_min, args.size_max)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=color, s=sizes, cmap="viridis", alpha=args.alpha, edgecolors="none")

    ax.set_xlabel("module_1 (increment)")
    ax.set_ylabel("module_2 (increment)")
    ax.set_zlabel("module_3 (increment)")
    ax.set_title(
        args.title
        if args.title
        else "3D scatter: x=module_1, y=module_2, z=module_3\nColor=Flow_Lmin_mean (L/min), Size=n_used"
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.08)
    cbar.set_label("Flow_Lmin_mean (L/min)")

    if args.output:
        output_path = Path(args.output)
    elif len(input_files) == 1:
        output_path = input_files[0].with_name(f"{input_files[0].stem}_3d_scatter.png")
    else:
        output_path = Path("flow_3d_scatter_merged.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved: {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
