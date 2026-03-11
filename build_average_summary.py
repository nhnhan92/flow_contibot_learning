import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


OUT_HEADERS = [
    "pwm_incr_module1",
    "pwm_incr_module2",
    "pwm_incr_module3",
    "pwm1_cur",
    "pwm2_cur",
    "pwm3_cur",
    "flow_Lmin_mean",
    "press_MPa_mean",
    "n_used",
    "n_pos_total",
    "n_removed",
    "n_rows_total",
]

DEFAULT_INPUTS = ["data/pwm_vs_pressure_calibration"]
PWM_COLS = ["pwm1_cur", "pwm2_cur", "pwm3_cur"]

# Default number of rows to skip at the start of each PWM group after a
# transition (settling time).  At the nominal 50 ms sampling interval this
# equals 0.5 s.  Adjust with --skip-rows if your sampling rate differs.
DEFAULT_SKIP_ROWS = 10


def top_n_select(pairs: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    """Return the top *n* (flow, press) pairs ranked by flow (highest first).

    Selecting by flow rank ensures both values come from the same steady-state
    readings (valve fully open), giving physically consistent paired averages.
    If len(pairs) <= n, all pairs are returned.
    """
    return sorted(pairs, key=lambda p: p[0], reverse=True)[:n]


def collect_input_files(
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
            iterator: Iterable[Path] = p.rglob(pattern) if recursive else p.glob(pattern)
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


def get_flow_column(df: object) -> str:
    if "flow_Lmin" in df.columns:
        return "flow_Lmin"
    if "flow_lpm" in df.columns:
        return "flow_lpm"
    raise ValueError("Required flow column not found. Expected flow_Lmin (or flow_lpm).")


def get_press_column(df: object) -> Optional[str]:
    """Return the pressure column name, or None if not present."""
    if "press_MPa" in df.columns:
        return "press_MPa"
    if "pressure_MPa" in df.columns:
        return "pressure_MPa"
    return None


def ensure_stats_key(stats: Dict[Tuple[int, int, int], dict], key: Tuple[int, int, int]) -> dict:
    if key not in stats:
        stats[key] = {
            "pos_pairs": [],   # list of (flow, press) tuples from positive-flow rows
            "n_rows_total": 0,
        }
    return stats[key]


def apply_common_filters(d: object, keep_all_zero_pwm: bool) -> object:
    """Drop rows with negative PWM values and optionally drop all-zero-PWM rows."""
    d = d[(d["pwm1_cur"] >= 0) & (d["pwm2_cur"] >= 0) & (d["pwm3_cur"] >= 0)]

    if not keep_all_zero_pwm:
        d = d[~((d["pwm1_cur"] == 0) & (d["pwm2_cur"] == 0) & (d["pwm3_cur"] == 0))]

    return d


def add_settling_mask(d: object, skip_rows: int) -> object:
    """Remove the first *skip_rows* rows of each PWM group after a transition.

    A transition is detected when any of the three PWM values changes from one
    row to the next, OR when the Arduino timer resets (t_ms decreases).  Both
    events indicate the start of a new steady-state window, and the initial
    readings during the transient/settling period are discarded.
    """
    if skip_rows <= 0:
        return d

    d = d.copy()
    row_idx_in_group: List[int] = []
    prev_pwm: tuple = None
    prev_t_ms: float = None
    idx = 0

    for _, row in d.iterrows():
        curr_pwm = (int(row["pwm1_cur"]), int(row["pwm2_cur"]), int(row["pwm3_cur"]))
        curr_t_ms = float(row["t_ms"])

        # Arduino reset (millis() wrapped or board restarted) or PWM changed
        reset = (prev_t_ms is not None) and (curr_t_ms < prev_t_ms)
        if curr_pwm != prev_pwm or reset:
            idx = 0
            prev_pwm = curr_pwm
        else:
            idx += 1

        row_idx_in_group.append(idx)
        prev_t_ms = curr_t_ms

    d["_row_in_group"] = row_idx_in_group
    d = d[d["_row_in_group"] >= skip_rows].drop(columns=["_row_in_group"])
    return d


def accumulate_raw_file(
    file_path: Path,
    stats: Dict[Tuple[int, int, int], dict],
    positive_threshold: float,
    keep_all_zero_pwm: bool,
    skip_rows: int,
) -> int:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if df.empty:
        return 0

    missing = [c for c in PWM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required PWM column(s): {', '.join(missing)}")

    flow_col = get_flow_column(df)
    press_col = get_press_column(df)

    # t_ms may be absent in some file variants; fall back to row index
    t_ms_series = (
        pd.to_numeric(df["t_ms"], errors="coerce")
        if "t_ms" in df.columns
        else pd.Series(range(len(df)), dtype=float)
    )

    col_dict = {
        "t_ms": t_ms_series,
        "pwm1_cur": pd.to_numeric(df["pwm1_cur"], errors="coerce"),
        "pwm2_cur": pd.to_numeric(df["pwm2_cur"], errors="coerce"),
        "pwm3_cur": pd.to_numeric(df["pwm3_cur"], errors="coerce"),
        "flow": pd.to_numeric(df[flow_col], errors="coerce"),
    }
    if press_col:
        col_dict["press"] = pd.to_numeric(df[press_col], errors="coerce")

    d = pd.DataFrame(col_dict)
    dropna_cols = ["pwm1_cur", "pwm2_cur", "pwm3_cur", "flow"]
    d = d.dropna(subset=dropna_cols).copy()
    d["pwm1_cur"] = d["pwm1_cur"].round().astype(int)
    d["pwm2_cur"] = d["pwm2_cur"].round().astype(int)
    d["pwm3_cur"] = d["pwm3_cur"].round().astype(int)
    if press_col:
        d["press"] = d["press"].fillna(0.0)

    d = apply_common_filters(d, keep_all_zero_pwm)
    if d.empty:
        return 0

    # Discard transient readings at the start of each PWM group
    d = add_settling_mask(d, skip_rows)
    if d.empty:
        return 0

    # Count total (post-settling) rows per PWM combination
    rows_total = d.groupby(PWM_COLS).size()
    for key, n_rows in rows_total.items():
        g = ensure_stats_key(stats, tuple(int(v) for v in key))
        g["n_rows_total"] += int(n_rows)

    # Collect (flow, press) pairs from positive-flow rows
    d_pos = d[d["flow"] > positive_threshold]
    if not d_pos.empty:
        has_press = "press" in d_pos.columns
        for key, grp in d_pos.groupby(PWM_COLS):
            g = ensure_stats_key(stats, tuple(int(v) for v in key))
            flows = grp["flow"].astype(float).tolist()
            if has_press:
                presses = grp["press"].astype(float).tolist()
            else:
                presses = [0.0] * len(flows)
            g["pos_pairs"].extend(zip(flows, presses))

    return int(len(d))


def build_output_rows(
    stats: Dict[Tuple[int, int, int], dict],
    base1: int,
    base2: int,
    base3: int,
    top_n: int,
    min_used: int,
    allow_negative_increments: bool,
) -> List[List[float]]:
    rows: List[List[float]] = []

    for (pwm1, pwm2, pwm3), g in stats.items():
        pairs = list(g["pos_pairs"])
        selected = top_n_select(pairs, top_n)

        n_pos_total = len(pairs)
        n_used = len(selected)
        if n_used < min_used:
            continue

        flow_mean = sum(p[0] for p in selected) / n_used
        press_mean = sum(p[1] for p in selected) / n_used
        n_removed = n_pos_total - n_used
        n_rows_total = int(g["n_rows_total"])

        incr1 = pwm1 - base1
        incr2 = pwm2 - base2
        incr3 = pwm3 - base3
        if (not allow_negative_increments) and (incr1 < 0 or incr2 < 0 or incr3 < 0):
            continue

        rows.append([
            incr1, incr2, incr3,
            pwm1, pwm2, pwm3,
            flow_mean, press_mean,
            n_used, n_pos_total, n_removed, n_rows_total,
        ])

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
    return rows


def fill_symmetric_rows(
    rows: List[List[float]],
    base2: int,
    base3: int,
) -> List[List[float]]:
    """Mirror each row by swapping module2 and module3.

    Column layout (OUT_HEADERS):
      [0] incr1  [1] incr2  [2] incr3
      [3] pwm1   [4] pwm2   [5] pwm3
      [6] flow   [7] press
      [8] n_used [9] n_pos_total [10] n_removed [11] n_rows_total
    """
    existing = {(int(r[0]), int(r[1]), int(r[2])) for r in rows}
    mirrored: List[List[float]] = []
    for r in rows:
        m1, m2, m3 = int(r[0]), int(r[1]), int(r[2])
        if m2 != m3 and (m1, m3, m2) not in existing:
            new_pwm2 = int(r[2]) + base2
            new_pwm3 = int(r[1]) + base3
            mirrored.append([
                r[0], r[2], r[1],
                r[3], new_pwm2, new_pwm3,
                r[6], r[7], r[8], r[9], r[10], r[11],
            ])
            existing.add((m1, m3, m2))
    combined = rows + mirrored
    combined.sort(key=lambda r: (r[0], r[1], r[2]))
    return combined


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build average flow+pressure summary from raw PWM calibration CSVs."
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="Input file(s) or directory path(s). Default: data/pwm_vs_pressure_calibration",
    )
    parser.add_argument("--pattern", default="*.csv", help="Glob for directory inputs (default: *.csv)")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    parser.add_argument(
        "--include-timestamp", action="store_true",
        help="Include *_timestamp.csv when scanning directories",
    )
    parser.add_argument(
        "--include-derived", action="store_true",
        help="Include derived files such as Average_*.csv (default: excluded)",
    )
    parser.add_argument(
        "--output-dir", default="data/pwm_vs_pressure_calibration/average",
        help="Output directory",
    )
    parser.add_argument("--output-name", default="", help="Output stem without extension")
    parser.add_argument(
        "--base1", type=int, default=148,
        help="Base (idle) PWM value for module 1; subtracted to give the increment column",
    )
    parser.add_argument("--base2", type=int, default=151, help="Base PWM value for module 2")
    parser.add_argument("--base3", type=int, default=148, help="Base PWM value for module 3")
    parser.add_argument(
        "--positive-threshold", type=float, default=0.0,
        help="Keep only rows where flow > threshold (default: 0.0, removes exact-zero readings)",
    )
    parser.add_argument(
        "--skip-rows", type=int, default=DEFAULT_SKIP_ROWS,
        help=(
            "Number of rows to discard at the start of each PWM group after a transition "
            "(settling time). At 50 ms sampling, 10 rows ≈ 0.5 s. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help=(
            "Number of highest-flow readings to average per PWM combination. "
            "Both flow and the corresponding pressure at those same moments are averaged. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--min-used", type=int, default=1,
        help=(
            "Minimum number of positive readings required to include a PWM point. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--keep-all-zero-pwm", action="store_true",
        help="Keep rows where pwm1=pwm2=pwm3=0 (idle baseline, default: drop)",
    )
    parser.add_argument(
        "--allow-negative-increments", action="store_true",
        help="Allow negative pwm_incr_module values in the output (default: filtered out)",
    )
    parser.add_argument(
        "--fill-symmetric", action="store_true",
        help=(
            "Mirror each measured point by swapping module2 and module3, "
            "filling the other half of the PWM space. "
            "flow(m1,m2,m3) == flow(m1,m3,m2) is assumed."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if pd is None:
        raise SystemExit("pandas is required.  Install with: pip install pandas")

    args.inputs = args.inputs or list(DEFAULT_INPUTS)
    input_files = collect_input_files(
        inputs=args.inputs,
        pattern=args.pattern,
        recursive=args.recursive,
        include_timestamp=args.include_timestamp,
        include_derived=args.include_derived,
    )
    if not input_files:
        raise SystemExit("No input CSV files found.")

    stats: Dict[Tuple[int, int, int], dict] = {}
    total_rows_used = 0

    for fpath in input_files:
        try:
            used = accumulate_raw_file(
                file_path=fpath,
                stats=stats,
                positive_threshold=args.positive_threshold,
                keep_all_zero_pwm=args.keep_all_zero_pwm,
                skip_rows=args.skip_rows,
            )
            total_rows_used += used
            print(f"[OK] {fpath} -> used rows: {used}")
        except Exception as e:
            print(f"[WARN] Failed to process {fpath}: {e}")

    if not stats:
        raise SystemExit("No valid rows found after parsing/filtering.")

    rows = build_output_rows(
        stats=stats,
        base1=args.base1,
        base2=args.base2,
        base3=args.base3,
        top_n=args.top_n,
        min_used=args.min_used,
        allow_negative_increments=args.allow_negative_increments,
    )

    measured_count = len(rows)
    if args.fill_symmetric:
        rows = fill_symmetric_rows(rows, base2=args.base2, base3=args.base3)

    output_dir = Path(args.output_dir)
    if args.output_name:
        stem = args.output_name
    elif len(input_files) == 1:
        stem = f"Average_{input_files[0].stem}"
    else:
        stem = f"Average_merged_{len(input_files)}files"

    csv_path = output_dir / f"{stem}.csv"
    write_csv(csv_path, OUT_HEADERS, rows)

    print("\nDone.")
    print(f"Input files   : {len(input_files)}")
    print(f"Rows used     : {total_rows_used}")
    print(f"Measured pts  : {measured_count}")
    if args.fill_symmetric:
        print(f"After mirror  : {len(rows)}  (+{len(rows) - measured_count} symmetric)")
    print(f"CSV output    : {csv_path}")


if __name__ == "__main__":
    main()
