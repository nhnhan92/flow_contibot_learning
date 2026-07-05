"""
thin_dataset.py — Collapse redundant rows in flow-tip collection CSVs.

Problem: data is logged at ~20 Hz for 1 s per waypoint → 18-20 nearly
identical rows per unique PWM command.  This inflates the dataset without
adding information and causes val loss to be artificially low (val samples
are near-duplicates of train samples after random shuffling).

Fix: for each *consecutive run* of the same (pwm1_cmd, pwm2_cmd, pwm3_cmd)
triplet, collapse all rows into a single row by:
  - Averaging  proc_flow1/2/3 and opti_x/y/z_mm over the *second half*
    of the run (settled portion, ignoring the initial transient).
  - Keeping the first-row value for pwm commands, state, timestamps, etc.

Usage
-----
    # Preview (no files written)
    python flowbot/proprioception_model/thin_dataset.py \\
        --data_dir data/flow_tip_free_100g --dry_run

    # Write averaged copies to a new folder (safe default)
    python flowbot/proprioception_model/thin_dataset.py \\
        --data_dir data/flow_tip_free_100g \\
        --out_dir  data/flow_tip_free_100g_avg

    # Overwrite originals in-place (originals backed up to _orig/)
    python flowbot/proprioception_model/thin_dataset.py \\
        --data_dir data/flow_tip_free_100g --inplace
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

_PWM_COLS  = ["pwm1_cmd", "pwm2_cmd", "pwm3_cmd"]
_FLOW_COLS = ["proc_flow1", "proc_flow2", "proc_flow3"]
_OPTI_COLS = ["opti_x_mm", "opti_y_mm", "opti_z_mm"]
_AVG_COLS  = _FLOW_COLS + _OPTI_COLS   # columns to average over the settled window


# ── Core averaging logic ──────────────────────────────────────────────────────

def _average_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive runs of identical PWM triplets into one row each.

    For each run:
      - Averaged over the second half (settled portion):
          proc_flow1/2/3, opti_x/y/z_mm
      - Taken from the first row (same value for all rows in run):
          pwm1/2/3_cmd, state, current_time, t_s, and any other columns
    """
    if not all(c in df.columns for c in _PWM_COLS):
        return df   # nothing to do if PWM columns missing

    pwm = df[_PWM_COLS].values.astype(int)

    # Assign a run-ID that increments whenever the PWM triplet changes
    run_ids = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        run_ids[i] = run_ids[i - 1] + int(not np.array_equal(pwm[i], pwm[i - 1]))

    df = df.copy()
    df["_run_id"] = run_ids

    rows = []
    avg_cols_present = [c for c in _AVG_COLS if c in df.columns]

    for _, run in df.groupby("_run_id", sort=False):
        # Base row: take first row for all non-averaged columns
        base = run.iloc[0].copy()

        # Average the sensor/position columns over the settled second half
        settled = run.iloc[len(run) // 2 :]
        for col in avg_cols_present:
            base[col] = float(settled[col].mean())

        rows.append(base)

    return (
        pd.DataFrame(rows)
        .drop(columns=["_run_id"])
        .reset_index(drop=True)
    )


# ── File-level helpers ────────────────────────────────────────────────────────

def _process_file(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    """Read src, average it, write to dst. Returns (original_rows, kept_rows)."""
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f"  WARNING: could not read {src.name}: {e}")
        return 0, 0

    original_rows = len(df)
    averaged      = _average_df(df)
    kept_rows     = len(averaged)

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        averaged.to_csv(dst, index=False)

    return original_rows, kept_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Average flow-tip CSVs: collapse each consecutive PWM run "
                    "into one row (mean of flow + opti over the settled second half)."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Folder containing source CSV files (searched recursively)")
    parser.add_argument("--out_dir",  default=None,
                        help="Output folder (default: <data_dir>_avg).  "
                             "Ignored when --inplace is set.")
    parser.add_argument("--inplace",  action="store_true",
                        help="Overwrite original files.  Originals are backed up to "
                             "<data_dir>_orig/ before modification.")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Print statistics only — do not write any files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist.")
        return

    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        print(f"No CSV files found under {data_dir}")
        return

    # Determine output directory
    if args.inplace:
        backup_dir = data_dir.parent / (data_dir.name + "_orig")
        if not args.dry_run:
            if backup_dir.exists():
                print(f"[avg] Backup dir already exists: {backup_dir} — skipping backup.")
            else:
                shutil.copytree(data_dir, backup_dir)
                print(f"[avg] Originals backed up to {backup_dir}")
        out_dir = data_dir
    else:
        out_dir = Path(args.out_dir) if args.out_dir else \
                  data_dir.parent / (data_dir.name + "_avg")

    mode_str = "DRY RUN — " if args.dry_run else ""
    print(f"[avg] {mode_str}source={data_dir}   "
          f"dest={out_dir if not args.inplace else '(in-place)'}")
    print(f"[avg] Found {len(csvs)} CSV files\n")

    total_before = 0
    total_after  = 0

    for src in csvs:
        rel  = src.relative_to(data_dir)
        dst  = out_dir / rel
        before, after = _process_file(src, dst, args.dry_run)
        removed = before - after
        pct     = 100.0 * removed / before if before else 0.0
        print(f"  {src.name:<50s}  {before:>5d} → {after:>4d} rows  "
              f"(-{removed:>4d}, {pct:.0f}%)")
        total_before += before
        total_after  += after

    total_removed = total_before - total_after
    print(f"\n[thin] Total: {total_before:,} → {total_after:,} rows  "
          f"(-{total_removed:,}, {100.0*total_removed/total_before:.0f}%)")

    if args.dry_run:
        print("[thin] Dry run complete — no files written.")
    elif args.inplace:
        print(f"[thin] Files overwritten in {data_dir}  (originals in {backup_dir})")
    else:
        print(f"[thin] Thinned files written to {out_dir}")


if __name__ == "__main__":
    main()
