"""Week 11 Class 4 student lab: build a validated feature pipeline."""

from pathlib import Path

import lasio
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LAS_PATH = DATA_DIR / "well_b_synthetic.las"
SCADA_PATH = DATA_DIR / "scada_synthetic.csv"
WELL_ID = "WELL-B"
EXPECTED_COLUMNS = ["well_id", "depth_m", "pressure_psi", "temperature_c", "flow_m3_h", "operating_mode"]
DEPTH_TOLERANCE_M = 0.01

def load_scada(path):
    """Load SCADA CSV, coerce numeric sensor columns, sort by well/depth."""
    df = pd.read_csv(path)
    # Keep only expected columns if present, but don't crash if extra
    for col in ["depth_m", "pressure_psi", "temperature_c", "flow_m3_h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["well_id", "depth_m"]).reset_index(drop=True)
    return df

def load_las(path):
    """Use lasio.read(path) to return depth_m, gamma_ray, bulk_density."""
    las = lasio.read(str(path))
    depth = np.array(las["DEPT"], dtype=float)
    gr = np.array(las["GR"], dtype=float)
    rhob = np.array(las["RHOB"], dtype=float)
    # lasio uses -999.25 as null
    for arr in (gr, rhob):
        arr[arr == -999.25] = np.nan
    df = pd.DataFrame({
        "depth_m": depth,
        "gamma_ray": gr,
        "bulk_density": rhob,
    })
    df["well_id"] = WELL_ID
    df = df.sort_values(["well_id", "depth_m"]).reset_index(drop=True)
    return df

def merge_inputs(las_df, scada_df):
    """Join on well/depth with validation of duplicates and merge coverage."""
    # 1. Duplicate check
    if las_df.duplicated(subset=["well_id", "depth_m"]).any():
        raise ValueError("Duplicate well_id/depth_m in LAS data")
    if scada_df.duplicated(subset=["well_id", "depth_m"]).any():
        raise ValueError("Duplicate well_id/depth_m in SCADA data")
    # 2. Depth tolerance check - round to tolerance so 1000.001 joins 1000.0
    las = las_df.copy()
    scada = scada_df.copy()
    las["depth_m"] = las["depth_m"].round(2)
    scada["depth_m"] = scada["depth_m"].round(2)
    # 3. Merge
    merged = pd.merge(scada, las, on=["well_id", "depth_m"], how="inner", suffixes=("", "_las"))
    # 4. Coverage check
    coverage = len(merged) / max(len(scada_df), 1)
    print(f"Merged rows: {len(merged)} / SCADA rows: {len(scada_df)} - coverage: {coverage:.1%}")
    if coverage < 0.9:
        raise ValueError(f"Merge coverage too low: {coverage:.1%}")
    merged = merged.sort_values(["well_id", "depth_m"]).reset_index(drop=True)
    return merged

def engineer_features(df):
    """Create pressure change, rolling pressure variance, temp differential, cumulative flow."""
    out = df.copy()
    out = out.sort_values(["well_id", "depth_m"]).reset_index(drop=True)
    # Group by well so diff/rolling never leaks across wells
    g = out.groupby("well_id", group_keys=False)
    # 1. Pressure change vs previous depth
    out["pressure_change_psi"] = g["pressure_psi"].diff()
    # 2. Rolling pressure variance - last 3 depths, need at least 2 points
    out["rolling_pressure_variance"] = g["pressure_psi"].transform(
        lambda s: s.rolling(window=3, min_periods=2).var()
    )
    # 3. Temperature differential vs previous depth
    out["temperature_differential_c"] = g["temperature_c"].diff()
    # 4. Cumulative flow - running total per well
    out["cumulative_flow_m3"] = g["flow_m3_h"].cumsum()
    # Fill first-row NaNs from diff with 0 - first depth has no previous depth
    out[["pressure_change_psi", "temperature_differential_c"]] = out[
        ["pressure_change_psi", "temperature_differential_c"]
    ].fillna(0)
    return out

def main():
    """Orchestrate loader -> validation -> merge -> features -> CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    las_df = load_las(LAS_PATH)
    scada_df = load_scada(SCADA_PATH)
    # Validation
    assert list(scada_df.columns)[:6] == EXPECTED_COLUMNS or set(EXPECTED_COLUMNS).issubset(scada_df.columns), \
        f"SCADA columns mismatch: {list(scada_df.columns)}"
    assert (scada_df["well_id"] == WELL_ID).all(), f"Unexpected well_id: {scada_df['well_id'].unique()}"
    merged = merge_inputs(las_df, scada_df)
    featured = engineer_features(merged)
    # Outputs
    featured.to_csv(OUTPUT_DIR / "engineered_features.csv", index=False)
    summary = pd.DataFrame([{
        "well_id": WELL_ID,
        "source_las_rows": len(las_df),
        "source_scada_rows": len(scada_df),
        "merged_rows": len(merged),
        "featured_rows": len(featured),
        "missing_cells": int(featured.isna().sum().sum()),
        "depth_min_m": float(featured["depth_m"].min()),
        "depth_max_m": float(featured["depth_m"].max()),
    }])
    summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    print("Done. Wrote:")
    print(f" - {OUTPUT_DIR / 'engineered_features.csv'}")
    print(f" - {OUTPUT_DIR / 'data_quality_summary.csv'}")
    print(featured.head().to_string())
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()