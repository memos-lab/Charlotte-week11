"""Week 11 Class 2 - Outlier audit."""
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = ["sample_id", "alcohol", "malic_acid", "color_intensity", "proline", "target"]
FLAG_COLS = ["malic_acid", "proline"]

def _flags(series: pd.Series):
    """Return Z-flags (|Z|>3, ddof=0) and IQR-flags (1.5*IQR). Safe for zero-variance."""
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        z_flag = pd.Series(False, index=s.index)
    else:
        z = (s - mean) / std
        z_flag = z.abs() > 3
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        iqr_flag = pd.Series(False, index=s.index)
    else:
        iqr_flag = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    return z_flag.fillna(False), iqr_flag.fillna(False)

def audit_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return retained, quarantine and quality summary tables."""
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = data.copy()
    # Domain checks - ONLY these quarantine a row
    malic_invalid = df["malic_acid"].isna() | (pd.to_numeric(df["malic_acid"], errors="coerce") < 0)
    proline_invalid = pd.to_numeric(df["proline"], errors="coerce") <= 0
    df["domain_invalid"] = malic_invalid | proline_invalid

    summary_rows = []
    for col in FLAG_COLS:
        z_flag, iqr_flag = _flags(df[col])
        df[f"{col}_z_flag"] = z_flag
        df[f"{col}_iqr_flag"] = iqr_flag
        summary_rows.append({
            "measurement": col,
            "total_rows": len(df),
            "retained_rows": int((~df["domain_invalid"]).sum()),
            "quarantined_rows": int(df["domain_invalid"].sum()),
            "z_flag_count": int(z_flag.sum()),
            "iqr_flag_count": int(iqr_flag.sum()),
            "review_needed_count": int((z_flag | iqr_flag).sum()),
        })

    df["review_needed"] = df["malic_acid_z_flag"] | df["malic_acid_iqr_flag"] | df["proline_z_flag"] | df["proline_iqr_flag"]

    retained = df[~df["domain_invalid"]].copy()
    quarantined = df[df["domain_invalid"]].copy()
    summary = pd.DataFrame(summary_rows)
    return {"retained": retained, "quarantined": quarantined, "summary": summary}

if __name__ == "__main__":
    data = pd.read_csv(Path(__file__).with_name("wine_quality_case_PUBLIC_DERIVED.csv"))
    results = audit_data(data)
    results["retained"].to_csv(Path(__file__).with_name("retained_flagged.csv"), index=False)
    results["quarantined"].to_csv(Path(__file__).with_name("quarantined_records.csv"), index=False)
    results["summary"].to_csv(Path(__file__).with_name("quality_summary.csv"), index=False)
    print(results["summary"])
    print(f"Retained: {len(results['retained'])}, Quarantined: {len(results['quarantined'])}")