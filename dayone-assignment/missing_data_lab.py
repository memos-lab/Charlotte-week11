"""Week 11 Class 1 — independent missing-data lab.

Run from this folder:

    python missing_data_lab.py

Fill TODO sections. Do not paste the teaching notebook solution.
"""

from pathlib import Path
import pandas as pd

DATA = Path(__file__).with_name("gapminder_instructional_missing.csv")

OUTPUT = Path(__file__).with_name("outputs")

REQUIRED = {"country", "continent", "year", "lifeExp", "pop", "gdpPercap"}

def load_and_validate(path: Path) -> pd.DataFrame:
    """Read source, check schema, unique country/year and positive population."""
    df = pd.read_csv(path)

    # Schema check
    missing_cols = REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # One row per country/year
    if df.duplicated(subset=["country", "year"]).any():
        raise ValueError("Duplicate country/year rows found. Expected one row per country/year.")

    # Positive population
    if (df["pop"] <= 0).any():
        raise ValueError("pop must be positive for all rows.")

    return df

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return column, missing_count, missing_pct for every source column."""
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
    })

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve original observed values; mark and fill only specified missing cells.

    - continent: derive missing label from other rows of same country
    - lifeExp: continent-level median, fallback global median
    - gdpPercap: continent-level median, fallback global median
    - add lifeExp_was_missing and gdpPercap_was_missing boolean columns
    """
    # Operate on a copy; NEVER replace all missing values with zero
    out = df.copy()

    # Flags BEFORE any filling
    out["lifeExp_was_missing"] = out["lifeExp"].isna()
    out["gdpPercap_was_missing"] = out["gdpPercap"].isna()

    # continent: from observed rows of the same country
    country_continent = (
        out.dropna(subset=["continent"])
        .drop_duplicates(subset=["country"])
        .set_index("country")["continent"]
    )
    out["continent"] = out["continent"].fillna(out["country"].map(country_continent))

    # lifeExp + gdpPercap: continent median, global median fallback
    for col in ["lifeExp", "gdpPercap"]:
        global_median = out[col].median()
        continent_median = out.groupby("continent")[col].transform("median")
        out[col] = out[col].fillna(continent_median).fillna(global_median)

    return out

def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

    source = load_and_validate(DATA)

    report = missingness_report(source)
    print(report.to_string(index=False))

    cleaned = impute_data(source)

    # Verify required fields contain no NA, original observed values unchanged
    for col in ["continent", "lifeExp", "gdpPercap"]:
        if cleaned[col].isna().any():
            raise ValueError(f"Unresolved missing values in {col}")

    if len(source) != len(cleaned):
        raise ValueError(f"Row count changed: {len(source)} -> {len(cleaned)}")

    for col in ["lifeExp", "gdpPercap"]:
        mask = source[col].notna()
        if not (source.loc[mask, col] == cleaned.loc[mask, col]).all():
            raise ValueError(f"Originally observed values in {col} changed!")

    report.to_csv(OUTPUT / "missingness_report.csv", index=False)

    cleaned.to_csv(OUTPUT / "imputed_gapminder.csv", index=False)

    print("Created reports in", OUTPUT)

if __name__ == "__main__":
    main()