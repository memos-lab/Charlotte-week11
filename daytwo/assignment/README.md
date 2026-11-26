# Week 11 Class 02 — Outlier Audit

## What this does
Audits 178 wine rows in `wine_quality_case_PUBLIC_DERIVED.csv`.
Flags `malic_acid` and `proline` for outliers, but only quarantines
rows that are domain-invalid. Raw values are never changed.

## Method
For each of `malic_acid` and `proline`, separately:
- **Z-score:** `|Z| > 3`, with `std(ddof=0)`
- **IQR:** outside `Q1 - 1.5*IQR` to `Q3 + 1.5*IQR`, using Pandas default quantiles

**Quarantine rule — domain only:**
- `proline <= 0` = invalid
- `malic_acid < 0` or missing = invalid

A statistical outlier alone is NOT quarantined. It stays in
`retained_flagged.csv` with `review_needed = True`.

## Outputs
- `retained_flagged.csv` — 177 rows, with flag columns, raw values preserved
- `quarantined_records.csv` — 1 row (proline = -45.00, domain-invalid)
- `quality_summary.csv` — per measurement: total, retained, quarantined,
  Z-flag count, IQR-flag count, review-needed count

Note: `malic_acid = 99.00` and `proline = 9999.00` are statistical
outliers but domain-valid, so they are retained with `review_needed = True`.

## Limitations
- Z-score assumes roughly symmetric data and is sensitive to the outliers themselves.
- IQR is robust but its 1.5x cutoff is a convention, not a physical law.
- A rare high reading can be legitimate — e.g. a genuinely high-proline
  wine variety, a different vintage, or a measurement in different units.
  That is why we flag for review instead of deleting.
- This audit does not prove a value is wrong, it only says "check this".

## How to run
```bash
pip install -r requirements.txt
python outlier_audit.py
python -m unittest test_outlier_audit.py

```

## Data source
Derived from the UCI Wine dataset, as provided via scikit-learn
(`sklearn.datasets.load_wine`). See `wine_quality_case_PUBLIC_DERIVED.csv`.