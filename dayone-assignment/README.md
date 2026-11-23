# Week 11 / Class 1 - Missing Data Lab

## Decisions
- Missingness report was generated from the source file BEFORE any filling.
- Added `lifeExp_was_missing` and `gdpPercap_was_missing` BEFORE filling, so imputed rows stay auditable.
- `continent` was filled only from observed rows for the same country. No guessing.
- `lifeExp` and `gdpPercap` were imputed with continent-level medians, with a global median fallback if a continent group is empty.
- Validation checks: 96 rows kept, no nulls in continent/lifeExp/gdpPercap, originally observed values unchanged.

## Risks / Limitation
Median imputation shrinks variance and makes imputed countries look more alike than they really are. It assumes countries in the same continent are similar, which is not always true, and it hides uncertainty - an imputed value looks like a real measurement unless you check the `_was_missing` flags. Treat imputed values as placeholders, not observed facts. The 17 blanks were deliberately created for class, so they do not reflect real-world missingness.

## Stretch: Excel vs CSV
Imported `gapminder_instructional_missing.xlsx` (Sheet1) and compared to CSV. Both have 96 rows, 6 columns, same column names, same dtypes, and same missing counts (continent: 3, lifeExp: 8, gdpPercap: 6, total 17). No duplicate country/year rows.

## Run
python missing_data_lab.py