# Week 11 Class 3 — Leakage-Safe Preprocessing (Iris)

## What I did
Built a leakage-safe preprocessing pipeline in `preprocess_iris.py`:
- Split data FIRST (test_size=0.2, random_state=42, stratify=species) before any fitting
- Fit the preprocessor ONLY on the training set, then transformed test with the same fitted transformer
- Numeric features: median imputation + StandardScaler
- Categorical features: most-frequent imputation + OneHotEncoder(handle_unknown="ignore")
- Dropped `species` (label) and `sample_id` (ID) from features

## How to run
```bash
pip install -r requirements.txt
python preprocess_iris.py --input iris_preprocessing_case_PUBLIC_DERIVED.csv --output-dir outputs
python -m unittest test_student_smoke.py