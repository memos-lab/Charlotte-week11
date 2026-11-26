# Week 11 Class 4 — Individual Submission

## Data sources & limitations

- **All data is synthetic** — no public / real well data was used.
- Sources: `lab/data/well_b_synthetic.las` (LAS well log, WELL-B) and `lab/data/scada_synthetic.csv` (SCADA time-series).
- Units assumed: Depth in **m**, GR in **API**, NPHI in **fraction (v/v)**, RHOB in **g/cc**, Temperature in **°C**, Pressure in **psi**, Flow in **bbl/day**.
- Limitation: Only 9 depth points (1000.0 – 1008.0 m), 1 well, no real geology. Results are for pipeline demonstration only, not for reservoir decisions.

## Running the project

```bash
# 1. Create + activate venv (Windows / Git Bash)
python -m venv .venv
source .venv/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt
# or if no requirements.txt:
pip install pandas

# 3. Run pipeline
python lab/feature_pipeline.py

# 4. Run tests
python -m unittest discover -v
# or:
python -m unittest lab.test_student_smoke -v