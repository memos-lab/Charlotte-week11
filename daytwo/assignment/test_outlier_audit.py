"""Tests for outlier audit."""
import unittest
import pandas as pd
from outlier_audit import audit_data

def _base():
    return pd.DataFrame({
        "sample_id": range(1, 11),
        "alcohol": [13.0]*10,
        "malic_acid": [2.0]*10,
        "color_intensity": [5.0]*10,
        "proline": [700.0]*10,
        "target": [0]*10,
    })

class TestOutlierAudit(unittest.TestCase):
    def test_missing_column_raises(self):
        df = _base().drop(columns=["proline"])
        with self.assertRaises(ValueError):
            audit_data(df)

    def test_zero_variance_does_not_crash_or_flag(self):
        # All same values: std=0, IQR=0 -> no flags, no crash
        result = audit_data(_base())
        self.assertEqual(len(result["retained"]), 10)
        self.assertFalse(result["retained"]["review_needed"].any())

    def test_domain_invalid_quarantined_only(self):
        df = _base()
        df.loc[0, "proline"] = -45.0  # invalid
        result = audit_data(df)
        self.assertEqual(len(result["quarantined"]), 1)
        self.assertEqual(result["quarantined"].iloc[0]["sample_id"], 1)
    def test_statistical_outlier_is_retained_with_review(self):
        df = _base()
        # Give malic_acid some real variation first, so IQR > 0
        df["malic_acid"] = [1.8, 2.0, 2.1, 1.9, 2.2, 2.0, 1.8, 2.1, 2.0, 2.0]
        # 99.0 is a huge statistical outlier but domain-valid (positive)
        df.loc[0, "malic_acid"] = 99.0
        result = audit_data(df)
        self.assertEqual(len(result["quarantined"]), 0)
        self.assertTrue(result["retained"].iloc[0]["review_needed"])
        self.assertTrue(result["retained"].iloc[0]["malic_acid_iqr_flag"])
if __name__ == "__main__":
    unittest.main()