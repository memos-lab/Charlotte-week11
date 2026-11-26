import unittest
from pathlib import Path

class ProjectFilesTest(unittest.TestCase):
    def test_inputs_exist(self):
        base = Path(__file__).resolve().parent / "data"
        for name in ["scada_synthetic.csv", "well_b_synthetic.las"]:
            with self.subTest(name=name):
                self.assertTrue((base / name).exists(), f"Missing {name}")

if __name__ == "__main__":
    unittest.main()