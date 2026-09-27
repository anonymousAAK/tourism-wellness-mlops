"""
Data registration for the "Visit with Us" Wellness Tourism project.

The raw dataset lives inside the GitHub repository (tourism_project/data/tourism.csv),
so "registering" it means validating that the file the pipeline is about to use is the
file we expect: it exists, it is not empty, it has every expected column, the target is
binary, and the IDs are unique. The script fails the CI job (non-zero exit) if any check
fails, which stops the downstream jobs from training on a broken file.
"""
import sys
import hashlib
from pathlib import Path

import pandas as pd

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"

# Every column promised in the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    # 1. File exists and is readable
    if not DATA_PATH.exists():
        fail(f"{DATA_PATH} not found. Add tourism.csv to tourism_project/data/.")
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        fail("Dataset is empty.")
    print(f"[OK] Loaded {DATA_PATH} -> {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Schema check: all expected columns must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        fail(f"Missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    print(f"[OK] All {len(EXPECTED_COLUMNS)} expected columns present")
    if extra:
        print(f"[INFO] Extra columns (will be dropped in data prep): {extra}")

    # 3. Target sanity: binary 0/1 with no missing values
    if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
        fail(f"Target '{TARGET}' must be binary 0/1 with no missing values.")
    print(f"[OK] Target '{TARGET}' is binary")

    # 4. Primary key sanity
    if df["CustomerID"].duplicated().any():
        fail("Duplicate CustomerID values found.")
    print("[OK] CustomerID is unique")

    # 5. Summary
    md5 = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()
    print("\n===== Dataset summary =====")
    print(f"Version fingerprint (md5): {md5}")
    print(f"Missing values (total)   : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows           : {int(df.duplicated().sum())}")
    print("Target distribution      :")
    dist = df[TARGET].value_counts().sort_index()
    for k, v in dist.items():
        print(f"   {k}: {v:>5}  ({v / len(df):.1%})")
    print("\nColumn types:")
    print(df[EXPECTED_COLUMNS].dtypes.value_counts().to_string())
    print("\nDataset registered successfully.")


if __name__ == "__main__":
    main()
