"""
Data preparation for the Wellness Tourism purchase-prediction pipeline.

- Loads the registered dataset straight from the repository data folder
- Cleans it (drops ID/index columns, fixes inconsistent category labels,
  removes duplicates, fills any missing values)
- Splits it into stratified train/test sets and saves them as CSVs in the
  repository root, where the GitHub Actions job uploads them as an artifact
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]   # row index + unique ID carry no signal
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape}")

# ---------------------------------------------------------------- clean
# 1. Remove columns that are identifiers, not features
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"Dropped identifier columns: {[c for c in DROP_COLS]}")

# 2. Standardise inconsistent category labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
print("Fixed Gender label 'Fe Male' -> 'Female':", df["Gender"].value_counts().to_dict())

# 3. Remove exact duplicate rows (possible once the ID column is gone)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows -> {len(df)} rows")

# 4. Fill any missing values defensively (current file has none, future drops may)
num_cols = df.select_dtypes(include="number").columns.drop(TARGET)
cat_cols = df.select_dtypes(exclude="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
print(f"Missing values after cleaning: {int(df.isna().sum().sum())}")

# ---------------------------------------------------------------- split
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------- save
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"\nTrain: {Xtrain.shape}, positive rate {ytrain.mean():.3f}")
print(f"Test : {Xtest.shape}, positive rate {ytest.mean():.3f}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
