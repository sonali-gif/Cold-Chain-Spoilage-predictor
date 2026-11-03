import pandas as pd
import os

# 1. Define dataset path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
BASE_DIR, "dataset", "dummy_spoilage_dataset.xlsx"
)

# 2. Load dataset

df = pd.read_excel(DATASET_PATH)

print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# 3. Check missing values

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

# 4. Remove completely empty columns

df = df.dropna(axis=1, how="all")

# 5. Remove unnecessary columns

columns_to_remove = [
    "S_No",
    "Stage_Verdict",
    "Visual_Notes"
]

df = df.drop(
columns=columns_to_remove,
errors="ignore"
)

# 6. Remove duplicate records

df = df.drop_duplicates()

# 7. Separate features and target

X = df.drop(
    columns=["Label", "Timestamp", "Sample_ID"],
    errors="ignore"
)

y = df["Label"]

print("\n========== PREPROCESSING RESULTS ==========")

print("Total Records:", len(df))

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(y.value_counts())

# 8. Save cleaned dataset

OUTPUT_PATH = os.path.join(
BASE_DIR, "dataset", "cleaned_dataset.csv"
)

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", OUTPUT_PATH)
