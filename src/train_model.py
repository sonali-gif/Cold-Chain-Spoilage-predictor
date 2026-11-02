import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. DEFINE PATHS
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "cleaned_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ------------------------------------------------

metadata_columns = ["Timestamp", "Sample_ID"]

X = df.drop(
    columns=["Label"] + metadata_columns,
    errors="ignore"
)

y = df["Label"]


# ------------------------------------------------
# 4. DATASET ANALYSIS
# ------------------------------------------------

print("\n========== LABEL DISTRIBUTION ==========")
print(y.value_counts())

print("\n========== AVERAGE SENSOR READINGS BY CLASS ==========")

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

analysis_df = X.copy()
analysis_df["Label"] = y

print(
    analysis_df.groupby("Label")[numeric_columns]
    .mean()
    .round(3)
    .to_string()
)

print("\n========== FEATURE VARIATION BY CLASS ==========")

variation_features = [
    "Spoilage_Signal",
    "MQ135_VOC_Index_Est",
    "Rs_R0_Ratio",
    "DHT22_Temp_C",
    "DHT22_Humidity_%"
]

print(
    analysis_df.groupby("Label")[variation_features]
    .std()
    .round(3)
    .to_string()
)

print("\n========== LABEL DISTRIBUTION PER SAMPLE ==========")

sample_label_counts = pd.crosstab(
    df["Sample_ID"], df["Label"]
)

print(sample_label_counts.to_string())

print("\n========== VEGETABLE TYPE COUNTS ==========")
print(df["Vegetable_Type"].value_counts().to_string())

print("\n========== DUPLICATE FEATURE RECORDS ==========")

print(
    "Duplicate input records:",
    X.duplicated().sum()
)


# ------------------------------------------------
# 5. GROUP-BASED TRAIN-TEST SPLIT
# ------------------------------------------------

# All readings belonging to one physical sample
# must stay together in either training or testing.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=df["Sample_ID"]
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ------------------------------------------------
# 6. VERIFY TRAINING AND TESTING DATA
# ------------------------------------------------

train_samples = set(df.iloc[train_idx]["Sample_ID"])
test_samples = set(df.iloc[test_idx]["Sample_ID"])

print("\n========== TRAINING DATA ==========")

print("Training records:", len(X_train))
print("Training samples:", len(train_samples))
print(y_train.value_counts())

print("\n========== TESTING DATA ==========")

print("Testing records:", len(X_test))
print("Testing samples:", len(test_samples))
print(y_test.value_counts())

# Check for sample overlap

overlapping_samples = train_samples.intersection(test_samples)

assert len(overlapping_samples) == 0, (
    "Data leakage detected! Some samples appear in both sets."
)

print("\nNo sample overlap detected!")


# Check whether all classes are present

assert set(y_train.unique()) == set(y.unique()), (
    "Some classes are missing from training data!"
)

assert set(y_test.unique()) == set(y.unique()), (
    "Some classes are missing from testing data!"
)

print("All four classes are present in training and testing!")


# ------------------------------------------------
# 7. IDENTIFY FEATURE TYPES
# ------------------------------------------------

categorical_features = ["Vegetable_Type"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]


# ------------------------------------------------
# 8. ENCODE CATEGORICAL DATA
# ------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "vegetable_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ------------------------------------------------
# 9. CREATE RANDOM FOREST MODEL
# ------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------
# 10. CREATE COMPLETE PIPELINE
# ------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", random_forest)
    ]
)


# ------------------------------------------------
# 11. TRAIN MODEL
# ------------------------------------------------

print("\nTraining Random Forest Model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ------------------------------------------------
# 12. MAKE PREDICTIONS
# ------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------
# 13. MODEL EVALUATION
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# Compare against majority-class baseline

majority_class = y_train.value_counts().idxmax()

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\n========== BASELINE COMPARISON ==========")

print("Majority class:", majority_class)

print(
    "Baseline accuracy:",
    round(baseline_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ------------------------------------------------
# 14. CONFUSION MATRIX
# ------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(
    cmap="Greens",
    xticks_rotation=45
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_DIR, "confusion_matrix.png")
)

plt.close()


# ------------------------------------------------
# 15. FEATURE IMPORTANCE
# ------------------------------------------------

feature_names = (
    model.named_steps["preprocessing"]
    .get_feature_names_out()
)

importances = (
    model.named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(importance_df.to_string(index=False))

importance_df.to_csv(
    os.path.join(MODEL_DIR, "feature_importance.csv"),
    index=False
)


# ------------------------------------------------
# 16. SAVE TRAINED MODEL
# ------------------------------------------------

MODEL_PATH = os.path.join(
    MODEL_DIR, "random_forest_model.pkl"
)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)
