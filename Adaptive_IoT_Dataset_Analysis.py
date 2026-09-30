"""
Adaptive AI-Based Edge–Fog–Cloud IoT Framework
Dataset Analysis Script

Windows dataset path:
C:\RM paper work\dataset.csv

Install:
pip install pandas numpy matplotlib scikit-learn
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATASET
# ============================================================
DATA_PATH = Path(r"C:\RM paper work\dataset.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check the path or change DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ADAPTIVE AI EDGE–FOG–CLOUD IoT DATASET ANALYSIS")
print("=" * 70)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# 2. DATA QUALITY
# ============================================================
print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isna().sum())

print("\nTotal missing:", df.isna().sum().sum())

print("\n--- DUPLICATES ---")
print("Duplicate rows:", df.duplicated().sum())

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe(include="all").T)

# ============================================================
# 3. TIMESTAMP FEATURES
# ============================================================
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    print("\n--- TIMESTAMP ---")
    print("Start:", df["Timestamp"].min())
    print("End  :", df["Timestamp"].max())
    print("Invalid timestamps:", df["Timestamp"].isna().sum())

    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    df["Day"] = df["Timestamp"].dt.day

# ============================================================
# 4. SMART-CITY VARIABLES
# ============================================================
PILLARS = [
    "Smart_Mobility",
    "Smart_Environment",
    "Smart_Government",
    "Smart_Economy",
    "Smart_People",
    "Smart_Living"
]

PILLARS = [c for c in PILLARS if c in df.columns]

print("\n--- SMART-CITY PILLARS ---")
print(PILLARS)

# ============================================================
# 5. DESCRIPTIVE ANALYSIS
# ============================================================
summary = df[PILLARS].agg(["mean", "std", "min", "max"]).T
print("\n--- PILLAR SUMMARY ---")
print(summary.round(2))

summary.to_csv("pillar_summary.csv")

# Average pillar plot
df[PILLARS].mean().sort_values(ascending=False).plot(
    kind="bar", figsize=(11, 5)
)
plt.title("Average Smart-City Pillar Scores")
plt.ylabel("Mean Value")
plt.xlabel("Smart-City Pillar")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("average_pillar_scores.png", dpi=300)
plt.show()

# Boxplot
df[PILLARS].plot(kind="box", figsize=(12, 6))
plt.title("Distribution of Smart-City Pillar Variables")
plt.ylabel("Value")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("pillar_boxplot.png", dpi=300)
plt.show()

# ============================================================
# 6. CORRELATION
# ============================================================
corr_data = df[PILLARS + ["SmartCity_Index"]]
corr = corr_data.corr()

print("\n--- CORRELATION MATRIX ---")
print(corr.round(4))

corr.to_csv("correlation_matrix.csv")

plt.figure(figsize=(10, 8))
plt.imshow(corr, interpolation="nearest", aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

index_corr = (
    corr["SmartCity_Index"]
    .drop("SmartCity_Index")
    .sort_values(ascending=False)
)

print("\n--- CORRELATION WITH SMARTCITY INDEX ---")
print(index_corr.round(4))

# ============================================================
# 7. RANDOM FOREST REGRESSION
#
# Predict SmartCity_Index from the six pillar variables.
# This is an exploratory ML experiment, NOT an orchestration
# performance benchmark.
# ============================================================
X = df[PILLARS].copy()
y = df["SmartCity_Index"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

model.fit(X_train, y_train)
prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
r2 = r2_score(y_test, prediction)

print("\n" + "=" * 70)
print("RANDOM FOREST REGRESSION")
print("=" * 70)
print("Training records:", len(X_train))
print("Testing records :", len(X_test))
print(f"MAE             : {mae:.4f}")
print(f"RMSE            : {rmse:.4f}")
print(f"Test R2         : {r2:.4f}")

# ============================================================
# 8. 5-FOLD CROSS VALIDATION
# ============================================================
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_r2 = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print("\n--- 5-FOLD CROSS VALIDATION ---")
print("R2 scores:", np.round(cv_r2, 4))
print(f"Mean R2 : {cv_r2.mean():.4f}")
print(f"Std R2  : {cv_r2.std():.4f}")

# ============================================================
# 9. ACTUAL VS PREDICTED
# ============================================================
plt.figure(figsize=(7, 6))
plt.scatter(y_test, prediction)

mn = min(y_test.min(), prediction.min())
mx = max(y_test.max(), prediction.max())

plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual SmartCity Index")
plt.ylabel("Predicted SmartCity Index")
plt.title("Actual vs Predicted SmartCity Index")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
rf = model.named_steps["rf"]

importance = pd.Series(
    rf.feature_importances_,
    index=PILLARS
).sort_values(ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance.round(4))

importance.to_csv("feature_importance.csv")

importance.sort_values().plot(
    kind="barh",
    figsize=(9, 5)
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# ============================================================
# 11. SMARTCITY INDEX TREND
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(df["SmartCity_Index"], marker="o")
plt.title("SmartCity Index Across Observations")
plt.xlabel("Observation")
plt.ylabel("SmartCity Index")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("smartcity_index_trend.png", dpi=300)
plt.show()

# ============================================================
# 12. PROXY-BASED ADAPTIVE ORCHESTRATION
#
# The supplied dataset has no observed placement target and no
# direct latency/security/energy/resource measurements.
#
# Therefore this section is a SIMULATION using transparent
# normalized proxies. Do NOT describe these placements as
# observed real-world Edge/Fog/Cloud measurements.
# ============================================================
def minmax(series):
    denominator = series.max() - series.min()
    if denominator == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )
    return (series - series.min()) / denominator


norm = pd.DataFrame(index=df.index)

for col in PILLARS:
    norm[col] = minmax(df[col])

# Proxy definitions
urgency = (
    0.60 * norm["Smart_Mobility"] +
    0.40 * norm["Smart_Environment"]
)

compute = norm[PILLARS].mean(axis=1)

privacy = (
    0.45 * norm["Smart_Government"] +
    0.30 * norm["Smart_People"] +
    0.25 * norm["Smart_Living"]
)

# Objective weights
ALPHA = 0.30  # latency
BETA = 0.20   # resource
GAMMA = 0.20  # security
DELTA = 0.15  # privacy
EPSILON = 0.15  # resilience

selected_layers = []
all_scores = []

for i in df.index:

    u = float(urgency.loc[i])
    c = float(compute.loc[i])
    p = float(privacy.loc[i])

    candidates = {
        "Edge": {
            "latency": 0.15 + 0.15 * (1-u),
            "resource": 0.30 + 0.55 * c,
            "security": 0.30 + 0.35 * p,
            "privacy": 0.10 + 0.15 * (1-p),
            "resilience": 0.35
        },

        "Fog": {
            "latency": 0.35 + 0.15 * (1-u),
            "resource": 0.20 + 0.35 * c,
            "security": 0.25 + 0.25 * p,
            "privacy": 0.25 + 0.10 * (1-p),
            "resilience": 0.20
        },

        "Cloud": {
            "latency": 0.65 + 0.20 * (1-u),
            "resource": 0.10 + 0.15 * c,
            "security": 0.20 + 0.15 * p,
            "privacy": 0.55 + 0.15 * p,
            "resilience": 0.10
        }
    }

    scores = {}

    for layer, v in candidates.items():

        score = (
            ALPHA * v["latency"] +
            BETA * v["resource"] +
            GAMMA * v["security"] +
            DELTA * v["privacy"] +
            EPSILON * v["resilience"]
        )

        scores[layer] = score

    best_layer = min(scores, key=scores.get)

    selected_layers.append(best_layer)

    all_scores.append({
        "Edge_Score": scores["Edge"],
        "Fog_Score": scores["Fog"],
        "Cloud_Score": scores["Cloud"],
        "Selected_Layer": best_layer
    })

orchestration = pd.DataFrame(
    all_scores,
    index=df.index
)

analysis_results = pd.concat(
    [
        df,
        pd.DataFrame({
            "Urgency_Proxy": urgency,
            "Compute_Proxy": compute,
            "Privacy_Proxy": privacy
        }),
        orchestration
    ],
    axis=1
)

print("\n" + "=" * 70)
print("SIMULATED ADAPTIVE ORCHESTRATION")
print("=" * 70)
print(
    "WARNING: Selected_Layer is simulated from proxy assumptions. "
    "It is not an observed dataset label."
)

print("\nLayer allocation:")
print(
    analysis_results["Selected_Layer"]
    .value_counts()
)

analysis_results.to_csv(
    "adaptive_orchestration_analysis_results.csv",
    index=False
)

analysis_results["Selected_Layer"].value_counts().plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Simulated Adaptive Workload Placement")
plt.xlabel("Processing Layer")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "simulated_workload_placement.png",
    dpi=300
)
plt.show()

# ============================================================
# 13. MANUSCRIPT RESULT SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("MANUSCRIPT-READY RESULTS")
print("=" * 70)

print(f"Dataset size: {len(df)} records × {len(pd.read_csv(DATA_PATH).columns)} columns")
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\nPillar means:")
print(df[PILLARS].mean().round(2))

print("\nRandom Forest:")
print(f"MAE = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"Test R2 = {r2:.4f}")
print(f"CV mean R2 = {cv_r2.mean():.4f}")
print(f"CV std R2 = {cv_r2.std():.4f}")

print("\nFeature importance:")
print(importance.round(4))

print("\nSimulated layer allocation:")
print(analysis_results["Selected_Layer"].value_counts())

print("\nAll analysis completed successfully.")
