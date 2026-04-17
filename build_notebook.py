"""Generate notebooks/airbnb_price_prediction.ipynb.

Run with: python3 build_notebook.py

No external dependencies. Uses only the stdlib (json). Re-run after editing
this script to regenerate the notebook. Safe to commit to the repo.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def md(text: str) -> dict:
    """Markdown cell from a plain string (preserves blank lines)."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _splitlines_keepends(text),
    }


def code(text: str) -> dict:
    """Code cell from a plain string."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _splitlines_keepends(text),
    }


def _splitlines_keepends(text: str) -> list[str]:
    """Jupyter stores source as a list of lines, each ending with a newline
    except possibly the last."""
    text = text.lstrip("\n").rstrip() + "\n"
    lines = text.splitlines(keepends=True)
    # Strip trailing newline off the final line to match nbformat convention.
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------


CELLS: list[dict] = []


# ----- Title + context -----------------------------------------------------
CELLS.append(md("""
# Austin Airbnb Listing Price Prediction

**AI 391M — Case Studies in Machine Learning · UT Austin · Spring 2026**
**Author:** Nitin Bhatnagar

**Research question.** *How well can standard machine-learning models predict nightly Airbnb listing prices in Austin, Texas from publicly available listing attributes, and which model family — regularized linear, gradient-boosted trees, or a feed-forward neural network — offers the best accuracy–interpretability tradeoff?*

This notebook is organized into numbered **Sections** and **Step** cells. Run one `Step` cell at a time and inspect its output before moving on. All sections are runnable end-to-end on Google Colab's free CPU runtime.

**Sections**

0. Setup & imports
1. Data acquisition (Inside Airbnb)
2. Exploratory Data Analysis
3. Data cleaning & feature engineering
4. Train / validation / test split
5. Baseline & Ridge regression
6. Random Forest
7. XGBoost + SHAP interpretation
8. PyTorch MLP
9. Results summary & comparison
10. Export figures for the paper
"""))


# ----- Section 0: Setup & imports -----------------------------------------
CELLS.append(md("""
## Section 0 · Setup & imports

Colab's base image has most of what we need. We only install `shap` explicitly (sometimes missing). XGBoost and PyTorch ship with Colab.
"""))

CELLS.append(code("""
# Step 0.1 — Install any missing packages (Colab runs this once per session).
!pip install -q shap==0.44.*
"""))

CELLS.append(code("""
# Step 0.2 — Imports, seeds, and path configuration.
import os
import gzip
import math
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths — work both when the notebook is cloned as a repo and when running in Colab.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
for d in (DATA_DIR, PROCESSED_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110

print("Project root:", PROJECT_ROOT)
"""))


# ----- Section 1: Data acquisition ----------------------------------------
CELLS.append(md("""
## Section 1 · Data acquisition

We pull the most recent Austin snapshot from [Inside Airbnb](https://insideairbnb.com/austin). Inside Airbnb publishes a fresh `listings.csv.gz` roughly monthly. Update the URL below if a newer snapshot is available, and record the snapshot date in `data/README.md` for reproducibility.
"""))

CELLS.append(code("""
# Step 1.1 — Download the Inside Airbnb Austin listings CSV (only if not cached).
LISTINGS_URL = "http://data.insideairbnb.com/united-states/tx/austin/2025-12-15/data/listings.csv.gz"
LISTINGS_PATH = DATA_DIR / "listings.csv.gz"

if not LISTINGS_PATH.exists():
    !wget -q -O "{LISTINGS_PATH}" "{LISTINGS_URL}"
    print("Downloaded to", LISTINGS_PATH)
else:
    print("Using cached file at", LISTINGS_PATH)

print("File size (MB):", round(LISTINGS_PATH.stat().st_size / 1e6, 2))
"""))

CELLS.append(code("""
# Step 1.2 — Load into pandas and inspect shape + columns.
df_raw = pd.read_csv(LISTINGS_PATH, compression="gzip", low_memory=False)
print("Rows:", len(df_raw))
print("Columns:", len(df_raw.columns))
df_raw.head(3)
"""))


# ----- Section 2: EDA ------------------------------------------------------
CELLS.append(md("""
## Section 2 · Exploratory Data Analysis

Understand the target (nightly price), the key candidate features, and any obvious data-quality issues before we commit to a preprocessing recipe.
"""))

CELLS.append(code("""
# Step 2.1 — Clean the price column and take the log.
# Inside Airbnb stores price as strings like "$1,234.00" — strip and cast to float.
df = df_raw.copy()
df["price_num"] = (
    df["price"].astype(str)
      .str.replace("$", "", regex=False)
      .str.replace(",", "", regex=False)
      .replace({"": np.nan, "nan": np.nan})
      .astype(float)
)
df["log_price"] = np.log1p(df["price_num"])
df[["price", "price_num", "log_price"]].describe()
"""))

CELLS.append(code("""
# Step 2.2 — Distributions of the target and a few key predictors.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["price_num"].clip(upper=df["price_num"].quantile(0.99)), bins=60, ax=axes[0])
axes[0].set(title="Nightly price (USD), 99th-pct clip", xlabel="price", ylabel="listings")
sns.histplot(df["log_price"].dropna(), bins=60, ax=axes[1], color="tab:orange")
axes[1].set(title="log1p(price)", xlabel="log price")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.3 — Geographic spread of listings, colored by log price.
sample = df.dropna(subset=["latitude", "longitude", "log_price"]).sample(min(5000, len(df)), random_state=SEED)
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(sample["longitude"], sample["latitude"], c=sample["log_price"],
                s=6, alpha=0.55, cmap="viridis")
ax.set(title="Austin Airbnb listings — log price", xlabel="Longitude", ylabel="Latitude")
plt.colorbar(sc, ax=ax, label="log1p(price)")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.4 — Median price by top-15 neighborhoods.
top_nbh = (
    df.groupby("neighbourhood_cleansed")["price_num"]
      .median()
      .sort_values(ascending=False)
      .head(15)
)
fig, ax = plt.subplots(figsize=(8, 5))
top_nbh.plot(kind="barh", ax=ax, color="steelblue")
ax.set(title="Top 15 Austin neighborhoods by median nightly price",
       xlabel="Median price (USD)", ylabel="")
ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.5 — Correlation heatmap of numeric candidate features.
numeric_cols = [
    "price_num", "bedrooms", "beds", "accommodates", "bathrooms",
    "minimum_nights", "number_of_reviews", "review_scores_rating",
    "reviews_per_month", "host_listings_count",
]
numeric_cols = [c for c in numeric_cols if c in df.columns]
corr = df[numeric_cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Pearson correlations among numeric features")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.6 — Missingness pattern for candidate feature columns.
missing = df[numeric_cols + ["neighbourhood_cleansed", "room_type", "property_type",
                             "amenities", "description", "host_since"]].isna().mean().sort_values()
missing.plot(kind="barh", figsize=(7, 5), color="tomato")
plt.title("Share of missing values per column (lower is better)")
plt.xlabel("fraction missing")
plt.tight_layout(); plt.show()
missing
"""))


# ----- Section 3: Data cleaning & feature engineering ----------------------
CELLS.append(md("""
## Section 3 · Data cleaning & feature engineering

Decisions (locked in here):

- Drop rows with no price.
- Winsorize the top and bottom **1%** of price to suppress extreme outliers (luxury rentals, test/$0 listings).
- Engineer four derived features: `amenity_count`, `desc_length`, `host_tenure_days`, `distance_to_downtown_km`.
- Keep `room_type` (one-hot), top-10 `property_type` + Other (one-hot), and `neighbourhood_cleansed` (target-encoded later inside the CV split so it's not leaky).
"""))

CELLS.append(code("""
# Step 3.1 — Drop missing-price rows, winsorize extremes.
df = df.dropna(subset=["price_num"]).copy()
lo, hi = df["price_num"].quantile([0.01, 0.99])
df = df[(df["price_num"] >= lo) & (df["price_num"] <= hi)].reset_index(drop=True)
df["log_price"] = np.log1p(df["price_num"])
print(f"After winsorization: {len(df):,} rows, price range ${lo:.0f}–${hi:.0f}")
"""))

CELLS.append(code("""
# Step 3.2 — Engineered features.

# amenity_count: parse the "amenities" string like ["Wifi", "Kitchen", ...] into a count.
def _count_amenities(s):
    if isinstance(s, str) and s.startswith("["):
        try:
            return len(json.loads(s.replace('""', '\"')))
        except Exception:
            return s.count(",") + 1
    return 0
df["amenity_count"] = df["amenities"].apply(_count_amenities)

# desc_length: character length of the listing description (proxy for host effort).
df["desc_length"] = df["description"].fillna("").astype(str).str.len()

# host_tenure_days: days between host_since and snapshot date.
df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
snapshot_date = df["host_since_dt"].max()
df["host_tenure_days"] = (snapshot_date - df["host_since_dt"]).dt.days.fillna(0)

# distance_to_downtown_km: haversine to the Texas Capitol (30.2672, -97.7431).
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
df["distance_to_downtown_km"] = _haversine(df["latitude"], df["longitude"], 30.2672, -97.7431)

df[["amenity_count", "desc_length", "host_tenure_days", "distance_to_downtown_km"]].describe()
"""))

CELLS.append(code("""
# Step 3.3 — Select features, handle missing numerics, one-hot encode small categoricals.
NUMERIC_FEATURES = [
    "bedrooms", "beds", "accommodates", "bathrooms",
    "minimum_nights", "number_of_reviews", "review_scores_rating",
    "reviews_per_month", "host_listings_count",
    "amenity_count", "desc_length", "host_tenure_days", "distance_to_downtown_km",
]
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]

# bathrooms is often stored as text like "1 bath" in newer snapshots — coerce.
if df["bathrooms"].dtype == object:
    df["bathrooms"] = df["bathrooms"].astype(str).str.extract(r"([0-9.]+)").astype(float)

# Keep top-10 property types; bucket the rest as "Other".
top_props = df["property_type"].value_counts().head(10).index
df["property_type_bucket"] = df["property_type"].where(df["property_type"].isin(top_props), "Other")

CATEGORICAL_FEATURES = ["room_type", "property_type_bucket"]
TARGET_ENCODE_COL = "neighbourhood_cleansed"

# Impute medians on numerics.
for c in NUMERIC_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

df_clean = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_ENCODE_COL, "log_price", "price_num"]].copy()
print("Clean frame:", df_clean.shape)
df_clean.head(3)
"""))


# ----- Section 4: Train/val/test split ------------------------------------
CELLS.append(md("""
## Section 4 · Train / validation / test split

An 80 / 10 / 10 split stratified by price quintile. Target encoding for `neighbourhood_cleansed` is **fit on the training fold only** and then applied to val and test, which avoids leakage.
"""))

CELLS.append(code("""
# Step 4.1 — Stratified split + leakage-safe target encoding for neighborhood.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_clean["price_bucket"] = pd.qcut(df_clean["price_num"], q=5, labels=False, duplicates="drop")

train_df, temp_df = train_test_split(
    df_clean, test_size=0.2, random_state=SEED, stratify=df_clean["price_bucket"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["price_bucket"]
)

# Target encode neighborhood on train; apply to val/test with train-fold global mean fallback.
global_mean = train_df["log_price"].mean()
nbh_means = train_df.groupby(TARGET_ENCODE_COL)["log_price"].mean()

def _encode(frame):
    return frame[TARGET_ENCODE_COL].map(nbh_means).fillna(global_mean)

for frame in (train_df, val_df, test_df):
    frame.loc[:, "nbh_te"] = _encode(frame)

# One-hot encode the two small categoricals using the training columns as the schema.
cat_dummies_train = pd.get_dummies(train_df[CATEGORICAL_FEATURES], drop_first=False)
cat_cols = cat_dummies_train.columns.tolist()

def _cat_dummies(frame):
    dummies = pd.get_dummies(frame[CATEGORICAL_FEATURES], drop_first=False)
    return dummies.reindex(columns=cat_cols, fill_value=0)

feature_cols = NUMERIC_FEATURES + ["nbh_te"] + cat_cols

def _assemble(frame):
    X = pd.concat([frame[NUMERIC_FEATURES + ["nbh_te"]].reset_index(drop=True),
                   _cat_dummies(frame).reset_index(drop=True)], axis=1)
    y = frame["log_price"].to_numpy()
    return X.astype(float), y

X_train, y_train = _assemble(train_df)
X_val,   y_val   = _assemble(val_df)
X_test,  y_test  = _assemble(test_df)

# Standardize numerics only (leave one-hots as 0/1).
scaler = StandardScaler()
X_train[NUMERIC_FEATURES + ["nbh_te"]] = scaler.fit_transform(X_train[NUMERIC_FEATURES + ["nbh_te"]])
X_val[NUMERIC_FEATURES + ["nbh_te"]]   = scaler.transform(X_val[NUMERIC_FEATURES + ["nbh_te"]])
X_test[NUMERIC_FEATURES + ["nbh_te"]]  = scaler.transform(X_test[NUMERIC_FEATURES + ["nbh_te"]])

# Raw USD prices (for MAE reporting).
price_train = train_df["price_num"].to_numpy()
price_val   = val_df["price_num"].to_numpy()
price_test  = test_df["price_num"].to_numpy()

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Feature count: {len(feature_cols)}")

# Cache the splits so later sections (and the paper) can reload them.
pd.to_pickle((X_train, y_train, price_train), PROCESSED_DIR / "train.pkl")
pd.to_pickle((X_val, y_val, price_val),       PROCESSED_DIR / "val.pkl")
pd.to_pickle((X_test, y_test, price_test),    PROCESSED_DIR / "test.pkl")
print("Wrote pickles to", PROCESSED_DIR)
"""))


# ----- Section 5: Baseline + Ridge ----------------------------------------
CELLS.append(md("""
## Section 5 · Baseline & Ridge regression

Two sanity checks before the heavy models:

- **Median baseline** — predict the median training log-price for every test row. A good model must beat this.
- **Ridge** — L2-regularized linear regression, tuned on the validation set.

We'll track three metrics across all models: RMSE on log-price (the loss we optimize), MAE on the raw USD price (human-readable), and R² (variance explained).
"""))

CELLS.append(code("""
# Step 5.1 — Shared evaluation helper + median baseline.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(name, y_log_pred, y_log_true, price_true):
    rmse_log = math.sqrt(mean_squared_error(y_log_true, y_log_pred))
    price_pred = np.expm1(y_log_pred)
    mae_usd = mean_absolute_error(price_true, price_pred)
    r2 = r2_score(y_log_true, y_log_pred)
    print(f"{name:20s}  log-RMSE={rmse_log:.4f}  MAE=${mae_usd:7.2f}  R²={r2:.3f}")
    return {"model": name, "log_rmse": rmse_log, "mae_usd": mae_usd, "r2": r2}

results = []

# Median baseline on log-price.
baseline_pred = np.full_like(y_test, fill_value=np.median(y_train), dtype=float)
results.append(evaluate("Median baseline", baseline_pred, y_test, price_test))
"""))

CELLS.append(code("""
# Step 5.2 — RidgeCV (grid over alpha), evaluate on test.
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), scoring="neg_root_mean_squared_error", cv=5)
ridge.fit(X_train, y_train)
print("Best alpha:", ridge.alpha_)
ridge_pred = ridge.predict(X_test)
results.append(evaluate("Ridge", ridge_pred, y_test, price_test))

# Quick look at top positive / negative Ridge coefficients.
coef = pd.Series(ridge.coef_, index=feature_cols).sort_values()
print("\\nTop 5 negative coefficients (push price DOWN):")
print(coef.head(5))
print("\\nTop 5 positive coefficients (push price UP):")
print(coef.tail(5))
"""))


# ----- Section 6: Random Forest -------------------------------------------
CELLS.append(md("""
## Section 6 · Random Forest

A non-parametric ensemble. Default hyperparameters are usually fine for tabular data; we lightly tune `n_estimators` and `max_depth` on the validation fold.
"""))

CELLS.append(code("""
# Step 6.1 — Fit a small grid of Random Forests, pick the best on validation.
from sklearn.ensemble import RandomForestRegressor

best = None
for n_est in [200, 400]:
    for max_depth in [None, 20]:
        rf = RandomForestRegressor(
            n_estimators=n_est, max_depth=max_depth, n_jobs=-1, random_state=SEED,
        )
        rf.fit(X_train, y_train)
        val_rmse = math.sqrt(mean_squared_error(y_val, rf.predict(X_val)))
        print(f"n_est={n_est} max_depth={max_depth}  val_log_rmse={val_rmse:.4f}")
        if best is None or val_rmse < best[0]:
            best = (val_rmse, rf)

rf = best[1]
rf_pred = rf.predict(X_test)
results.append(evaluate("Random Forest", rf_pred, y_test, price_test))
"""))


# ----- Section 7: XGBoost + SHAP -----------------------------------------
CELLS.append(md("""
## Section 7 · XGBoost + SHAP interpretation

XGBoost with early stopping on the validation fold. After fitting, we use SHAP to produce a globally-interpretable feature-importance summary.
"""))

CELLS.append(code("""
# Step 7.1 — Train XGBoost with early stopping.
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    random_state=SEED,
    early_stopping_rounds=50,
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Best iteration:", xgb.best_iteration)
"""))

CELLS.append(code("""
# Step 7.2 — Evaluate XGBoost on the held-out test set.
xgb_pred = xgb.predict(X_test)
results.append(evaluate("XGBoost", xgb_pred, y_test, price_test))
"""))

CELLS.append(code("""
# Step 7.3 — SHAP summary plot (top 15 features). This is the key interpretability figure.
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test.sample(min(1000, len(X_test)), random_state=SEED))
sample_cols = X_test.sample(min(1000, len(X_test)), random_state=SEED)

shap.summary_plot(shap_values, sample_cols, max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_shap_xgboost.png", dpi=150, bbox_inches="tight")
plt.show()
"""))


# ----- Section 8: PyTorch MLP ---------------------------------------------
CELLS.append(md("""
## Section 8 · PyTorch MLP

A small feed-forward network (256 → 128 → 64, ReLU, dropout 0.2) trained with Adam on MSE of log-price. Early-stops on validation RMSE. The question is whether this narrow-but-deep model beats XGBoost on a tabular dataset — current literature (Shwartz-Ziv & Armon 2022) says usually not.
"""))

CELLS.append(code("""
# Step 8.1 — Build tensors + DataLoaders.
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def _to_tensor(X, y):
    return torch.tensor(X.to_numpy(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

Xt_tr, yt_tr = _to_tensor(X_train, y_train)
Xt_va, yt_va = _to_tensor(X_val,   y_val)
Xt_te, yt_te = _to_tensor(X_test,  y_test)

train_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=256, shuffle=True)
"""))

CELLS.append(code("""
# Step 8.2 — Define the MLP and train with early stopping on validation RMSE.
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP(in_dim=Xt_tr.shape[1]).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

EPOCHS, PATIENCE = 80, 8
best_val, best_state, bad_epochs = float("inf"), None, 0
history = {"train": [], "val": []}

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward(); optim.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_pred = model(Xt_va.to(device)).cpu().numpy()
    val_rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    train_rmse = math.sqrt(np.mean(train_losses))
    history["train"].append(train_rmse); history["val"].append(val_rmse)

    if val_rmse < best_val:
        best_val, bad_epochs = val_rmse, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch+1:3d}  train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}")

if best_state is not None:
    model.load_state_dict(best_state)
print("Best val log-RMSE:", round(best_val, 4))
"""))

CELLS.append(code("""
# Step 8.3 — Evaluate the MLP on the held-out test set + training curve figure.
model.eval()
with torch.no_grad():
    mlp_pred = model(Xt_te.to(device)).cpu().numpy()
results.append(evaluate("MLP (PyTorch)", mlp_pred, y_test, price_test))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history["train"], label="train")
ax.plot(history["val"],   label="val")
ax.set(xlabel="epoch", ylabel="log-RMSE", title="MLP training curve")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_training_curve.png", dpi=150, bbox_inches="tight")
plt.show()
"""))


# ----- Section 9: Results summary -----------------------------------------
CELLS.append(md("""
## Section 9 · Results summary & comparison

All four models evaluated on the same held-out test set. The table below is what goes into §4 (Results) of the paper.
"""))

CELLS.append(code("""
# Step 9.1 — Assemble the results table.
results_df = pd.DataFrame(results).set_index("model")
results_df = results_df.round({"log_rmse": 4, "mae_usd": 2, "r2": 3})
results_df
"""))

CELLS.append(code("""
# Step 9.2 — Bar chart comparison.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
results_df["log_rmse"].plot(kind="bar", ax=axes[0], color="tab:blue")
axes[0].set(title="Log-price RMSE (lower is better)", ylabel="RMSE")
axes[0].tick_params(axis="x", rotation=30)

results_df["mae_usd"].plot(kind="bar", ax=axes[1], color="tab:orange")
axes[1].set(title="MAE on raw USD price", ylabel="MAE ($)")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

CELLS.append(code("""
# Step 9.3 — Predicted vs. actual scatter for the best model (XGBoost).
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, xgb_pred, alpha=0.25, s=10)
lo, hi = min(y_test.min(), xgb_pred.min()), max(y_test.max(), xgb_pred.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1)
ax.set(xlabel="Actual log-price", ylabel="Predicted log-price",
       title="XGBoost predictions vs. actuals (test set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_xgb_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

# Residual distribution figure for each model.
fig, ax = plt.subplots(figsize=(7, 4))
for name, pred in [("Ridge", ridge_pred), ("RF", rf_pred), ("XGBoost", xgb_pred), ("MLP", mlp_pred)]:
    sns.kdeplot(y_test - pred, ax=ax, label=name, fill=True, alpha=0.2)
ax.set(title="Residuals on log-price (test set)", xlabel="y_true − y_pred")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
"""))


# ----- Section 10: Export figures -----------------------------------------
CELLS.append(md("""
## Section 10 · Export artifacts for the paper

Persist the final results table and confirm all figures landed in `paper/figures/`. These are what the paper's `![](...)` lines will reference.
"""))

CELLS.append(code("""
# Step 10.1 — Save the results table as CSV and list the exported figures.
results_csv = FIG_DIR.parent / "results_table.csv"
results_df.to_csv(results_csv)
print("Saved:", results_csv)
print()
print("Figures in", FIG_DIR)
for p in sorted(FIG_DIR.iterdir()):
    if p.is_file():
        print("  ", p.name, "·", round(p.stat().st_size/1024, 1), "KB")
"""))


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out_path = Path(__file__).parent / "notebooks" / "airbnb_price_prediction.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {out_path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
