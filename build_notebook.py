"""Build the Colab notebook from plain strings. Re-run after editing."""

from __future__ import annotations

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text):
    text = text.lstrip("\n").rstrip() + "\n"
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


CELLS = []


# -- Title -----------------------------------------------------------------
CELLS.append(md("""
# Austin Airbnb listing price prediction

AI 391M · Case Studies in ML · UT Austin · Spring 2026
Author: Nitin Bhatnagar

**Research question.** How well can standard ML models predict nightly Airbnb prices in Austin, Texas from public listing attributes? I compare three families: Ridge regression, Random Forest / XGBoost, and a small PyTorch MLP.

The notebook runs top-to-bottom on Colab's free CPU runtime. Section headers mark the pipeline stages; each code cell is numbered `Step N.m`. I deliberately kept things explicit rather than factoring everything into helpers so a grader can read it like a script.
"""))


# -- Section 0 -------------------------------------------------------------
CELLS.append(md("""
## Section 0 · Setup

Colab already has pandas, sklearn, xgboost, torch. Only `shap` needs to be installed (and sometimes Colab already has it too, just the wrong version for NumPy 2).
"""))

CELLS.append(code("""
# Step 0.1: install shap. Colab ships NumPy 2.x so we need shap >= 0.46.
# If pip warns "NumPy 1.x cannot be run in NumPy 2.0", do
# Runtime -> Restart session, then skip this cell and go to 0.2.
!pip install -q -U "shap>=0.46"
"""))

CELLS.append(code("""
# Step 0.2: imports and paths.
import os, math, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Works whether I cloned the repo locally or opened this from GitHub in Colab.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT",
                                    Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
for d in (DATA_DIR, PROCESSED_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110
print("project root:", PROJECT_ROOT)
"""))


# -- Section 1 -------------------------------------------------------------
CELLS.append(md("""
## Section 1 · Get the data

Inside Airbnb publishes a fresh Austin snapshot every month or so at https://insideairbnb.com/austin. This project pins the 2025-09-16 release. If you want a newer one, replace the URL below and note the date in `data/README.md`.
"""))

CELLS.append(code("""
# Step 1.1: download the Austin listings CSV if we don't already have it locally.
LISTINGS_URL  = "https://data.insideairbnb.com/united-states/tx/austin/2025-09-16/data/listings.csv.gz"
LISTINGS_PATH = DATA_DIR / "listings.csv.gz"

if not LISTINGS_PATH.exists():
    !wget -q -O "{LISTINGS_PATH}" "{LISTINGS_URL}"
    print("downloaded to", LISTINGS_PATH)
else:
    print("using cached file at", LISTINGS_PATH)

print("size (MB):", round(LISTINGS_PATH.stat().st_size / 1e6, 2))
"""))

CELLS.append(code("""
# Step 1.2: load and peek at the shape.
df_raw = pd.read_csv(LISTINGS_PATH, compression="gzip", low_memory=False)
print("rows:", len(df_raw), " columns:", len(df_raw.columns))
df_raw.head(3)
"""))


# -- Section 2 -------------------------------------------------------------
CELLS.append(md("""
## Section 2 · EDA

Before committing to a cleaning recipe, I want to look at the target (nightly price), a few candidate predictors, and the missingness pattern. Six quick diagnostic plots.
"""))

CELLS.append(code("""
# Step 2.1: price is stored as "$1,234.00" strings. Strip and cast, then log.
df = df_raw.copy()
df["price_num"] = (df["price"].astype(str)
                              .str.replace("$", "", regex=False)
                              .str.replace(",", "", regex=False)
                              .replace({"": np.nan, "nan": np.nan})
                              .astype(float))
df["log_price"] = np.log1p(df["price_num"])
df[["price", "price_num", "log_price"]].describe()
"""))

CELLS.append(code("""
# Step 2.2: price distributions. Clip the top 1% on the raw scale so we can see the bulk.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["price_num"].clip(upper=df["price_num"].quantile(0.99)), bins=60, ax=axes[0])
axes[0].set(title="Nightly price (USD), 99th-pct clip", xlabel="price", ylabel="listings")
sns.histplot(df["log_price"].dropna(), bins=60, ax=axes[1], color="tab:orange")
axes[1].set(title="log1p(price)", xlabel="log price")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.3: where are the listings and how does price vary across Austin?
sample = df.dropna(subset=["latitude", "longitude", "log_price"]).sample(
    min(5000, len(df)), random_state=SEED)
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(sample["longitude"], sample["latitude"], c=sample["log_price"],
                s=6, alpha=0.55, cmap="viridis")
ax.set(title="Austin Airbnb listings, colored by log price",
       xlabel="Longitude", ylabel="Latitude")
plt.colorbar(sc, ax=ax, label="log1p(price)")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.4: top 15 neighborhoods by median nightly price.
top_nbh = (df.groupby("neighbourhood_cleansed")["price_num"]
             .median()
             .sort_values(ascending=False)
             .head(15))
fig, ax = plt.subplots(figsize=(8, 5))
top_nbh.plot(kind="barh", ax=ax, color="steelblue")
ax.set(title="Top 15 Austin neighborhoods by median nightly price",
       xlabel="Median price (USD)", ylabel="")
ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.5: correlation heatmap across numeric candidate features.
numeric_cols = [c for c in [
    "price_num", "bedrooms", "beds", "accommodates", "bathrooms",
    "minimum_nights", "number_of_reviews", "review_scores_rating",
    "reviews_per_month", "host_listings_count",
] if c in df.columns]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[numeric_cols].corr(numeric_only=True),
            annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Pearson correlations (numeric features)")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# Step 2.6: how bad is missingness on the columns I care about?
missing = (df[numeric_cols + ["neighbourhood_cleansed","room_type","property_type",
                              "amenities","description","host_since"]]
             .isna().mean().sort_values())
missing.plot(kind="barh", figsize=(7, 5), color="tomato")
plt.title("Fraction of missing values per column")
plt.xlabel("fraction missing")
plt.tight_layout(); plt.show()
missing
"""))


# -- Section 3 -------------------------------------------------------------
CELLS.append(md("""
## Section 3 · Cleaning and feature engineering

What I decided after looking at Section 2:

- Drop rows with no price.
- Winsorize the top and bottom 1% of price. There are luxury listings at $10k/night and $0 listings that look like test accounts; neither is a useful signal for this task.
- Add four hand-crafted features: amenity count, description length, how long the host has been on the platform, and distance to downtown.
- One-hot `room_type` and the top-10 property types (others bucketed as "Other"). `neighbourhood_cleansed` gets target-encoded later, inside the training fold only, so the test set doesn't leak.
"""))

CELLS.append(code("""
# Step 3.1: drop missing prices, winsorize top/bottom 1%.
df = df.dropna(subset=["price_num"]).copy()
lo, hi = df["price_num"].quantile([0.01, 0.99])
df = df[(df["price_num"] >= lo) & (df["price_num"] <= hi)].reset_index(drop=True)
df["log_price"] = np.log1p(df["price_num"])
print(f"after winsorization: {len(df):,} rows  (price ${lo:.0f}-${hi:.0f})")
"""))

CELLS.append(code("""
# Step 3.2: build the four engineered features.

# amenity_count: amenities comes as a JSON list string like ["Wifi","Kitchen",...].
def count_amenities(s):
    if isinstance(s, str) and s.startswith("["):
        try:
            return len(json.loads(s.replace('""', '\"')))
        except Exception:
            return s.count(",") + 1
    return 0
df["amenity_count"] = df["amenities"].apply(count_amenities)

# desc_length: rough proxy for how much effort the host put into the listing.
df["desc_length"] = df["description"].fillna("").astype(str).str.len()

# host_tenure_days: gap between host_since and the snapshot date.
df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
snapshot_date = df["host_since_dt"].max()
df["host_tenure_days"] = (snapshot_date - df["host_since_dt"]).dt.days.fillna(0)

# distance_to_downtown_km: haversine to the Texas Capitol (30.2672, -97.7431).
R = 6371.0
lat1 = np.radians(df["latitude"]); lon1 = np.radians(df["longitude"])
lat2 = np.radians(30.2672);        lon2 = np.radians(-97.7431)
dlat = lat2 - lat1; dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
df["distance_to_downtown_km"] = 2 * R * np.arcsin(np.sqrt(a))

df[["amenity_count","desc_length","host_tenure_days","distance_to_downtown_km"]].describe()
"""))

CELLS.append(code("""
# Step 3.3: pick features, impute medians, bucket rare property types.
NUMERIC = [c for c in [
    "bedrooms", "beds", "accommodates", "bathrooms",
    "minimum_nights", "number_of_reviews", "review_scores_rating",
    "reviews_per_month", "host_listings_count",
    "amenity_count", "desc_length", "host_tenure_days", "distance_to_downtown_km",
] if c in df.columns]

# Newer snapshots store bathrooms as "1 bath" text. Pull the number out.
if df["bathrooms"].dtype == object:
    df["bathrooms"] = df["bathrooms"].astype(str).str.extract(r"([0-9.]+)").astype(float)

top_props = df["property_type"].value_counts().head(10).index
df["property_type_bucket"] = df["property_type"].where(df["property_type"].isin(top_props), "Other")

CATS = ["room_type", "property_type_bucket"]
NBH  = "neighbourhood_cleansed"

for c in NUMERIC:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

df_clean = df[NUMERIC + CATS + [NBH, "log_price", "price_num"]].copy()
print("clean frame:", df_clean.shape)
df_clean.head(3)
"""))


# -- Section 4 -------------------------------------------------------------
CELLS.append(md("""
## Section 4 · Splits

80 / 10 / 10 stratified on price quintile. Target-encode `neighbourhood_cleansed` on the train fold only, then reuse those means on val and test so we don't leak.
"""))

CELLS.append(code("""
# Step 4.1: stratified split + leakage-safe target encoding + scaling.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_clean["bucket"] = pd.qcut(df_clean["price_num"], q=5, labels=False, duplicates="drop")
train_df, tmp = train_test_split(df_clean, test_size=0.2, random_state=SEED,
                                 stratify=df_clean["bucket"])
val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=SEED,
                                   stratify=tmp["bucket"])

# Target encoding fit on train fold only.
global_mean = train_df["log_price"].mean()
nbh_means   = train_df.groupby(NBH)["log_price"].mean()
for fr in (train_df, val_df, test_df):
    fr.loc[:, "nbh_te"] = fr[NBH].map(nbh_means).fillna(global_mean)

# One-hot schema is locked to the training columns.
train_cats = pd.get_dummies(train_df[CATS], drop_first=False)
cat_cols   = train_cats.columns.tolist()

def assemble(fr):
    dummies = pd.get_dummies(fr[CATS], drop_first=False).reindex(columns=cat_cols, fill_value=0)
    X = pd.concat([fr[NUMERIC + ["nbh_te"]].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1).astype(float)
    y = fr["log_price"].to_numpy()
    return X, y

X_train, y_train = assemble(train_df)
X_val,   y_val   = assemble(val_df)
X_test,  y_test  = assemble(test_df)

# Scale numerics; leave one-hots alone.
scaler = StandardScaler()
scale_cols = NUMERIC + ["nbh_te"]
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_val[scale_cols]   = scaler.transform(X_val[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

# Keep raw USD prices around so I can report dollar-MAE later.
price_train = train_df["price_num"].to_numpy()
price_val   = val_df["price_num"].to_numpy()
price_test  = test_df["price_num"].to_numpy()

feature_cols = list(X_train.columns)
print(f"train {X_train.shape}   val {X_val.shape}   test {X_test.shape}")
print(f"feature count: {len(feature_cols)}")

# Cache so I can reload from any section without rerunning 2 + 3 + 4.
pd.to_pickle((X_train, y_train, price_train), PROCESSED_DIR / "train.pkl")
pd.to_pickle((X_val,   y_val,   price_val),   PROCESSED_DIR / "val.pkl")
pd.to_pickle((X_test,  y_test,  price_test),  PROCESSED_DIR / "test.pkl")
print("wrote pickles to", PROCESSED_DIR)
"""))


# -- Section 5 -------------------------------------------------------------
CELLS.append(md("""
## Section 5 · Baseline and Ridge

Sanity-check the pipeline with a median-log-price baseline (a floor any real model has to clear), then fit Ridge regression with CV over alpha. I track three metrics on every model from here on: log-RMSE (what I'm optimizing), MAE in dollars (easier to talk about), and R².
"""))

CELLS.append(code("""
# Step 5.1: shared evaluator + median baseline.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

results = []

def score(name, pred_log, y_log, price_true):
    rmse = math.sqrt(mean_squared_error(y_log, pred_log))
    mae  = mean_absolute_error(price_true, np.expm1(pred_log))
    r2   = r2_score(y_log, pred_log)
    print(f"{name:20s}  log-RMSE={rmse:.4f}  MAE=${mae:7.2f}  R2={r2:.3f}")
    return {"model": name, "log_rmse": rmse, "mae_usd": mae, "r2": r2}

# Median baseline on the log scale.
baseline = np.full_like(y_test, fill_value=np.median(y_train), dtype=float)
results.append(score("Median baseline", baseline, y_test, price_test))
"""))

CELLS.append(code("""
# Step 5.2: RidgeCV over a log-spaced alpha grid, then test.
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), scoring="neg_root_mean_squared_error", cv=5)
ridge.fit(X_train, y_train)
print("best alpha:", ridge.alpha_)
ridge_pred = ridge.predict(X_test)
results.append(score("Ridge", ridge_pred, y_test, price_test))

coef = pd.Series(ridge.coef_, index=feature_cols).sort_values()
print("\\ntop 5 coefficients pushing price DOWN:")
print(coef.head(5))
print("\\ntop 5 coefficients pushing price UP:")
print(coef.tail(5))
"""))


# -- Section 6 -------------------------------------------------------------
CELLS.append(md("""
## Section 6 · Random Forest

Defaults are usually fine on tabular data. I try two values of `n_estimators` and two of `max_depth` on the val fold and pick the best.
"""))

CELLS.append(code("""
# Step 6.1: tiny hyperparameter sweep on the val fold.
from sklearn.ensemble import RandomForestRegressor

best = None
for n_est in (200, 400):
    for md_ in (None, 20):
        m = RandomForestRegressor(n_estimators=n_est, max_depth=md_,
                                  n_jobs=-1, random_state=SEED).fit(X_train, y_train)
        v = math.sqrt(mean_squared_error(y_val, m.predict(X_val)))
        print(f"n_est={n_est} max_depth={md_}  val_log_rmse={v:.4f}")
        if best is None or v < best[0]:
            best = (v, m)

rf = best[1]
rf_pred = rf.predict(X_test)
results.append(score("Random Forest", rf_pred, y_test, price_test))
"""))


# -- Section 7 -------------------------------------------------------------
CELLS.append(md("""
## Section 7 · XGBoost and SHAP

XGBoost with early stopping on the val fold. I want SHAP values afterwards too because the built-in `feature_importances_` is a bit too coarse for what I want to say in the paper.
"""))

CELLS.append(code("""
# Step 7.1: XGBoost with early stopping on val.
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9,
    tree_method="hist", random_state=SEED,
    early_stopping_rounds=50,
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("best iteration:", xgb.best_iteration)
"""))

CELLS.append(code("""
# Step 7.2: XGBoost on the test set.
xgb_pred = xgb.predict(X_test)
results.append(score("XGBoost", xgb_pred, y_test, price_test))
"""))

CELLS.append(code("""
# Step 7.3: SHAP summary plot, top 15 features. This is Figure 3 in the paper.
import shap

sample_idx = X_test.sample(min(1000, len(X_test)), random_state=SEED).index
X_sample   = X_test.loc[sample_idx]

explainer  = shap.TreeExplainer(xgb)
shap_vals  = explainer.shap_values(X_sample)

shap.summary_plot(shap_vals, X_sample, max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_shap_xgboost.png", dpi=150, bbox_inches="tight")
plt.show()
"""))


# -- Section 8 -------------------------------------------------------------
CELLS.append(md("""
## Section 8 · MLP

Small PyTorch feed-forward net (256 → 128 → 64, ReLU, 0.2 dropout). Adam on MSE of log-price, early stopping on val RMSE. I'm not expecting this to beat XGBoost — Shwartz-Ziv & Armon (2022) argue that on tabular data it usually won't — but I want to confirm it on my dataset so the paper has the comparison.
"""))

CELLS.append(code("""
# Step 8.1: tensors + loader.
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

Xt_tr = torch.tensor(X_train.to_numpy(), dtype=torch.float32); yt_tr = torch.tensor(y_train, dtype=torch.float32)
Xt_va = torch.tensor(X_val.to_numpy(),   dtype=torch.float32); yt_va = torch.tensor(y_val,   dtype=torch.float32)
Xt_te = torch.tensor(X_test.to_numpy(),  dtype=torch.float32); yt_te = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=256, shuffle=True)
"""))

CELLS.append(code("""
# Step 8.2: define the MLP and train with early stopping on val log-RMSE.
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

model = MLP(Xt_tr.shape[1]).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

EPOCHS, PATIENCE = 80, 8
best_val, best_state, bad = float("inf"), None, 0
history = {"train": [], "val": []}

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        l = loss_fn(model(xb), yb)
        l.backward(); opt.step()
        losses.append(l.item())

    model.eval()
    with torch.no_grad():
        vp = model(Xt_va.to(device)).cpu().numpy()
    val_rmse   = math.sqrt(mean_squared_error(y_val, vp))
    train_rmse = math.sqrt(np.mean(losses))
    history["train"].append(train_rmse); history["val"].append(val_rmse)

    if val_rmse < best_val:
        best_val, bad = val_rmse, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"early stop at epoch {epoch+1}")
            break
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch+1:3d}  train={train_rmse:.4f}  val={val_rmse:.4f}")

if best_state is not None:
    model.load_state_dict(best_state)
print("best val log-RMSE:", round(best_val, 4))
"""))

CELLS.append(code("""
# Step 8.3: test-set score + training curve figure.
model.eval()
with torch.no_grad():
    mlp_pred = model(Xt_te.to(device)).cpu().numpy()
results.append(score("MLP", mlp_pred, y_test, price_test))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history["train"], label="train")
ax.plot(history["val"],   label="val")
ax.set(xlabel="epoch", ylabel="log-RMSE", title="MLP training curve")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_training_curve.png", dpi=150, bbox_inches="tight")
plt.show()
"""))


# -- Section 9 -------------------------------------------------------------
CELLS.append(md("""
## Section 9 · Results

All four models on the same held-out test split. This is what goes into §4 of the paper.
"""))

CELLS.append(code("""
# Step 9.1: assemble the results table.
results_df = pd.DataFrame(results).set_index("model").round({"log_rmse": 4, "mae_usd": 2, "r2": 3})
results_df
"""))

CELLS.append(code("""
# Step 9.2: bar chart comparison (log-RMSE and MAE side by side).
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
results_df["log_rmse"].plot(kind="bar", ax=axes[0], color="tab:blue")
axes[0].set(title="log-price RMSE (lower is better)", ylabel="RMSE")
axes[0].tick_params(axis="x", rotation=30)
results_df["mae_usd"].plot(kind="bar", ax=axes[1], color="tab:orange")
axes[1].set(title="MAE on raw USD price", ylabel="MAE ($)")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

CELLS.append(code("""
# Step 9.3: predicted-vs-actual for XGBoost + residual KDEs.
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, xgb_pred, alpha=0.25, s=10)
lo, hi = min(y_test.min(), xgb_pred.min()), max(y_test.max(), xgb_pred.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1)
ax.set(xlabel="actual log-price", ylabel="predicted log-price",
       title="XGBoost predictions vs actuals (test set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_xgb_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
for name, pred in [("Ridge", ridge_pred), ("RF", rf_pred), ("XGBoost", xgb_pred), ("MLP", mlp_pred)]:
    sns.kdeplot(y_test - pred, ax=ax, label=name, fill=True, alpha=0.2)
ax.set(title="residuals on log-price (test set)", xlabel="y_true - y_pred")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
"""))


# -- Section 10 ------------------------------------------------------------
CELLS.append(md("""
## Section 10 · Save artifacts

Write out the results table and sanity-check that every figure landed in `paper/figures/`.
"""))

CELLS.append(code("""
# Step 10.1: save results + list figures.
results_csv = FIG_DIR.parent / "results_table.csv"
results_df.to_csv(results_csv)
print("saved:", results_csv)
print()
print("figures in", FIG_DIR)
for p in sorted(FIG_DIR.iterdir()):
    if p.is_file():
        print(" ", p.name, round(p.stat().st_size/1024, 1), "KB")
"""))


# ---------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.11",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}


def main():
    out = Path(__file__).parent / "notebooks" / "airbnb_price_prediction.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
