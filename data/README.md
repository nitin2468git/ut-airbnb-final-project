# Data

## Primary source: Inside Airbnb

The raw dataset comes from **Inside Airbnb** (https://insideairbnb.com), an independent, non-commercial research project that scrapes publicly visible Airbnb listings city by city. The Austin page is at:

- https://insideairbnb.com/austin/

Download the most recent `listings.csv.gz` from the "Get the Data" tab. Record the snapshot date below when you run the project.

**Snapshot pinned for this project:** _fill in after first download, e.g. 2026-04-10_

The file contains one row per active Austin listing (~15,000 rows) and ~75 columns, including price, location, host metadata, review counts, review scores, amenities, and listing descriptions. License: Creative Commons CC0 1.0 (public-domain dedication).

## How it is loaded

The first cell of [`../notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) downloads `listings.csv.gz` via `wget` into this `data/` directory when the notebook is run in Colab. The CSV is not committed to the repository (see `.gitignore`).

## Backup mirror: Kaggle

If Inside Airbnb is unreachable, the Kaggle community mirror "Airbnb Listings — Austin TX" provides a comparable snapshot:

- https://www.kaggle.com/datasets/airbnb/austin-texas-listings

## Processed splits

After `02_preprocess.ipynb` runs, it writes the following pickles to `data/processed/`:

| File | Shape (approx) | Contents |
|---|---|---|
| `train.pkl` | ~12,000 × F | Training features + target |
| `val.pkl` | ~1,500 × F | Validation features + target |
| `test.pkl` | ~1,500 × F | Held-out test features + target |

Where `F` is the final feature count after one-hot / target encoding (~50–80 columns). These pickles are also git-ignored; regenerate by re-running `02_preprocess.ipynb`.

## Ethics note

Inside Airbnb data contains listing IDs and host IDs that could in principle be re-identified. This project uses aggregate features only and does not attempt to re-identify hosts or map listings to specific residential addresses beyond the neighborhood granularity that Airbnb itself publishes.
