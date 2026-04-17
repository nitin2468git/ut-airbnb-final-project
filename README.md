# Austin Airbnb Listing Price Prediction

Final project for **AI 391M — Case Studies in Machine Learning**, Spring 2026, The University of Texas at Austin.

**Author:** Nitin Bhatnagar
**Research question:** *How well can standard machine-learning models predict nightly Airbnb listing prices in Austin, Texas from publicly available listing attributes, and which model family — regularized linear, gradient-boosted trees, or a feed-forward neural network — offers the best accuracy–interpretability tradeoff?*
**Course module:** 8 — ML for Shared Economy (with methods from Modules 2–4).
**Submission deadline:** 2026-04-20.

---

## Open the notebook in Google Colab

One notebook, organized into clearly labeled sections. Each code cell is numbered (`Step N`) so you execute one at a time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitin2468git/ut-airbnb-final-project/blob/main/notebooks/airbnb_price_prediction.ipynb)

**Notebook sections:**

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

---

## How to work on this project

1. **Clone locally (already done):** the repo lives at the path named on your Mac.
2. **Open the notebook in Colab** using the badge above.
3. **Run cells one at a time** — look for `## Step N` headers in the markdown cells. Verify each step's output before moving on.
4. **Save back to GitHub** from Colab: `File → Save a copy in GitHub...` — this writes a commit directly to `main`, so your local clone stays in sync after `git pull`.
5. The raw CSV is git-ignored — Step 1 downloads it on demand.

## Data

See [`data/README.md`](data/README.md) for dataset provenance and download instructions. The raw CSV is **not committed** — each Colab session pulls a fresh copy from Inside Airbnb.

## Reproducing the paper

```
pandoc paper/paper.md \
  --citeproc \
  --bibliography=references/references.bib \
  --csl=references/apa-7th-edition.csl \
  -o paper/paper.docx
```

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                           # where to get the data
│   └── processed/                          # intermediate pickles (git-ignored)
├── notebooks/
│   └── airbnb_price_prediction.ipynb       # single consolidated notebook
├── paper/
│   ├── outline.md                          # section skeleton with word budgets
│   ├── paper.md                            # filled in after results
│   └── figures/                            # PNGs exported by the notebook
└── references/
    └── references.bib                      # 30+ citations (>=15 scholarly)
```

## AI use disclosure

This project was scaffolded with assistance from Anthropic's Claude (Opus 4.7) for repository setup, notebook templating, and paper-outline drafting. All modeling decisions, interpretation, and final prose are the author's own work. A full disclosure appears in Appendix A of the paper.
