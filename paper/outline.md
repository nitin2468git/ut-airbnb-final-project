# Paper outline — Austin Airbnb Listing Price Prediction

Target length: **3,000–5,000 words**, ~10–12 pages at Times New Roman 12pt, 1.5 spacing.
Citation style: **APA 7th**.

---

## Abstract (~200 words)

- One-sentence motivation: short-term-rental pricing is a signature question of the shared economy.
- One-sentence research question.
- Data: Inside Airbnb Austin snapshot, ~15k listings.
- Methods: Ridge, Random Forest, XGBoost, and a small PyTorch MLP.
- Headline result: XGBoost wins on RMSE/MAE; MLP does not beat XGBoost; SHAP reveals that room type, bedrooms, and location dominate.
- Implication: gradient-boosted trees remain the right default for tabular hedonic pricing; interpretability tools narrow the gap with linear models.

---

## §1 Introduction and research background (~700 words)

**Narrative arc.** Airbnb is a canonical shared-economy platform; understanding what drives nightly price is useful for hosts, guests, regulators, and researchers. Austin is a useful local lens because of its active STR policy environment and tourism profile.

**Required citations (≥5):**
- Shared-economy framing: `sundararajan2016sharing`, `zervas2017airbnb`.
- STR-specific economics: `wachsmuth2018airbnb`, `barron2021sharing`, `horn2017regulating`.
- Hedonic-pricing theory: `rosen1974hedonic`, `sheppard1999hedonic`.
- Austin/policy context: `austin2023str`.

**Research question** (restate formally).

**Contributions** (≥3 small, honest claims):
1. First public comparison of four model families on a 2026 Austin Inside Airbnb snapshot.
2. Feature engineering that isolates location (distance to downtown, neighborhood target-encoding) from unit attributes.
3. SHAP-based feature attribution for interpretability of the winning model.

---

## §2 Research question and methods (~600 words)

**Problem formulation.** Supervised regression on log(price). Define features `X` and target `y = log1p(price)`.

**Model families and why each:**
- **Ridge regression** — interpretable baseline; coefficients have direct hedonic-pricing interpretation (`hoerl1970ridge`, `tibshirani1996lasso`).
- **Random Forest** — non-parametric ensemble, handles non-linearities (`breiman2001random`).
- **XGBoost** — state-of-the-art on tabular regression (`chen2016xgboost`).
- **MLP** — small feed-forward network; tests whether DL buys anything here (`shwartz2022tabular`, `gorishniy2021tabular`, `lecun2015deep`).

**Evaluation protocol.** 80/10/10 split stratified on price quintiles (`kohavi1995cross`). Metrics: RMSE on log-price (primary), MAE on raw USD (interpretable), R² (variance explained). All hyperparameters tuned on validation set; test set is touched once.

**Tooling:** scikit-learn (`pedregosa2011scikit`), XGBoost (`chen2016xgboost`), SHAP (`lundberg2017shap`), PyTorch (`paszke2019pytorch`).

---

## §3 Materials and data sources (~600 words)

**Data source.** Inside Airbnb Austin snapshot — `insideairbnb2026`, `cox2015airbnb`. License CC0. ~15k listings, 75+ raw columns. Pin the snapshot date.

**Ethics note.** Brief discussion of re-identification risk; this paper uses only aggregate features.

**Preprocessing pipeline** (table):

| Step | Action |
|---|---|
| 1 | Drop rows missing price; strip `$` and commas; cast to float |
| 2 | Winsorize top/bottom 1% of price |
| 3 | Apply `log1p` to price |
| 4 | Feature engineering: amenity count, description length, host tenure in days, haversine distance to downtown (30.2672, -97.7431) |
| 5 | Categorical encoding: one-hot for `room_type`, top-10 `property_type` + Other; target-encode `neighbourhood_cleansed` |
| 6 | Standard-scale numeric features |
| 7 | 80/10/10 split stratified by price quintile |

**Feature table.** Final feature count and brief description of each engineered feature.

**Supplementary citations:**
- Location value: `chen2019location`, `anselin2013spatial`.
- Text as signal: `ghose2012designing`.
- Review/reputation: `fradkin2021search`, `li2019repeat`.

---

## §4 Results (~1,000 words)

**Table 1 — Model performance on held-out test set.**

| Model | Log-price RMSE | Raw-price MAE (USD) | R² |
|---|---|---|---|
| Median baseline | _fill_ | _fill_ | _fill_ |
| Ridge | _fill_ | _fill_ | _fill_ |
| Random Forest | _fill_ | _fill_ | _fill_ |
| XGBoost | _fill_ | _fill_ | _fill_ |
| MLP | _fill_ | _fill_ | _fill_ |

**Figure 1.** Predicted vs. actual scatter for XGBoost (test set).
**Figure 2.** Residual distribution per model.
**Figure 3.** SHAP summary plot for XGBoost (top 15 features).
**Figure 4.** Log-price by neighborhood (top 15).

**Discussion of results** (as you fill these in):
- Which features dominate? Expect: `room_type=Entire home/apt`, `bedrooms`, `accommodates`, `distance_to_downtown_km`, neighborhood target-encoding.
- Where does the model fail? Likely very high-price outliers and sparse neighborhoods.
- Neighborhood effects — is there evidence of a downtown / East Austin premium?

---

## §5 Discussion and limitations (~800 words)

**Tree vs. neural net on tabular data.** Cite `shwartz2022tabular`, `gorishniy2021tabular`. Explain why MLP didn't win.

**Interpretability–accuracy tradeoff.** Ridge is transparent but biased; XGBoost is accurate but needs SHAP (`lundberg2017shap`). SHAP narrows the interpretability gap.

**Limitations (honest list):**
- Single snapshot — no temporal dynamics; prices shift seasonally.
- Selection bias — only currently active, scrape-visible listings.
- No demand-side data (guest searches, conversion).
- No image features (listing photos); literature suggests these matter (`kalehbasti2019airbnb`).
- Possible fairness concerns: historical discrimination in the platform (`edelman2017racial`) may leave price-residual imprints.

**Policy implications** (~200 words within the section): relevance to Austin STR regulation (`austin2023str`, `lee2016airbnb`), pricing transparency for hosts, and data access for planners (`gurran2017airbnb`).

---

## §6 Conclusion (~300 words)

- Recap research question and findings in one paragraph.
- Two-sentence restatement of the headline (XGBoost best; MLP no improvement; location + capacity dominate).
- Future work: temporal panel, image features, fairness audit.

---

## Appendix A — AI use disclosure

This project used Anthropic's Claude (Opus 4.7) in a Claude Code session to:
- Draft the repository scaffolding (README, requirements, .gitignore).
- Pre-populate a single Colab notebook with section headers and starter code.
- Produce an initial BibTeX seed of 35 references and this paper outline.

All empirical results, figures, numerical interpretations, and final prose were authored by the student. No model outputs were copy-pasted as paper body text without substantial editing. The session transcript is available on request.

## Appendix B — Code availability

Code lives at `https://github.com/captnitinbhatnagar/ut-airbnb-final-project`. The single notebook `notebooks/airbnb_price_prediction.ipynb` reproduces every figure and table in this paper when run on Google Colab (free tier, CPU runtime). See `README.md` for the Open-in-Colab badge.

---

## Word-budget totals (target)

| Section | Words |
|---|---|
| Abstract | 200 |
| §1 Intro | 700 |
| §2 Methods | 600 |
| §3 Data | 600 |
| §4 Results | 1,000 |
| §5 Discussion | 800 |
| §6 Conclusion | 300 |
| **Total body** | **4,200** |

Leaves room within the 3,000–5,000 band for a tight cut or expansion.
