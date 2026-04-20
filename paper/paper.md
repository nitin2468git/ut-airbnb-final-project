---
title: "Predicting Airbnb nightly prices in Austin: Ridge, Random Forest, XGBoost, and a small neural network"
author: "Nitin Bhatnagar"
course: "AI 391M, Case Studies in Machine Learning, UT Austin"
date: "April 2026"
bibliography: ../references/references.bib
csl: apa-7th-edition.csl
---

## Abstract

Short-term rental pricing serves as a handy proving ground for applied machine learning: it sits at the crossroads of hedonic pricing theory for heterogeneous goods, rich public data, and policy relevance. This work predicts nightly Airbnb rates in Austin using only the listing-page features you can see, then pits four supervised-regression models against each other on identical data splits: Ridge regression, Random Forest, XGBoost, and a compact PyTorch feed-forward network. Using the latest Austin Inside Airbnb snapshot (2025-09-16; after cleaning, 10,306 listings) with leakage-safe neighborhood target encoding and haversine distance to downtown as location signals, XGBoost comes out on top. It achieves a log-price RMSE of 0.38, a dollars MAE of $68, and an R-squared of 0.76 on the test set. The small three-layer MLP trails Ridge, echoing recent findings that gradient-boosted trees remain strong performers for medium-sized tabular data. SHAP analyses highlight capacity, location, and product type as the principal price drivers, aligning with Ridge coefficients and classic hedonic pricing.

## 1. Introduction and background

Short-term rental platforms like Airbnb are now a staple of the urban shared economy. Guests treat them as hotel alternatives; hosts use them to supplement long-term rents; cities view them as policy challenges. For me, pricing is interesting because it sits at consumer microeconomics, urban form, and applied ML. A listing's nightly price isn't set by an idealized clearing market; hosts set it, guests accept or reject, and reviews, demand, and competition reshape it over time. That makes it a natural testbed for supervised regression on tabular data and a real stake for hosts and planners.

I focus on Austin, Texas, for three reasons: a large, stable Airbnb market; an active, data-rich policy landscape (Austin has updated its short-term rental rules several times in the last decade and publishes enforcement data); and it's where I live, so I can sanity-check geographic patterns like downtown, the Drag, East Austin, and the Domain against model output.

Hedonic pricing theory underpins the analysis. Rosen (1974) formalized how a heterogeneous good's price decomposes into the marginal contributions of its attributes, and Sheppard (1999) surveyed housing-market applications. A short-term rental is a natural fit: a downtown two-bedroom differs from a suburban shared room, and the combination of attributes sets the price. Prior Airbnb work has used hedonic-style regressions to study neighborhood effects, causal impacts on rents, and competitive effects on hotels (Wachsmuth & Weisler, 2018; Barron et al., 2021; Horn & Merante, 2017; Zervas et al., 2017). A broader urban-planning context and the evolution of pricing research are well documented (Gurran & Phibbs, 2017; Sundararajan, 2016). In ML for Airbnb pricing, newer work has blended tabular features with sentiment signals or applied neural nets to related housing markets (Kalehbasti et al., 2019; Xu & Zhang, 2022). The toolkit shift, from linear hedonic models to gradient-boosted trees and, to a degree, deep learning (LeCun et al., 2015), matters because stakeholders often want to understand why a price is what it is.

This project's contribution is modest but precise: a clean, reproducible four-model comparison on the latest available Austin Inside Airbnb data (Sept 2025), with feature engineering that separates location effects from unit attributes, and side-by-side reporting of accuracy and interpretability via SHAP. Specific contributions are: (1) an up-to-date Austin pricing baseline using Ridge, Random Forest, XGBoost, and a small neural network on identical data splits; (2) a feature-engineering pipeline that uses leakage-safe neighborhood target encoding and haversine distance to downtown as location signals; and (3) a SHAP-based interpretation showing which features dominate predictions and by how much.

The paper is organized as follows: Section 2 outlines the research question, model families, and evaluation protocol. Section 3 describes the data source and preprocessing. Section 4 presents test-set accuracy and SHAP results. Section 5 discusses why tree-based models tend to win, where they fall short, and the limitations. Section 6 concludes.

## 2. Research question and methods

Two-part question: first, how accurately can a supervised model predict an Austin Airbnb's nightly price from listing-page attributes (capacity, room type, location, host tenure, amenities, and review stats)? Second, does the answer depend on the model family, and what is the trade-off between accuracy and interpretability?

The problem is framed as supervised regression on a transformed price. Prices are right-skewed and span wide ranges, so predicting raw dollars would overweight the luxury tail. I predict the natural log of one plus the price, stabilizing variance and making additive residuals meaningful; dollars are recovered by exponentiating when needed.

Four models with increasing capacity are evaluated, chosen to span interpretability to flexibility. The modeling choices and hyperparameters follow standard practices from Hastie et al. (2009) and Géron (2022), with neural-network specifics from Goodfellow et al. (2016).

- Ridge regression: linear with L2 penalty (Hoerl & Kennard, 1970; Tibshirani, 1996). Ridge coefficients map to hedonic price contributions. Regularization is tuned via 5-fold cross-validation on the training set.
- Random Forest: an ensemble of decision trees that handles non-linearities and interactions and provides feature importances (Breiman, 2001). Two values for n_estimators (200, 400) and two max_depth values (unlimited, 20) are tested; the best on validation log-RMSE is kept.
- XGBoost: regularized gradient boosting (Chen & Guestrin, 2016). Up to 2,000 trees, learning rate 0.05, max_depth 6, subsample and colsample_bytree 0.9, early stopping after 50 rounds on the validation set.
- Small MLP: three hidden layers (256, 128, 64) with ReLU, 0.2 dropout, trained with Adam (lr 0.001, weight decay 0.0001), batch size 256, up to 80 epochs with early stopping (patience 8), implemented in PyTorch (Paszke et al., 2019).

Evaluation uses an 80/10/10 train/validation/test split, stratified by price quintile to ensure coverage across the distribution (Kohavi, 1995). The test set is scored only once, after model selection.

Metrics reported on the test set for each model: log-price RMSE (the optimization target), MAE on the raw price (dollar-scale, intuitive), and R^2 on log-price. For the winner, SHAP values are computed on a random 1,000-row test sample to explain per-feature contributions (Lundberg & Lee, 2017).

The workflow relies on scikit-learn (Pedregosa et al., 2011), XGBoost, PyTorch, and SHAP, with pandas (pandas development team, 2024) for data handling. It runs on Google Colab's free CPU runtime (Google, 2026) in under five minutes.

## 3. Materials and data sources

**Data source and snapshot**

All data come from Inside Airbnb, which scrapes publicly visible listings and republishes them under CC0 (Cox, 2026). The Austin snapshot dated 2025-09-16 is used, the most recent at the time. Inside Airbnb has been widely used in short-term rental research and policy work. The full listings file is a gzipped CSV with 10,533 rows and 79 columns, covering every active Austin listing on the snapshot date.

No direct scraping of Airbnb or paid data is used. Inside Airbnb's methods are documented, the data are reproducible from public archives, and this choice helps reproducibility.

**Target variable**

The target is the nightly price in USD, stored in the raw CSV as a string like $1,234. I strip the currency symbol and separators and cast to float. Before cleaning, prices range from near zero (likely test or inactive listings) to above $10,000 for luxury properties. The median is $162 and the mean is $238, echoing the right-skew typical of this market. I work with log(1 + price) to stabilize variance.

**Cleaning and winsorization**

Two steps: drop rows with missing price, then winsorize at the 1st and 99th percentiles to remove both zero-price entries and extreme outliers. After this, there are 10,306 rows, with prices from $28 to $2,432. This keeps legitimate high-end listings while preventing any single listing from dominating training.

**Feature engineering**

From 79 features, nine numeric attributes are used: bedrooms, beds, bathrooms, accommodates, minimum nights, number of reviews, review scores rating, reviews per month, and host listings count. Four engineered features are added: an amenity count parsed from the amenities field, the description length as a proxy for marketing effort (supported by studies linking textual content to listing quality; Ghose et al., 2012), host tenure in days, and haversine distance to the Texas State Capitol (downtown center).

Categorical variables are one-hot encoded: room_type (entire home, private room, shared room, hotel room) and a bucketed property-type covering the ten most common types plus "Other." Neighborhoods are handled with leakage-safe target encoding: each neighborhood is replaced by the mean log-price of training-fold listings in that neighborhood. If a validation or test neighborhood hasn't appeared in training, it gets the global training-fold mean. The target encoding is fit only on the training fold to prevent leakage.

After median imputations for missing numeric values and assembling one-hot encodes, the final feature matrix has 29 columns. The training, validation, and test partitions contain 8,244; 1,031; and 1,031 rows, respectively, stratified by price quintile. Processed splits are saved as pickle files to ensure identical data across model runs.

## 4. Results

**Headline accuracy**

The results (compared to a median baseline) show XGBoost as the clear winner on all three metrics: log-RMSE 0.3819, MAE $68.07, and R^2 0.762. Random Forest follows closely at 0.4051 / $71.62 / 0.732. Ridge lands at 0.4863 / $87.93 / 0.614, and the MLP underperforms Ridge at 0.4945 / $99.80 / 0.601. The median baseline sits at RMSE 0.7883 with an $128.66 MAE and a slightly negative R^2, reflecting the log-price transformation's effect.

Table 1. Test-set performance of all four models plus the median baseline. Bold marks the best model on each metric.

| Model | log-RMSE | MAE (USD) | R^2 |
|---|---|---|---|
| Median baseline | 0.7883 | $128.66 | -0.014 |
| Ridge regression | 0.4863 | $87.93 | 0.614 |
| Random Forest | 0.4051 | $71.62 | 0.732 |
| **XGBoost** | **0.3819** | **$68.07** | **0.762** |
| MLP | 0.4945 | $99.80 | 0.601 |

A figure mirrors Table 1 as side-by-side bars for log-RMSE and dollar MAE, showing the same ordering: tree-based models win consistently.

![Figure 1. Test-set log-price RMSE (left) and mean absolute error in USD (right) across all four models plus the median baseline. Lower is better on both panels.](figures/fig_model_comparison.png)

XGBoost's MAE of about $68 means the model is off by roughly sixty-eight dollars per night on held-out data, with a median price of $162. That's a meaningful improvement for flagging mispriced listings, though not a substitute for a full revenue-management system.

**Does the MLP help?**

The standout finding is that the MLP does not beat tree-based models. It trails Random Forest by about $32 in MAE and is worse on every metric. This aligns with literature suggesting gradient-boosted trees excel on medium-sized tabular data (Shwartz-Ziv & Armon, 2022; Gorishniy et al., 2021). The training curve shows the network converges quickly and then plateaus, indicating the shortfall is not a training issue but an inductive-bias mismatch: trees' structure better matches hedonic pricing patterns in this setting.

![Figure 2. MLP training and validation log-RMSE by epoch. The model converges within about ten epochs and early-stopping triggers shortly after.](figures/fig_training_curve.png)

**What drives XGBoost's predictions?**

SHAP summary plots reveal stable, interpretable top drivers. Bathrooms dominate: more bathrooms push price up, with large positive SHAP contributions. The target-encoded neighborhood is next, confirming location as a primary predictor. Accommodates and bedrooms follow, reinforcing capacity as a key driver. Distance to downtown is fifth, farther listings tend to be cheaper, consistent with central-location premiums. Reviews per month shows a negative contribution on average, likely reflecting higher turnover for cheaper, smaller units.

![Figure 3. SHAP summary plot for XGBoost on a random 1,000-row sample of the test set. Features ordered top-to-bottom by mean absolute SHAP value. Dot color encodes feature value (red high, blue low); horizontal position encodes SHAP contribution to the log-price prediction.](figures/fig3_shap_xgboost.png)

Ridge coefficients tell a similar story: large positive values appear for neighborhood, bathrooms, accommodates, entire-home type, and certain property-type categories, with negative signals for shared rooms, private-room listings, and higher review frequency. The convergence of two very different models on the same feature importance strengthens the interpretation.

**Residual structure**

Predicted vs. actual log-prices show XGBoost clustering tightly around the diagonal in the middle of the distribution (roughly $55–$400), with some under-prediction at the upper tail (high-end listings).

![Figure 4. XGBoost predicted vs actual log-price on the test set. The dashed red line marks perfect prediction. The cloud is tight along the diagonal through most of the distribution, with under-prediction in the upper tail.](figures/fig1_xgb_pred_vs_actual.png)

Residual densities indicate XGBoost and Random Forest yield the tightest error distributions, with Ridge and MLP showing heavier tails. No systematic bias is detected.

![Figure 5. Test-set residual (actual minus predicted log-price) densities for all four models. XGBoost and Random Forest are tightest around zero; Ridge and MLP have heavier tails.](figures/fig2_residuals.png)

## 5. Discussion

**What the results imply**

The main takeaway: on the 2025 Austin Inside Airbnb data, XGBoost delivers the most accurate predictions among the four models, with a practical MAE of about $68 and a median listing price of $162. The SHAP explanations line up with hedonic-pricing expectations: capacity, location, and product differentiation dominate, and Ridge coefficients align with SHAP contributions. The agreement between a linear and a non-linear model on what moves price is strong evidence for the interpretability of the results.

The neural network result is informative: a well-tuned MLP underperforms compared to Ridge here. It reinforces the view that, for medium-sized tabular data, deeper networks don't automatically win; the inductive biases of gradient-boosted trees often fit hedonic pricing structures better.

**Interpretability vs. accuracy**

The accuracy gap between Ridge and XGBoost is modest (roughly four hundredths in log-RMSE, about twenty dollars in MAE). In a hedonic-pricing context from 1999, Ridge's interpretability would have been the deciding factor; today, SHAP allows similar interpretability for XGBoost. Here, Ridge coefficients and SHAP values tell the same story, suggesting the interpretability trade-off is small on this dataset. In other settings with strong feature interactions, SHAP may reveal a different importance picture, and in such cases XGBoost would be the preferred starting point with SHAP for interpretation.

**Policy implications**

Austin has continuously revised its short-term rental ordinance (Nieuwland & van Melik, 2020; City of Austin, 2023). While this model isn't a policy tool by itself, its outputs are relevant to policy discussions. The neighborhood encoding shows up among the top SHAP contributors, quantifying the location premium hosts capture. This aligns with broader urban-economics findings about the impact of local amenities (Chen & Rosenthal, 2008), and with the spatial-econometric framing of Anselin (1988). The presence of entire-home listings as a major predictor mirrors regulatory focus on owner-occupied vs non-owner-occupied types (Lee, 2016) and supports discussions about how pricing signals relate to housing affordability concerns.

**Limitations**

Four limitations to flag: (1) a single monthly snapshot prevents seasonality and time trends from being modeled; a panel across multiple months would help separate cross-sectional and time-varying effects. (2) Inside Airbnb is scraped data, not the platform's internal calendar data, so the observed price is the public listing price, not actual booking dynamics. (3) The features are purely tabular; listing images and text likely carry predictive signals that a multi-modal model could exploit. (4) The train/validation/test split is random, not geographic; holding out entire neighborhoods would test geographic generalization. Broader fairness concerns (Edelman et al., 2017) and reputation-system dynamics (Fradkin et al., 2021) and guest-side location preferences (Yang et al., 2018) also bound the interpretation.

**Future work**

Three avenues are worth pursuing with more time: (a) a multi-snapshot panel to enable difference-in-differences analyses of ordinance changes; (b) an image-augmented model using a pretrained vision encoder to capture visual quality signals from listing photos; and (c) a cross-city study training on Austin and evaluating on Houston, Dallas, and San Antonio to see which features transfer.

## 6. Conclusion

I built a reproducible, Colab-ready pipeline that predicts Austin Airbnb nightly prices from publicly observable attributes and compared four model families on identical data splits. XGBoost achieved the best test performance (log-RMSE 0.38; MAE $68; median price $162), with Random Forest a close second. Ridge, despite its simplicity, outperformed a three-layer MLP, reinforcing the broader finding that gradient-boosted trees are a solid default for medium-sized tabular regression.

SHAP analyses on the XGBoost model highlighted capacity, location, and product type as the dominant price drivers, matching hedonic-pricing theory. The agreement between Ridge coefficients and SHAP explanations strengthens the interpretation.

Practically, this shows that a modest, off-the-shelf ML pipeline can produce accurate, interpretable estimates of a listing's market-clearing nightly rate from publicly visible attributes, without proprietary data or heavy computational resources. It lowers barriers to applying this approach to other cities and to informing short-term rental policy debates with quantitative evidence.

The code and paper are available at https://github.com/nitin2468git/ut-airbnb-final-project, reproducing every figure and table from the same data snapshot in under five minutes on a free-tier Google Colab CPU runtime.

---

## References

Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer Academic. https://doi.org/10.1007/978-94-015-7799-1

Barron, K., Kung, E., & Proserpio, D. (2021). The effect of home-sharing on house prices and rents: Evidence from Airbnb. *Marketing Science*, *40*(1), 23–47. https://doi.org/10.1287/mksc.2020.1227

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Chen, Y., & Rosenthal, S. S. (2008). Local amenities and life-cycle migration: Do people move for jobs or fun? *Journal of Urban Economics*, *64*(3), 519–537. https://doi.org/10.1016/j.jue.2008.05.005

City of Austin Development Services Department. (2023). *Short-term rental ordinance and enforcement report*. https://www.austintexas.gov/department/short-term-rentals

Cox, M. (2026). *Inside Airbnb: Austin, TX listings snapshot* [Dataset]. Inside Airbnb. https://insideairbnb.com/austin

Edelman, B., Luca, M., & Svirsky, D. (2017). Racial discrimination in the sharing economy: Evidence from a field experiment. *American Economic Journal: Applied Economics*, *9*(2), 1–22. https://doi.org/10.1257/app.20160213

Fradkin, A., Grewal, E., & Holtz, D. (2021). Reciprocity and unveiling in two-sided reputation systems: Evidence from an experiment on Airbnb. *Marketing Science*, *40*(6), 1013–1029. https://doi.org/10.1287/mksc.2021.1311

Géron, A. (2022). *Hands-on machine learning with scikit-learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Ghose, A., Ipeirotis, P. G., & Li, B. (2012). Designing ranking systems for hotels on travel search engines by mining user-generated and crowdsourced content. *Marketing Science*, *31*(3), 493–520. https://doi.org/10.1287/mksc.1110.0700

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Google. (2026). *Google Colaboratory*. https://colab.research.google.com

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. In *Advances in Neural Information Processing Systems 34* (pp. 18932–18943).

Gurran, N., & Phibbs, P. (2017). When tourists move in: How should urban planners respond to Airbnb? *Journal of the American Planning Association*, *83*(1), 80–92. https://doi.org/10.1080/01944363.2016.1249011

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, *12*(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

Horn, K., & Merante, M. (2017). Is home sharing driving up rents? Evidence from Airbnb in Boston. *Journal of Housing Economics*, *38*, 14–24. https://doi.org/10.1016/j.jhe.2017.08.002

Kalehbasti, P. R., Nikolenko, L., & Rezaei, H. (2019). Airbnb price prediction using machine learning and sentiment analysis. *arXiv preprint arXiv:1907.12665*. https://arxiv.org/abs/1907.12665

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In *IJCAI*, *14*(2), 1137–1145.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, *521*(7553), 436–444. https://doi.org/10.1038/nature14539

Lee, D. (2016). How Airbnb short-term rentals exacerbate Los Angeles's affordable housing crisis. *Harvard Law & Policy Review*, *10*, 229–253.

Li, J., Moreno, A., & Zhang, D. J. (2015). *Agent behavior in the sharing economy: Evidence from Airbnb* (Ross School of Business Working Paper No. 1298). University of Michigan. https://ssrn.com/abstract=2708279

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).

Nieuwland, S., & van Melik, R. (2020). Regulating Airbnb: How cities deal with perceived negative externalities of short-term rentals. *Current Issues in Tourism*, *23*(7), 811–825. https://doi.org/10.1080/13683500.2018.1504899

The pandas development team. (2024). *pandas: Powerful Python data analysis toolkit* (Version 2.x) [Computer software]. https://pandas.pydata.org

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure competition. *Journal of Political Economy*, *82*(1), 34–55. https://doi.org/10.1086/260169

Sheppard, S. (1999). Hedonic analysis of housing markets. In *Handbook of regional and urban economics* (Vol. 3, pp. 1595–1635). Elsevier. https://doi.org/10.1016/S1574-0080(99)80010-8

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, *81*, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011

Sundararajan, A. (2016). *The sharing economy: The end of employment and the rise of crowd-based capitalism*. MIT Press.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, *58*(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

Wachsmuth, D., & Weisler, A. (2018). Airbnb and the rent gap: Gentrification through the sharing economy. *Environment and Planning A: Economy and Space*, *50*(6), 1147–1170. https://doi.org/10.1177/0308518X18778038

Xu, X., & Zhang, Y. (2022). Rent index forecasting through neural networks. *Journal of Economic Studies*, *49*(8), 1321–1339. https://doi.org/10.1108/JES-06-2021-0316

Yang, Y., Mao, Z., & Tang, J. (2018). Understanding guest satisfaction with urban hotel location. *Journal of Travel Research*, *57*(2), 243–259. https://doi.org/10.1177/0047287517691153

Zervas, G., Proserpio, D., & Byers, J. W. (2017). The rise of the sharing economy: Estimating the impact of Airbnb on the hotel industry. *Journal of Marketing Research*, *54*(5), 687–705. https://doi.org/10.1509/jmr.15.0204

---

## Appendix A. AI use disclosure

This project used Anthropic's Claude in a Claude Code session for four auxiliary purposes. First, repository and notebook scaffolding: the initial GitHub repo layout, the single-notebook cell structure, and the first drafts of cell-level code were generated by Claude; I reviewed every cell, fixed bugs (including the Inside Airbnb download URL and a SHAP / NumPy version conflict in Colab), and confirmed each step's output against my own reading of the data. Second, figure generation: the five figures in Section 4 are exported directly from the notebook I executed end-to-end on Google Colab. Third, bibliography construction and verification: Claude produced an initial BibTeX seed list, and I used web-search tools (via Claude) to verify every reference against its source; five seed entries that did not match a real publication or had the wrong venue/year were replaced with real, verifiable sources. Fourth, light editorial assistance on the paper draft: Claude suggested places to insert citations and to embed figures into the body I had written, which I then reviewed and accepted. The prose of this paper is my own writing. All empirical findings, model-selection decisions, numerical results, and the final written argument are my own work. The session transcript is available on request.

## Appendix B. Code availability

All code is available at https://github.com/nitin2468git/ut-airbnb-final-project. The single notebook `notebooks/airbnb_price_prediction.ipynb` reproduces every figure and table in this paper when run on Google Colab's free CPU runtime. See the repository README for the Open-in-Colab badge and step-by-step usage instructions.
