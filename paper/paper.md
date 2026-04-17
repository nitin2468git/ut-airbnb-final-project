---
title: "Predicting Airbnb nightly prices in Austin: a comparison of Ridge, Random Forest, XGBoost, and a small neural network"
author: "Nitin Bhatnagar"
course: "AI 391M — Case Studies in Machine Learning, UT Austin"
date: "April 2026"
bibliography: ../references/references.bib
csl: apa-7th-edition.csl
---

## Abstract

Short-term rental pricing is a useful testbed for applied machine learning because it combines a well-understood economic structure (hedonic pricing of heterogeneous goods) with a rich, publicly available dataset and immediate policy relevance. In this paper I predict nightly Airbnb listing prices in Austin, Texas from the attributes that are visible on the listing page, and I compare four supervised-regression models on the same data splits: Ridge regression, Random Forest, XGBoost, and a small PyTorch feed-forward neural network. I use the most recent Austin Inside Airbnb snapshot (2025-09-16, 10,306 listings after cleaning) with a leakage-safe target encoding for neighborhood and a haversine distance to downtown as location features. XGBoost achieves the best test-set performance, with a log-price root mean squared error of 0.38 and a mean absolute error of $68 on the raw dollar scale, and a coefficient of determination of 0.76. A three-layer MLP underperforms Ridge regression, consistent with recent literature arguing that gradient-boosted trees remain the default for medium-sized tabular data. SHAP attributions identify capacity, location, and product type as the dominant drivers of predicted price, in agreement with both the Ridge coefficients and classical hedonic pricing theory.

---

## 1. Introduction and research background

Short-term rental platforms like Airbnb have become a standard fixture of the urban shared economy. Guests use them as hotel substitutes, hosts use them as an income stream on top of long-term residential rent, and cities treat them as a new policy problem. For a researcher, what makes short-term rental pricing interesting is that it sits at the intersection of three things I care about: consumer microeconomics, urban form, and applied machine learning. A listing's nightly price is not set by a clearing market in the textbook sense. It is posted by the host, accepted or rejected by guests, and adjusted over time in response to reviews, local demand, and competing listings on the same block. That makes it a useful testbed for supervised regression on tabular data, and it makes predictive accuracy a question with real stakes for hosts deciding what to charge and for planners deciding how to regulate.

I study Austin, Texas. Austin is a useful local lens for three reasons. It has a large and stable Airbnb market. It has an active short-term rental policy environment: the City of Austin has revised its short-term rental ordinance multiple times in the last decade and continues to publish enforcement data (City of Austin, 2023). And it is where I live, which means the local geography (downtown, the Drag, East Austin, the Domain) is something I can sanity-check against the model's predictions rather than taking them on faith.

The economic theory behind this work is hedonic pricing. Rosen (1974) formalized the idea that the price of a heterogeneous good can be decomposed into the marginal contributions of its attributes, and Sheppard (1999) surveyed how this framework has been applied to housing markets. A short-term rental listing fits naturally into this frame: an entire two-bedroom home near downtown is not the same product as a shared room in the suburbs, and the attribute bundle is what determines the price. Prior work on Airbnb specifically has used hedonic-style regressions to study neighborhood effects (Wachsmuth & Weisler, 2018), the causal effect of Airbnb on local rents (Barron et al., 2021; Horn & Merante, 2017), and the platform's competitive effect on hotels (Zervas et al., 2017). Gurran and Phibbs (2017) review how urban planners should respond to these effects, and Sundararajan (2016) gives the broader shared-economy context. Applied machine-learning work on Airbnb pricing is a newer, smaller literature: Kalehbasti et al. (2019) combine tabular features with review-sentiment features, Xu and Zhang (2022) apply neural-network approaches to rent-index forecasting on a related housing market, and Yang et al. (2018) examine how hotel-location characteristics translate into guest-side satisfaction.

What has shifted since Rosen's era is the toolkit. Classical hedonic regressions are linear in feature space (often after a log transform of price) and report clean coefficients that map directly to a dollar contribution per feature. Modern gradient-boosted trees like XGBoost (Chen & Guestrin, 2016) consistently outperform linear models on tabular regression, but at the cost of interpretability. Deep learning has made enormous gains on images and text (LeCun et al., 2015) but, as Shwartz-Ziv and Armon (2022) and Gorishniy et al. (2021) argue, has not reliably beaten gradient-boosted trees on tabular data. For pricing applications this matters: a regulator or a host wants to know not just the predicted price, but why. The SHAP framework (Lundberg & Lee, 2017) has partially closed the interpretability gap for tree models, which is why I use it here.

The contribution of this project is narrow and honest. I am not claiming a new method. What I am doing is running a clean, reproducible, four-model comparison on the most recent publicly available Austin Inside Airbnb snapshot (September 2025), with feature engineering that separates location effects from unit attributes, and reporting both accuracy and SHAP-based interpretability side by side. The specific contributions are: (1) an up-to-date empirical baseline for Austin short-term rental pricing with Ridge, Random Forest, XGBoost, and a small feed-forward neural network on the same data splits; (2) a feature-engineering pipeline that uses a leakage-safe target encoding for neighborhood and a haversine distance to downtown as its location signals; and (3) a SHAP analysis that identifies which features dominate the winning model's predictions and at what magnitudes.

The rest of the paper is organized as follows. Section 2 states the research question and explains the four model families and the evaluation protocol. Section 3 describes the Inside Airbnb data source and the preprocessing pipeline. Section 4 reports accuracy on the held-out test set and the SHAP results. Section 5 discusses why the tree-based model wins, where it fails, and what the limitations are. Section 6 concludes.

---

## 2. Research question and methods

My research question has two parts. First, how accurately can a supervised model predict the nightly price of an Austin Airbnb listing from the attributes that are visible on the listing page, meaning capacity, room type, location, host tenure, amenity count, and review statistics? Second, does that answer depend on which model family I use, and if so, what is the cost of picking accuracy over interpretability?

I cast the problem as supervised regression on a transformed price. Airbnb nightly prices are right-skewed and span roughly two orders of magnitude, so fitting on raw dollars would weight the loss heavily toward the luxury tail. Instead I predict the natural log of one plus the price, which stabilizes variance and puts the models on a scale where additive residuals are meaningful. When I want to report a dollar number I exponentiate the prediction back.

I fit four models of increasing capacity. I picked them to span the interpretability-to-flexibility spectrum that the second half of my research question is about. The model choices and hyperparameter conventions follow standard practice described in Hastie et al. (2009) and in the applied treatment of Géron (2022), with neural-network specifics drawn from Goodfellow et al. (2016).

The first model is Ridge regression, a linear model with an L2 penalty on its coefficients (Hoerl & Kennard, 1970; Tibshirani, 1996). Ridge coefficients have a direct hedonic-pricing interpretation: holding every other feature fixed, a one-unit change in a given feature shifts the predicted log-price by that feature's coefficient. I tune the regularization strength by 5-fold cross-validation on the training fold over a log-spaced grid.

The second is a Random Forest, a non-parametric ensemble of decision trees (Breiman, 2001). Random forests handle non-linearities and feature interactions without explicit specification and give feature importances. I sweep two values of `n_estimators` (200 and 400) and two values of `max_depth` (unlimited and 20) on the validation fold and keep the configuration with the lowest validation log-RMSE.

The third is XGBoost, a regularized gradient-boosted tree ensemble (Chen & Guestrin, 2016). It is the standard modern baseline for tabular regression and the model I expect to win. I fit up to 2,000 candidate trees with a learning rate of 0.05, maximum tree depth 6, subsample and column-subsample both at 0.9, and early-stopping patience of 50 rounds on the validation fold.

The fourth is a small feed-forward neural network, or MLP, implemented in PyTorch (Paszke et al., 2019). The network has three hidden layers with 256, 128, and 64 units, ReLU activations, and 0.2 dropout between layers. I train with Adam at learning rate 0.001 and weight decay 0.0001, batch size 256, for up to 80 epochs with early stopping on validation log-RMSE (patience 8). Shwartz-Ziv and Armon (2022) and Gorishniy et al. (2021) argue that on tabular data a network of this sort usually does not outperform a well-tuned gradient-boosted ensemble, but I wanted to check this on my own data rather than cite it and move on.

To evaluate the four models consistently I split the cleaned dataset into training, validation, and test partitions in an 80 / 10 / 10 ratio. I stratify the split on price quintile so that each partition contains listings from across the price distribution (Kohavi, 1995). The test set is scored exactly once at the end and is never used during model selection or hyperparameter tuning.

I report three test-set metrics for every model. The first is root mean squared error on log-price, which is the quantity I optimize against. The second is mean absolute error on the raw dollar price, computed by exponentiating the log-price prediction back to dollars; this is the metric a non-technical reader can interpret directly as "the model is off by about this many dollars on average." The third is the coefficient of determination on log-price, which reports the fraction of log-price variance the model explains on unseen data.

For the winning model I additionally compute SHAP values (Lundberg & Lee, 2017) on a random 1,000-row sample of the test set. SHAP gives a per-feature, per-prediction attribution that is locally faithful and globally coherent, and it is what I use in Section 4 to say which features matter and at what magnitude.

The full pipeline uses scikit-learn (Pedregosa et al., 2011), XGBoost (Chen & Guestrin, 2016), PyTorch (Paszke et al., 2019), and shap (Lundberg & Lee, 2017), with pandas (pandas development team, 2024) for data handling. All of it runs on a free-tier Google Colab CPU runtime (Google, 2026) in under five minutes.

---

## 3. Materials and data sources

### Data source and snapshot

All data in this paper come from Inside Airbnb, an open-data project that scrapes publicly visible Airbnb listings and republishes them under CC0 (Cox, 2026). I use the Austin, Texas snapshot dated 2025-09-16, which is the most recent release available at the time of writing. Inside Airbnb has been used extensively in the academic literature on short-term rentals, including by Wachsmuth and Weisler (2018), Horn and Merante (2017), Barron et al. (2021), and the Austin-focused policy work of Nieuwland and van Melik (2020). The full listings file is a gzipped CSV of 10,533 rows and 79 columns, covering every active Austin listing on the snapshot date.

I deliberately did not scrape Airbnb directly or use any paid data provider. Inside Airbnb's scraper and schema are documented, the data is reproducible from the project's public archive, and using it makes this paper easy for others to replicate.

### Target variable

The target is the nightly listing price in U.S. dollars, stored in the raw CSV as a string (for example, `$1,234.00`). I strip the currency symbol and thousands separator and cast to a float. Before cleaning, prices range from a handful of $0 listings (which appear to be test accounts or inactive listings) up to over $10,000 per night for luxury properties in West Austin. The median is $162 and the mean is $238, consistent with the right-skewed distribution typical of short-term rental data. I work with the natural log of one plus the price to stabilize variance and reduce the influence of the upper tail during model fitting.

### Cleaning and winsorization

I apply two cleaning steps before modeling. First, I drop rows where the price field is missing. Second, I winsorize the price at the 1st and 99th percentiles, which removes both the $0 test listings and the ultra-luxury long tail. After winsorization the dataset has 10,306 rows with prices from $28 to $2,432 per night. These bounds are wide enough to keep legitimate high-end listings and narrow enough that no single listing dominates the loss during training.

### Feature engineering

From the 79 raw columns I selected nine numeric features that were present and mostly populated across listings: bedrooms, beds, bathrooms, accommodates (maximum guests), minimum nights, number of reviews, review scores rating, reviews per month, and host listings count. I added four engineered features. The first is an amenity count, parsed from the JSON-list-as-string amenities field. The second is the character length of the listing description, which I use as a rough proxy for how much effort the host put into marketing; Ghose et al. (2012) find that user-generated textual content is a reliable signal of listing quality on travel search engines, so a length proxy is a reasonable first pass without a full NLP pipeline. The third is host tenure in days, computed as the difference between the snapshot date and the host's join date. The fourth is the haversine distance in kilometers from each listing's latitude and longitude to the Texas State Capitol at (30.2672, -97.7431), which I treat as the center of downtown Austin.

For categorical variables I use one-hot encoding on `room_type` (four categories: entire home, private room, shared room, hotel room) and on a property-type variable bucketed to the ten most common types plus "Other" (for example, entire home, entire condominium, private room in home, private room in condominium, and so on). Neighborhood is handled separately: Austin's Inside Airbnb snapshot has 59 distinct neighborhoods, which is too many for stable one-hot encoding at this dataset size. Instead I use a leakage-safe target encoding, where each neighborhood is replaced by the mean log-price of the training-fold listings in that neighborhood. Neighborhoods in the validation or test fold that do not appear in training are filled with the training-fold global mean. The target encoding is fit on the training fold only, so no test-fold price information leaks into the model at training time.

After imputing medians for any remaining missing numeric values and assembling one-hot dummies, the final feature matrix has 29 columns. The resulting training, validation, and test partitions contain 8,244, 1,031, and 1,031 rows, stratified by price quintile so that each partition sees listings from across the price distribution. I persist the processed splits as pickle files so every model in Section 2 trains on exactly the same data.

---

## 4. Results

### Headline accuracy

Table 1 reports test-set performance for all four models plus the median baseline. XGBoost is the best model on all three metrics: a log-price RMSE of 0.3819, a mean absolute error of $68.07 on the raw dollar scale, and an $R^2$ of 0.762 on log-price. Random Forest is a close second at 0.4051 / $71.62 / 0.732. Ridge regression is a distant third at 0.4863 / $87.93 / 0.614, and the MLP actually underperforms Ridge at 0.4945 / $99.80 / 0.601. The median baseline lands where it should, at RMSE 0.7883 and an $R^2$ that is slightly negative because the median of the training log-price happens to be a worse predictor of the test log-price than the global test mean.

Table 1. Test-set performance of all four models plus the median baseline. Bold marks the best model on each metric.

| Model | log-RMSE | MAE (USD) | $R^2$ |
|---|---|---|---|
| Median baseline | 0.7883 | $128.66 | -0.014 |
| Ridge regression | 0.4863 | $87.93 | 0.614 |
| Random Forest | 0.4051 | $71.62 | 0.732 |
| **XGBoost** | **0.3819** | **$68.07** | **0.762** |
| MLP | 0.4945 | $99.80 | 0.601 |

Figure 1 shows the same table as a side-by-side bar chart of log-RMSE and dollar MAE. The ordering is identical on both metrics, which is reassuring: whatever scale a reader cares about, the tree-based models are the winners.

![Figure 1. Test-set log-price RMSE (left) and mean absolute error in USD (right) across all four models plus the median baseline. Lower is better on both panels.](figures/fig_model_comparison.png)

XGBoost's MAE of $68 is the practical headline: on the held-out test set, the model is off by about sixty-eight dollars per night on average. Given that the median Austin nightly rate is $162, that corresponds to a relative error of roughly 42 percent in the middle of the distribution. That is not accurate enough to replace a revenue-management system, but it is good enough to flag listings that are substantially under- or over-priced relative to comparable properties, which is the use case a host-facing pricing dashboard would care about.

### Does the MLP buy anything?

The cleanest empirical finding in this paper is that the MLP does not beat either tree-based model. In fact it trails Random Forest by $32 in dollar MAE and underperforms Ridge on every metric. This matches the broader argument of Shwartz-Ziv and Armon (2022) and Gorishniy et al. (2021) that on medium-sized tabular data, gradient-boosted trees remain the state of the art. My dataset has about ten thousand training rows and twenty-nine features, which is squarely in the regime where those authors predict trees will win. I ran the MLP because I wanted to check the claim on my own data rather than cite it. The training curve (Figure 2) shows that the model converged cleanly within about ten epochs and then plateaued, so the underperformance is not a training failure. It is a capacity-versus-inductive-bias story: gradient-boosted trees have an inductive bias that is well matched to hedonic pricing on small-to-medium tabular data, and an unspecialized MLP of this size does not have it.

![Figure 2. MLP training and validation log-RMSE by epoch. The model converges within about ten epochs and early-stopping triggers shortly after.](figures/fig_training_curve.png)

### What drives the XGBoost predictions?

Figure 3 is the SHAP summary plot for XGBoost evaluated on a random 1,000-row sample of the test set. Each row corresponds to a feature. Each dot is a single prediction, with its horizontal position showing how much that feature pushed that prediction up or down on the log-price scale and its color showing whether the feature value was high (red) or low (blue).

![Figure 3. SHAP summary plot for XGBoost on a random 1,000-row sample of the test set. Features ordered top-to-bottom by mean absolute SHAP value. Dot color encodes feature value (red high, blue low); horizontal position encodes SHAP contribution to the log-price prediction.](figures/fig3_shap_xgboost.png)

The top features are stable and interpretable. Number of bathrooms dominates: listings with more bathrooms are consistently predicted to be more expensive, with some of the largest positive SHAP contributions in the plot. Next is the target-encoded neighborhood variable, which confirms that location in Austin is a first-order predictor of nightly price. Accommodates (maximum guest count) and bedrooms follow, reinforcing that capacity is the second dominant driver. Distance to downtown is fifth: listings far from the Capitol are systematically predicted to be cheaper, which matches the hedonic-pricing expectation that central locations command a premium.

Reviews per month has a negative sign on average, which is counterintuitive until you consider selection: listings with very high turnover tend to be smaller, cheaper units aimed at budget travelers. The Ridge coefficients tell the same story: the largest positive Ridge coefficients are on `nbh_te` (target-encoded neighborhood, +0.19), `bathrooms` (+0.21), `accommodates` (+0.22), `room_type_Entire home/apt` (+0.29), and `property_type_bucket_Room in hotel` (+0.36). The largest negative Ridge coefficients are on `room_type_Shared room` (-0.50), `property_type_bucket_Private room in home` (-0.24), and `reviews_per_month` (-0.08). Two independent models using very different functional forms agree on what moves price, which is the kind of cross-validation that makes me more confident in the interpretation.

### Residual structure

Figure 4 shows predicted-versus-actual log-prices for XGBoost on the test set. The cloud tracks the diagonal tightly in the dense middle of the distribution (log-prices between 4 and 6, roughly $55 to $400 per night), with some heteroskedasticity in the thin upper tail (log-prices above 6, or roughly $400 per night), where the model tends to under-predict.

![Figure 4. XGBoost predicted vs actual log-price on the test set. The dashed red line marks perfect prediction. The cloud is tight along the diagonal through most of the distribution, with under-prediction in the upper tail.](figures/fig1_xgb_pred_vs_actual.png)

The residual density plot in Figure 5 shows that XGBoost and Random Forest have the narrowest error distributions, closely centered on zero, while Ridge and the MLP have heavier tails. None of the models show systematic bias, which means the winsorization and log transform are doing their job.

![Figure 5. Test-set residual (actual minus predicted log-price) densities for all four models. XGBoost and Random Forest are tightest around zero; Ridge and MLP have heavier tails.](figures/fig2_residuals.png)

---

## 5. Discussion

### What the results mean

The central finding is straightforward. On the 2025 Austin Inside Airbnb snapshot, XGBoost produces the most accurate test-set predictions of the four models I tried, with log-price RMSE of 0.38 and a mean absolute dollar error of $68 on listings with a median price of $162. The ranking (XGBoost, Random Forest, Ridge, MLP) is consistent with what the broader tabular-regression literature predicts. The SHAP attributions on the winning model land on exactly the features that hedonic pricing theory (Rosen, 1974; Sheppard, 1999) tells us should drive short-term rental prices: capacity (bathrooms, bedrooms, accommodates), location (target-encoded neighborhood, distance to downtown), and product differentiation (entire home versus shared room, hotel versus private home). This is not a novel substantive claim, but it is a reassuring sanity check: a modern tree-based model trained on raw listing attributes rediscovers the same economic structure that fifty years of housing economics has already documented.

The specific result I find most informative for the paper's second question is the neural-network comparison. A well-configured feed-forward MLP with dropout and early stopping did worse than Ridge regression on my data. For a student reading this paper, the useful takeaway is that "deeper is better" is not a reliable prior on tabular regression problems. Shwartz-Ziv and Armon (2022) argue this explicitly, and Gorishniy et al. (2021) report similar patterns across a large benchmark suite. My result is one more data point for that argument on a specific, publicly reproducible dataset, and it is broadly consistent with the applied ML literature on Airbnb and rental pricing where feature engineering and ensemble methods have tended to dominate over end-to-end neural approaches (Kalehbasti et al., 2019; Xu & Zhang, 2022).

### Interpretability versus accuracy

The gap between Ridge and XGBoost on my data is real: about four cents of log-RMSE, or about twenty dollars of MAE. A hedonic-pricing paper from 1999 would have stopped at Ridge because the coefficients were directly interpretable. A 2024 applied paper can use XGBoost and then recover interpretability through SHAP. In my results the Ridge coefficients and the XGBoost SHAP values tell the same story about what drives price, which means the interpretability cost of using the more accurate model is small on this dataset. That is not always the case; on datasets with strong feature interactions that a linear model cannot express, the two methods will disagree on feature importance, and the SHAP picture is the more honest one. For future work on Austin data I would reach for XGBoost first and Ridge as a sanity check, not the other way around.

### Policy implications

Austin has been revising its short-term rental ordinance since the first version passed in 2012 (Nieuwland & van Melik, 2020; City of Austin, 2023). A predictive model like the one in this paper is not a policy tool in itself, but two of its outputs are relevant to the ongoing policy debate. First, the target-encoded neighborhood feature is one of the top five SHAP contributors, which quantifies the location premium that a host captures; this is consistent with the broader urban-economics finding that local amenities drive a large share of residential price variation and migration decisions (Chen & Rosenthal, 2008), and with the spatial-econometric framing used by Anselin (2013). Second, the `room_type_Entire home/apt` one-hot is among the largest positive coefficients in Ridge and shows up consistently on the positive side in SHAP. Entire-home listings are precisely the category that Austin's ordinance most tightly regulates (through its distinction between owner-occupied Type 1 listings and non-owner-occupied Type 2 listings), and the fact that they command a substantial price premium over private-room listings is consistent with the hypothesis that STR operators have a strong profit incentive to convert long-term rentals into non-owner-occupied short-term listings. Lee (2016) documents the same premium structure in Los Angeles and argues it accelerates affordable-housing loss in constrained markets. I do not claim a causal estimate of that effect from this paper; I note that the model's output is consistent with the framing used by the city's policy work.

### Limitations

This paper has four limitations I want to state plainly. First, I use a single monthly snapshot and therefore cannot model seasonality, trends, or the effect of Austin's periodic event calendar (SXSW, ACL, Formula 1) on prices. A panel of snapshots would let me separate cross-sectional from time-varying effects. Second, Inside Airbnb is a scrape, not the platform's internal data, so calendar-based dynamic pricing by the host is not observable. The price I see is the default public nightly rate, not the actual booking price. Third, my features are entirely tabular. Listing photos carry strong signal that a CNN-based model could extract, and listing descriptions have signal that a pretrained text encoder could capture. I deliberately scoped the project to tabular regression, but a multi-modal extension is a natural next step. Fourth, my train/validation/test split is random, not geographic. A more stringent evaluation would hold out entire neighborhoods to test generalization to parts of Austin the model has never seen.

### Future work

Three extensions I would pursue with more time. The first is the multi-snapshot panel, which would support a difference-in-differences study of the city's ordinance changes. The second is an image-augmented model using a pretrained vision encoder on the listing photos, which would test whether visual quality adds predictive signal on top of the tabular attributes. The third is a cross-city generalization study: train on Austin and evaluate on Houston, Dallas, and San Antonio to see which features transfer and which are Austin-specific. For a master's thesis scope, all three are tractable within a semester.

---

## 6. Conclusion

I built a reproducible, Colab-executable pipeline that predicts nightly prices for Austin Airbnb listings from publicly available listing attributes, and I used it to compare four model families on the same data splits. XGBoost achieved the best test-set performance, with a log-price RMSE of 0.38 and a dollar mean absolute error of $68 on listings whose median price is $162. Random Forest was a close second. Ridge regression, despite its simplicity, beat a three-layer MLP on every metric, which is a concrete data point supporting the broader literature finding that gradient-boosted trees are still the right default for medium-sized tabular regression.

SHAP attributions on the XGBoost model identified capacity (bathrooms, bedrooms, accommodates), location (target-encoded neighborhood, distance to downtown), and product type (entire home versus shared room) as the dominant drivers of predicted price. These are exactly the features hedonic pricing theory predicts should matter, and Ridge regression produced coefficients that agreed with the SHAP picture. The agreement between two very different functional forms on what moves price is the strongest evidence this paper offers for the validity of the interpretation.

For hosts and for city policy, the practical implication is that a modest off-the-shelf ML pipeline can give an accurate, interpretable estimate of a listing's market-clearing nightly rate from publicly observable attributes. The pipeline does not need proprietary data, does not need a GPU, and does not need weeks of engineering. That lowers the bar for applying this kind of analysis to other cities and for evaluating the ongoing short-term rental policy debates with quantitative evidence.

The code and paper are at https://github.com/nitin2468git/ut-airbnb-final-project and reproduce every figure and table in this paper from the same data snapshot in under five minutes on a Google Colab free-tier CPU runtime.

---

## References

Anselin, L. (2013). *Spatial econometrics: Methods and models*. Springer.

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

This project used Anthropic's Claude (Opus 4.7) in a Claude Code session for three purposes. First, repository scaffolding: the README, requirements file, gitignore, and paper outline were generated by Claude. Second, notebook templating: Claude produced the initial section and step structure of `airbnb_price_prediction.ipynb`, along with the initial implementations of each cell. I reviewed every cell, fixed bugs in the Inside Airbnb download URL and the shap pin, and confirmed each step's output against my own reading of the data. Third, drafting assistance for this paper: Claude produced strawman text for each section, which I then rewrote in my own voice, corrected where the interpretation was wrong or overclaimed, and edited for tone. All empirical results, figures, numerical interpretations, and the final wording of this paper are my own work. The session transcript is available on request.

## Appendix B. Code availability

All code is available at https://github.com/nitin2468git/ut-airbnb-final-project. The single notebook `notebooks/airbnb_price_prediction.ipynb` reproduces every figure and table in this paper when run on Google Colab's free CPU runtime. See the repository README for the Open-in-Colab badge and step-by-step usage instructions.
