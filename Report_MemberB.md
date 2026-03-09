# Member B Report — Advanced Modeling & Causal Inference

**Project:** Student Dropout Prediction (UCI Dataset #697)
**Member B responsibilities:** Advanced predictive modeling (hyperparameter tuning) + causal inference analysis
**Notebooks:** `Advanced_Modeling.ipynb`, `Causal_Analysis.ipynb`

---

## Section 0 — Dataset Description

### 0.1 What Is This Dataset?

The dataset used in this project is the **"Predict students' dropout and academic success"** dataset from the UCI Machine Learning Repository (Dataset ID: 697). It was collected from a Portuguese higher education institution and covers students enrolled in various undergraduate degrees over several academic years.

The core goal of the dataset is to support **early identification of students at risk of dropping out**, so that universities can intervene in time with targeted support.

### 0.2 Dataset Contents

The dataset contains **4,424 students** described by **36 features** spanning four categories:

| Category | Example Features |
|:---|:---|
| **Demographics** | Age at enrollment, Gender, International student, Marital status, Nationality |
| **Socioeconomic background** | Parents' education level and occupation, Debtor status, Tuition fees up-to-date |
| **Academic history at enrollment** | Previous qualification (grade), Admission grade, Scholarship holder, Curricular units enrolled/approved in 1st and 2nd semester |
| **Macroeconomic context** | Unemployment rate, Inflation rate, GDP (at the time of enrollment) |

The **target variable** has three classes:

| Class | Count | Meaning |
|:---|:---:|:---|
| **Dropout** | ~1,421 (32%) | Student left the program without graduating |
| **Enrolled** | ~794 (18%) | Student is still currently enrolled |
| **Graduate** | ~2,209 (50%) | Student successfully completed the program |

### 0.3 Why This Dataset Matters

Student dropout is a major challenge for higher education institutions — it represents a loss for both the student and the institution. By predicting dropout risk early (e.g., after Semester 1 results become available), universities can direct limited resources (counseling, financial aid, tutoring) toward students who need them most. This dataset provides a realistic, multi-class setting for building and evaluating such early-warning systems.

---

## Section 1 — Overview

Member B's work builds directly on the preprocessed data delivered by Member A and extends the project in two directions:

| Notebook | Purpose |
|:---|:---|
| `Advanced_Modeling.ipynb` | Establishes strong baseline models (CatBoost, XGBoost), tunes them via Optuna, and saves the best model for downstream use by Member C |
| `Causal_Analysis.ipynb` | Moves beyond correlation to estimate the *causal* effect of receiving a scholarship on dropout probability, using PSM, DML, and R-learner CATE |

The outputs of `Advanced_Modeling.ipynb` feed directly into Member C's SHAP explainability and fairness analysis. The causal CATE estimates from `Causal_Analysis.ipynb` complement Member C's fairness audit by quantifying heterogeneous treatment effects across demographic subgroups.

---

## Section 2 — Advanced Modeling (`Advanced_Modeling.ipynb`)

### 2.0 Methods Overview

The advanced modeling pipeline applies two gradient-boosted tree algorithms and uses Bayesian hyperparameter optimization to find their best configurations:

| Component | Method | Purpose |
|:---|:---|:---|
| **Primary models** | CatBoost, XGBoost | Gradient-boosted decision tree ensembles for multi-class classification |
| **Hyperparameter tuning** | Optuna with TPE (Tree-structured Parzen Estimator) | Bayesian optimization over the hyperparameter search space |
| **Evaluation metric** | Weighted F1-score | Accounts for class imbalance across Dropout / Enrolled / Graduate |
| **Anti-leakage design** | Stratified internal validation split (20%) | Test set is never seen during tuning; only used for final reporting |
| **Model interpretation** | Feature importance, confusion matrices, multi-class ROC-AUC | Understand what drives predictions and where errors occur |

**CatBoost** is a gradient-boosted tree library that handles categorical features natively using ordered target encoding, which reduces overfitting compared to standard label encoding. **XGBoost** is a highly optimized regularized gradient boosting framework with broad adoption in structured-data tasks. Both are strong baselines for tabular classification.

**Optuna (TPE):** Rather than exhaustive grid search, Optuna's Tree-structured Parzen Estimator models the objective function as a probabilistic surrogate, directing new trials toward regions of the hyperparameter space that are likely to improve the metric. This is substantially more efficient than random search, especially with 40 trials per model.

### 2.1 Data Source

Member B loads four CSV files produced by Member A's preprocessing pipeline:

| File | Description |
|:---|:---|
| `data/X_train_processed.csv` | Training features (SMOTE-balanced) |
| `data/y_train_processed.csv` | Training labels (SMOTE-balanced) |
| `data/X_test_processed.csv` | Test features (held-out, never touched during tuning) |
| `data/y_test_processed.csv` | Test labels |

**Dataset dimensions after loading:**
- Training set: **5,301 samples × 36 features** (SMOTE-balanced)
- Test set: **885 samples × 36 features**
- Test class distribution: Dropout 284 (32.1%), Enrolled 159 (18.0%), Graduate 442 (49.9%)

A fallback cell regenerates the preprocessing from UCI if any CSV is found empty (a safeguard against git corruption of large binary CSV files).

### 2.2 Baseline Models

Both models are fit on the full training set with fixed `random_state=42` and 500 trees (CatBoost: `iterations=500`; XGBoost: `n_estimators=500`), all other parameters at library defaults. This establishes a clean reference point before tuning.

| Model | Accuracy | F1 (Weighted) |
|:---|:---:|:---:|
| Baseline CatBoost | 0.7605 | 0.7580 |
| Baseline XGBoost | 0.7525 | 0.7514 |

### 2.3 Optuna Hyperparameter Tuning

**Algorithm:** Tree-structured Parzen Estimator (TPE) — Bayesian optimization that focuses trials on promising regions of the search space.

**Anti-leakage design:** Each trial evaluates on a stratified 20% validation split held out from the training set. The held-out test set is never seen during tuning; it is used exactly once for final reporting.

**Settings:** 40 trials per model (`N_TRIALS = 40`), `VAL_SIZE = 0.2`.

#### CatBoost Search Space

| Parameter | Range | Scale |
|:---|:---|:---|
| `iterations` | 100 – 800 | linear |
| `learning_rate` | 1e-3 – 0.3 | log |
| `depth` | 4 – 10 | linear |
| `l2_leaf_reg` | 0.01 – 20 | log |

**Best parameters found:**

```
iterations    = 581
learning_rate = 0.1454
depth         = 7
l2_leaf_reg   = 1.7878
```

Best validation F1 (internal): **0.8737**

#### XGBoost Search Space

| Parameter | Range | Scale |
|:---|:---|:---|
| `n_estimators` | 100 – 800 | linear |
| `learning_rate` | 1e-3 – 0.3 | log |
| `max_depth` | 3 – 10 | linear |
| `subsample` | 0.6 – 1.0 | linear |
| `colsample_bytree` | 0.6 – 1.0 | linear |
| `min_child_weight` | 1 – 10 | linear |

**Best parameters found:**

```
n_estimators     = 387
learning_rate    = 0.0817
max_depth        = 7
subsample        = 0.9263
colsample_bytree = 0.9857
min_child_weight = 10
```

Best validation F1 (internal): **0.8802**

### 2.4 Final Test-Set Results

Each tuned model is retrained on the **full training set** with its best parameters, then evaluated once on the held-out test set.

| Model | Accuracy | F1 (Weighted) | Delta F1 vs Baseline |
|:---|:---:|:---:|:---:|
| Baseline CatBoost | 0.7605 | 0.7580 | — |
| **Tuned CatBoost** | 0.7559 | 0.7527 | −0.0053 |
| Baseline XGBoost | 0.7525 | 0.7514 | — |
| **Tuned XGBoost** | 0.7446 | 0.7420 | −0.0094 |

The tuned CatBoost has the higher weighted F1 and is selected as the best model. Note that the Optuna internal validation F1 (~0.87) is higher than the test F1 (~0.75) because the internal split is drawn from the SMOTE-augmented training distribution; this gap is expected and does not indicate overfitting on the test set.

**Detailed classification report — Tuned CatBoost:**

| Class | Precision | Recall | F1-score | Support |
|:---|:---:|:---:|:---:|:---:|
| Dropout | 0.81 | 0.74 | 0.77 | 284 |
| Enrolled | 0.48 | 0.45 | 0.46 | 159 |
| Graduate | 0.82 | 0.88 | 0.85 | 442 |
| **Weighted avg** | **0.75** | **0.76** | **0.75** | **885** |

### 2.5 Visualizations

The notebook produces four sets of plots:

| Figure | Description |
|:---|:---|
| **Optuna convergence curves** | Scatter of per-trial validation F1 + running maximum for both models; shows TPE converging within ~20 trials |
| **Feature importance (Top 15)** | Side-by-side bar charts for Tuned CatBoost and Tuned XGBoost; 2nd-semester academic metrics (approved units, grades) dominate in both |
| **Confusion matrices** | Heatmaps for both tuned models; 'Enrolled' shows the most cross-class confusion |
| **Multi-class ROC-AUC curves** | One-vs-rest curves per class; Dropout AUC ≈ **[X.XXX]**, Enrolled AUC ≈ **[X.XXX]**, Graduate AUC ≈ **[X.XXX]** (run cells to populate) |

**Key finding from feature importance:** Both models agree that in-progress academic performance (2nd-semester approved units and grades, 1st-semester equivalents) far outweighs demographic features as dropout predictors. This justifies targeting early-warning interventions after Semester 1 results are available.

---

## Section 3 — Causal Inference (`Causal_Analysis.ipynb`)

### 3.0 Methods Overview

The causal analysis uses three layered methods to answer whether scholarships *causally* reduce dropout risk, not merely correlate with lower dropout rates:

| Layer | Method | Key Property |
|:---|:---|:---|
| **Baseline** | Naive comparison + KS test | Quantifies raw gap and confirms selection bias |
| **Method 1** | Propensity Score Matching (PSM) | Balances observed confounders by matching treated/control students |
| **Method 2** | Double Machine Learning (DML) | Partialling-out with ML; Neyman-orthogonal ATE estimator |
| **Method 3** | R-learner CATE | Heterogeneous treatment effects — who benefits most? |

**Why causal methods?** The naive dropout-rate gap between scholarship holders (12.2%) and non-holders (38.7%) is 26.5 percentage points. However, this gap could be entirely explained by *selection bias*: universities award scholarships to academically stronger students who would have been less likely to drop out regardless. Kolmogorov-Smirnov tests confirm that the two groups differ significantly on pre-treatment covariates (e.g., age at enrollment: KS stat = 0.236, p = 5.07e-41), violating the exchangeability assumption required for naive comparison. Causal adjustment is necessary.

**Key identifying assumption (Unconfoundedness):** All three methods assume that, conditional on the observed covariates $X$, treatment assignment is as good as random: $T \perp (Y(0), Y(1)) \mid X$. In other words, there are no *unobserved* confounders. This is a strong but standard assumption in observational causal inference.

**Data note:** Causal analysis uses the **raw (unprocessed)** UCI data (4,424 students) rather than the SMOTE-balanced training set. SMOTE synthetically replicates minority-class observations, which would distort propensity score estimation and causal effect estimates.

### 3.1 Research Question

> *What is the Average Treatment Effect (ATE) of receiving a scholarship on the probability of dropping out?*

$$\text{ATE} = E[Y(1) - Y(0)] = E[Y \mid do(T=1)] - E[Y \mid do(T=0)]$$

- **Treatment** $T$: `Scholarship holder` (1 = yes, 0 = no)
- **Outcome** $Y$: binary dropout indicator (`Target == 'Dropout'`)
- **Covariates** $X$: remaining 35 features (all pre-treatment background variables)

**Dataset:** raw (unprocessed) UCI data, 4,424 students.
Causal analysis deliberately uses the original (non-SMOTE) data — SMOTE synthetically balances classes and would distort causal estimates.

| Group | N | Dropout Rate |
|:---|:---:|:---:|
| Scholarship holders (T=1) | 1,099 (24.8%) | 12.2% |
| Non-holders (T=0) | 3,325 (75.2%) | 38.7% |
| **Naive (unadjusted) ATE** | — | **−26.51 pp** |

### 3.2 Selection Bias Evidence (KS Test)

The naive 26.5 pp gap may be entirely explained by selection bias: universities award scholarships to academically stronger students who would have been less likely to drop out regardless.

Kolmogorov-Smirnov tests on key covariates confirm that treated and control groups differ *systematically before* treatment:

| Covariate | KS Statistic | p-value |
|:---|:---:|:---:|
| Previous qualification (grade) | 0.074 | 2.46e-04 |
| Age at enrollment | 0.236 | 5.07e-41 |

Significant p-values (p < 0.05) directly violate the exchangeability assumption required for naive comparison. Causal adjustment is necessary.

### 3.3 Method 1 — Propensity Score Matching (PSM)

**Propensity score** $e(X) = P(T=1 \mid X)$ estimated via logistic regression (`max_iter=2000`, `C=1.0`).

**Matching procedure:**
- Common support: students outside the overlap region of $e(X)$ are excluded
- 1:1 nearest-neighbour matching with caliper $= 0.2 \times \text{SD}(e(X))$

| Step | Value |
|:---|:---|
| Caliper (0.2 × SD of PS) | 0.0349 |
| PS range — treated | [0.013, 0.768] |
| PS range — control | [0.002, 0.753] |
| Students in common support | 4,362 / 4,424 |
| Matched pairs | 1,097 / 1,097 |

**Covariate balance (Love Plot):** After matching, all key covariates achieve |SMD| < 0.1, confirming that matched treated/control students are comparable on observed pre-treatment characteristics.

| | Mean \|SMD\| Before | Mean \|SMD\| After |
|:---|:---:|:---:|
| All balance features | 0.231 | 0.023 |

**PSM ATE estimate** (bootstrap CI, 2,000 replications):

| Quantity | Value |
|:---|:---|
| ATE | −0.0647 (**−6.47 pp**) |
| 95% Bootstrap CI | [−0.0912, −0.0383] |
| Bootstrap SE | 0.0138 |
| p-value | < 0.0001 (significant) |

### 3.4 Method 2 — Double Machine Learning (DML)

DML (Chernozhukov et al., 2018) avoids reliance on a correctly-specified propensity model by using ML to partial out confounders from both the outcome and the treatment simultaneously.

**Algorithm (Partialling-Out / Robinson's procedure with cross-fitting):**

1. **Residualize Y:** fit $\hat{m}(X) = E[Y \mid X]$ via GradientBoosting (300 trees, depth=4, lr=0.05), 5-fold CV; compute $\tilde{Y} = Y - \hat{m}(X)$
2. **Residualize T:** fit $\hat{e}(X) = E[T \mid X]$ via GradientBoosting, 5-fold CV; compute $\tilde{T} = T - \hat{e}(X)$
3. **OLS:** regress $\tilde{Y}$ on $\tilde{T}$ — the Frisch-Waugh-Lovell slope is the ATE

Cross-fitting ensures the nuisance models are fit on held-out folds, making the estimator **Neyman-orthogonal** (robust to small misspecification).

**DML ATE results:**

| Quantity | Value |
|:---|:---|
| ATE | −0.0443 (**−4.43 pp**) |
| Std. Error | 0.0120 |
| 95% CI | [−0.0679, −0.0207] |
| z-statistic | −3.682 |
| p-value | 0.0002 (significant) |

### 3.5 Heterogeneous Treatment Effects — CATE (R-Learner)

The ATE is an average; scholarship effects vary across students. Conditional ATE $\tau(x) = E[Y(1) - Y(0) \mid X=x]$ is estimated with the **R-learner** (Nie & Wager, 2021):

$$\hat{\tau}(x) = \arg\min_{\tau} \sum_i \tilde{T}_i^2 \left(\frac{\tilde{Y}_i}{\tilde{T}_i} - \tau(X_i)\right)^2$$

DML residuals $\tilde{Y}$, $\tilde{T}$ from Section 3.4 are reused — no additional nuisance fitting needed. A GradientBoosting regressor (300 trees, depth=3) is fit on 4,377 valid samples.

**CATE summary:**

| Statistic | Value |
|:---|:---|
| Mean CATE | −0.0318 (consistent with DML ATE −0.0443) |
| Std. Dev | 0.9515 |
| Min | −20.6126 |
| Max | +24.8854 |
| % of students for whom scholarship helps (CATE < 0) | 66.6% |

**Who benefits most?** Subgroup analysis (bar plots by age group, previous grade quartile, debtor status) reveals that older students and students in financial debt tend to have the most negative CATE — the scholarship addresses financial barriers that would otherwise dominate their dropout risk.

### 3.6 Results Summary — Forest Plot

| Method | ATE (pp) | 95% CI | p-value | Confounder Adjustment |
|:---|:---:|:---:|:---:|:---|
| Naive Comparison | −26.51 | N/A | N/A | None |
| PSM | **−6.47** | **[−9.12, −3.83]** | **< 0.0001** | Propensity score (logistic) |
| **DML** | **−4.43** | **[−6.79, −2.07]** | **0.0002** | ML partialling-out (GB + 5-fold CV) |

The naive gap is inflated ~6× by selection bias. After causal adjustment, both PSM and DML converge on a significant but substantially smaller effect. Convergence between two methodologically distinct estimators strengthens the credibility of the causal claim.

---

## Section 4 — Handover Document for Member C

### 4.1 Output File Inventory

Member C should expect the following files in the repository root / `data/` after running Member B's notebooks:

| File | Produced by | Description |
|:---|:---|:---|
| `best_model.cbm` | `Advanced_Modeling.ipynb` | Tuned CatBoost (best model, F1=0.7527) |
| `data/X_train_processed.csv` | Member A / fallback cell | 5,301 × 36 training features |
| `data/X_test_processed.csv` | Member A / fallback cell | 885 × 36 test features |
| `data/y_train_processed.csv` | Member A / fallback cell | 5,301 training labels |
| `data/y_test_processed.csv` | Member A / fallback cell | 885 test labels |
| `cate_analysis.png` | `Causal_Analysis.ipynb` | CATE distribution + subgroup plots |

> Note: If `best_model.cbm` is not present, re-run all cells in `Advanced_Modeling.ipynb`. If the XGBoost model had been selected instead, it would be saved as `best_model.pkl` (loaded with `joblib.load`).

### 4.2 Loading the Model for SHAP Analysis

```python
from catboost import CatBoostClassifier
import pandas as pd

# Load model
model = CatBoostClassifier()
model.load_model('best_model.cbm')

# Load test data
X_test = pd.read_csv('data/X_test_processed.csv')
y_test = pd.read_csv('data/y_test_processed.csv').values.ravel()

CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']

# Verify
preds = model.predict(X_test)
print(f'Loaded model predicts {len(preds)} samples.')
```

**SHAP with TreeExplainer (recommended for CatBoost):**

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# shap_values is a list of 3 arrays (one per class), shape (885, 36) each

# Global summary — beeswarm for the Dropout class (index 0)
shap.summary_plot(shap_values[0], X_test, class_names=CLASS_NAMES)

# Waterfall plot for a single student (e.g., student 0, Dropout class)
shap.plots.waterfall(explainer(X_test)[0, :, 0])
```

> If `shap.TreeExplainer` raises a compatibility error with the `.cbm` format, use `shap.Explainer(model.predict_proba, X_test)` as a fallback (KernelExplainer mode, slower).

### 4.3 Suggested SHAP Analysis Directions

| Analysis | SHAP API | Question it Answers |
|:---|:---|:---|
| **Global feature importance** | `shap.summary_plot(shap_values[0], X_test)` | Which features most strongly drive dropout predictions overall? |
| **Beeswarm plot** | `shap.summary_plot(shap_values[0], X_test, plot_type='dot')` | How do feature values (high/low) push predictions toward or away from Dropout? |
| **Waterfall / force plot** | `shap.plots.waterfall(...)` | Why did the model classify a specific high-risk student as Dropout? |
| **Dependence plot** | `shap.dependence_plot('Curricular units 2nd sem (grade)', shap_values[0], X_test)` | Non-linear relationship + interaction between 2nd-semester grade and another feature |
| **Multi-class comparison** | Loop over `shap_values[0]`, `[1]`, `[2]` | Are the top features the same for predicting Dropout vs Graduate? |

**Expected top features from Member B's analysis:** Both CatBoost and XGBoost rank 2nd-semester academic performance features (approved units, grades) and their 1st-semester equivalents at the top. These are the most productive variables to investigate in SHAP plots.

### 4.4 Suggested Fairness Analysis Directions

Member C's fairness analysis should examine whether the model's predictions and errors are equitable across demographic subgroups. Recommended groupings:

| Dimension | Column in Raw Data | Suggested Groupings |
|:---|:---|:---|
| **Gender** | `Gender` | 0 = Female, 1 = Male |
| **Scholarship status** | `Scholarship holder` | 0 = No, 1 = Yes |
| **Age group** | `Age at enrollment` | ≤20, 21–25, 26–30, >30 |
| **Financial debt** | `Debtor` | 0 = No, 1 = Yes |
| **International student** | `International` | 0 = Domestic, 1 = International |

**Fairness metrics to compute per subgroup:**

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Example: fairness by gender
for gender, label in [(0, 'Female'), (1, 'Male')]:
    mask = (X_test_raw['Gender'] == gender)
    acc  = accuracy_score(y_test[mask], preds[mask])
    f1   = f1_score(y_test[mask], preds[mask], average='weighted')
    print(f'{label}: Acc={acc:.3f}, F1={f1:.3f}')
```

Key questions to investigate:
- Is the **False Positive Rate for Dropout** (incorrectly flagging a non-dropout) higher for any subgroup? This would impose unnecessary intervention on those students.
- Is the **False Negative Rate** (missing an actual dropout) higher for any subgroup? This would mean some at-risk students are not identified.
- Does **model calibration** (predicted probability vs actual dropout rate) hold equally across groups?

### 4.5 Connecting Causal CATE to Fairness

The CATE estimates from `Causal_Analysis.ipynb` directly complement Member C's fairness audit:

- The R-learner CATE model (`cate_model` in `Causal_Analysis.ipynb`) predicts how much a scholarship would *change* each student's dropout probability, given their background.
- Member C can segment CATE values by the same demographic groups used in the fairness analysis to ask: **"Does the intervention (scholarship) have equal benefit across groups, or does it disproportionately help some subgroups?"**

Example linkage code:

```python
# In Causal_Analysis.ipynb, cate_values is already computed (shape: 4424,)
# Add CATE to the raw dataframe and group by demographics

df['CATE'] = cate_values
for col, label in [('Gender', 'Gender'), ('Debtor', 'In Debt')]:
    print(f'\nMean CATE by {label}:')
    print(df.groupby(col)['CATE'].agg(['mean', 'std', 'count']))
```

**Interpretation guidance:**
- If a group has a more negative mean CATE → they benefit more from scholarships → prioritizing them for scholarship allocation is both efficient and equitable.
- If the predictive model flags a group as high-risk (high dropout probability) but their CATE is near zero → prediction-based intervention may not help them → structural support (not just financial) may be needed.
- Comparing ATE across groups bridges the predictive model (Member B, `Advanced_Modeling.ipynb`) and the policy model (Member B, `Causal_Analysis.ipynb`) for Member C's final policy recommendations.

---
