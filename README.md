# Generalized Influential Outlier Metric

Research code for the **Influential Outlier Metric (IOM)** with flexible Gaussian-mixture base distributions. Neural spline normalizing flows transform SHAP values and model residuals into latent representations, supporting unimodal and multimodal data and optional conditioning variables.

The metric combines two components for each observation:

```text
IOM[i] = Q_shap[i] * Q_residual[i]
```

Each `Q` is the squared norm of the latent observation after standardization within its most likely Gaussian-mixture component. Large scores identify observations for further investigation based on their joint SHAP and residual behavior.

## Repository contents

| File | Purpose |
| --- | --- |
| [`iom.py`](iom.py) | Flow fitting, regularization selection, comparison of mixture sizes, IOM scores, and reference thresholds. |
| [`mode_assign.py`](mode_assign.py) | Diagonal Gaussian-mixture responsibilities, mode assignment, whitening, and goodness-of-fit diagnostics. |
| [`nf_multimodal.ipynb`](nf_multimodal.ipynb) | Flow experiments with unimodal, bimodal, and trimodal synthetic distributions. |
| [`IOM_ME.ipynb`](IOM_ME.ipynb) | Simulated mixed-effects analyses, including linear and random-forest workflows at student and school levels. |
| [`IOM_PISA.ipynb`](IOM_PISA.ipynb) | PISA student- and school-level analyses with mixed-effects boosting. Requires external data. |

## Setup

Run commands from the repository root. The project currently has no package installer or pinned dependency file.

```bash
python -m pip install numpy pandas scipy scikit-learn torch normflows tqdm pingouin matplotlib seaborn statsmodels shap optuna joblib xgboost
```

The PISA notebook additionally uses `pyspark` and `pyreadstat`:

```bash
python -m pip install pyspark pyreadstat
```

Flow fitting automatically uses CUDA when available and otherwise runs on CPU. Training can be time-consuming: regularization selection fits a separate flow for each candidate penalty.

## Computing IOM scores

First fit your predictive model and compute its SHAP values and residuals. The core module accepts these quantities; it does not fit the predictive model or calculate SHAP values for you.

The following example assumes `shap_values` contains an `(n, p)` numeric array and `residuals` contains an `(n,)` numeric array for the **same observations in the same order**. A single SHAP component may also be passed as an `(n,)` array.

```python
import numpy as np
from iom import InfluentialOutlierMetric

metric = InfluentialOutlierMetric()

flow_options = dict(
    lambdas=[0.0, 0.01, 0.1],
    K=6,             # Number of spline transformations
    hidden=6,        # Conditioner width
    layers=2,        # Conditioner hidden blocks
    epoch=500,
    loc=[0.0],       # Initial mixture component means
    scale=[1.0],     # Initial component standard deviations
    tail_bound=7,
    num_bins=8,
    torch_seed=42,
)

results_shap = metric.find_best_lambda(
    np.asarray(shap_values), **flow_options
)
results_resid = metric.find_best_lambda(
    np.asarray(residuals), **flow_options
)

scores = metric.IOM(results_shap, results_resid)
print(scores.sort_values(ascending=False).head(10))
```

These settings are starting values, not a guarantee that the fit will pass the diagnostics. `scores` is a pandas Series with a new positional index, also stored as `metric.IOM_`; preserve your observation identifiers separately.

### Inputs and preprocessing

- Supply finite numeric arrays. Encode categorical conditioning variables before passing them as `context`.
- Pass `context=context_array` to either fitting call to condition the spline transformations. It must have the same number of rows as the corresponding data.
- The flow centers its input using the training-set mean but does not scale its variance. Context is both centered and scaled. Any additional input standardization is the caller's responsibility.
- Supply `K`, `hidden`, `layers`, `epoch`, `loc`, `scale`, `tail_bound`, and `num_bins` explicitly when using `find_best_lambda`.
- `loc` and `scale` contain one scalar per mixture component; initialization repeats each scalar across input dimensions. Scales are positive standard deviations. The mixture parameters are trainable, and the base distribution does not depend on context.

### Regularization and mixture selection

`find_best_lambda` tries penalties in the supplied order and keeps the last passing fit before the first failure. Use ascending `lambdas` to search for increasing regularization. If the first fit fails, it raises `RuntimeError("Increase complexity of normalizing flow.")`; revisit training duration, flow capacity, preprocessing, or the mixture initialization.

Acceptance currently requires a normality test for every mixture component, all component p-values above 0.01, a combined normality p-value above 0.05, and a pooled chi-square Cramér–von Mises p-value above 0.05. Components with fewer than 20 assigned observations are skipped and therefore cannot satisfy acceptance. The `alpha` argument to `find_best_lambda` does not currently change these hard-coded criteria.

Each successful result includes `lambda`, `model`, `x_scaler`, `context_scaler`, `latent`, `weights`, `means`, `scales`, `test_distance`, and `chi_z`.

To explore multiple modes, fit another candidate with, for example, `loc=[-3.0, 3.0]` and `scale=[1.0, 1.0]`. Compare successful candidate dictionaries with:

```python
# candidates = [unimodal_result, bimodal_result]
selection = metric.select_flow_by_transport_modes(candidates, dim=p)
chosen = candidates[selection["selected_index"]]
```

Here, `p` is the dimension of the data used for those fits. The selection score is mean held-out squared transport distance plus `rho` times the number of free mixture parameters. The default `rho` is `log(len(candidates)) / len(candidates)`; pass it explicitly to use another penalty.

For a single fit without automatic goodness-of-fit selection, use `norm_flow` with an explicit `lam`. It returns the fitted model, latent values for all observations, and held-out squared transport distances. Direct calls to the returned model require preprocessing with its stored scalers.

### Reference thresholds

```python
threshold_05, threshold_01 = metric.find_threshold(p=p)
```

Here, `p` is the number of SHAP dimensions. This computes upper-tail thresholds for the product of independent chi-square variables with `p` and 1 degrees of freedom. Interpreting them as significance thresholds depends on those reference assumptions; hard mixture assignments and dependence between the two components can affect calibration.

## Running the notebooks

Start with `nf_multimodal.ipynb` for distribution experiments or `IOM_ME.ipynb` for a simulated analysis. Run cells in order and keep the repository root on Python's import path so that `iom` and `mode_assign` can be imported. The notebooks are research workflows rather than a packaged command-line application.

`IOM_PISA.ipynb` uses data from [PISA](https://www.oecd.org/en/data/datasets/pisa-2022-database.html#data).
