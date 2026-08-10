import numpy as np

from scipy.stats import chi2, kstest, cramervonmises, shapiro, norm, jarque_bera
from scipy.special import logsumexp
from pingouin import multivariate_normality


# ============================================================
# Input handling: diagonal covariance only
# ============================================================
def coerce_diag_gmm_inputs(X, weights, means, scales):
    """
    Coerce inputs for a diagonal Gaussian mixture.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, d)
        Latent observations after normalizing flow.

    weights : array-like, shape (K,)
        Mixture weights.

    means : array-like, shape (K,) or (K, d)
        Component means.

    scales : array-like, shape (K,) or (K, d)
        Component standard deviations, not variances.

    Returns
    -------
    X : array, shape (n, d)
    weights : array, shape (K,)
    means : array, shape (K, d)
    scales : array, shape (K, d)
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    scales = np.asarray(scales, dtype=float)

    if X.ndim == 1:
        X = X[:, None]

    if means.ndim == 1:
        means = means[:, None]

    if scales.ndim == 1:
        scales = scales[:, None]

    K, d = means.shape

    if weights.shape != (K,):
        raise ValueError(f"weights shape {weights.shape}, expected ({K},).")

    if scales.shape != (K, d):
        raise ValueError(f"scales shape {scales.shape}, expected ({K},{d}).")

    if X.shape[1] != d:
        raise ValueError(f"X has dimension {X.shape[1]}, but means has dimension {d}.")

    if np.any(scales <= 0):
        raise ValueError("All scales must be positive standard deviations.")

    weights = weights / weights.sum()

    return X, weights, means, scales


# ============================================================
# Diagonal Gaussian log density
# ============================================================
def diag_gaussian_logpdf(X, mean, scale):
    """
    Log density of N(mean, diag(scale^2)).

    X     : shape (n, d)
    mean  : shape (d,)
    scale : shape (d,)
    """
    z = (X - mean) / scale
    d = X.shape[1]

    return (
        -0.5 * np.sum(z**2, axis=1)
        - np.sum(np.log(scale))
        - 0.5 * d * np.log(2.0 * np.pi)
    )


# ============================================================
# Responsibilities and hard assignment
# ============================================================
def diag_gmm_responsibilities(X, weights, means, scales):
    """
    Compute posterior responsibilities:

        gamma_ik = P(component k | X_i)
    """
    X, weights, means, scales = coerce_diag_gmm_inputs(
        X, weights, means, scales
    )

    n = X.shape[0]
    K = len(weights)

    logp = np.empty((n, K), dtype=float)

    for k in range(K):
        logp[:, k] = (
            np.log(weights[k])
            + diag_gaussian_logpdf(
                X,
                mean=means[k],
                scale=scales[k],
            )
        )

    log_gamma = logp - logsumexp(logp, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    return gamma


def hard_assign_modes(X, weights, means, scales):
    """
    Assign each observation to the most likely component.
    """
    gamma = diag_gmm_responsibilities(X, weights, means, scales)
    labels = gamma.argmax(axis=1)
    max_resp = gamma.max(axis=1)

    return labels, gamma, max_resp


# ============================================================
# Whiten residuals after hard assignment
# ============================================================
def whiten_by_assigned_mode_diag(X, weights, means, scales):
    """
    Hard-assign observations to the most likely mode, then whiten
    each observation using that mode's diagonal mean and standard deviation.

        R_ij = (X_ij - mu_{k,j}) / sigma_{k,j}

    If assignment is correct and the GMM is correct, then within each mode:

        R_i ~ N(0, I_d)

    and pooled:

        Q_i = sum_j R_ij^2 ~ chi2_d
    """
    X, weights, means, scales = coerce_diag_gmm_inputs(
        X, weights, means, scales
    )

    labels, gamma, max_resp = hard_assign_modes(X, weights, means, scales)

    n, d = X.shape
    K = len(weights)

    R = np.empty_like(X, dtype=float)

    for k in range(K):
        idx = labels == k

        if not np.any(idx):
            continue

        R[idx] = (X[idx] - means[k]) / scales[k]

    Q = np.sum(R**2, axis=1)

    return {
        "R": R,
        "Q": Q,
        "labels": labels,
        "gamma": gamma,
        "max_resp": max_resp,
        "X": X,
        "weights": weights,
        "means": means,
        "scales": scales,
        "n": n,
        "d": d,
        "K": K,
    }


# ============================================================
# Normality test by mode:
#   - Jarque-Bera if d = 1
#   - Henze-Zirkler if d >= 2
# ============================================================
def normality_by_mode(R, labels, min_n=20, alpha=0.05):
    """
    Run normality tests within each assigned mode.

    If d == 1:
        Uses Jarque-Bera test on the whitened residuals R[:, 0].

    If d >= 2:
        Uses Henze-Zirkler multivariate normality test via pingouin.

    Returns
    -------
    rows : list of dict
    """
    R = np.asarray(R, dtype=float)
    labels = np.asarray(labels)

    n, d = R.shape
    rows = []

    for k in np.unique(labels):
        idx = labels == k
        Rk = R[idx]
        nk = Rk.shape[0]

        row = {
            "mode": int(k),
            "n": int(nk),
            "test": None,
            "stat": np.nan,
            "pvalue": np.nan,
            "normal": None,
            "note": "",
        }

        if nk < min_n:
            row["note"] = f"Skipped: n < {min_n}"
            rows.append(row)
            continue

        if d == 1:
            jb = jarque_bera(Rk[:, 0])

            row.update({
                "test": "Jarque-Bera",
                "stat": float(jb.statistic),
                "pvalue": float(jb.pvalue),
                "normal": bool(jb.pvalue > alpha),
                "note": "OK",
            })

        else:
            hz = multivariate_normality(Rk, alpha=alpha)

            row.update({
                "test": "Henze-Zirkler",
                "stat": float(hz.hz),
                "pvalue": float(hz.pval),
                "normal": bool(hz.normal),
                "note": "OK",
            })

        rows.append(row)

    return rows


# ============================================================
# Stouffer method for combining mode-level normality p-values
# ============================================================
def stouffer_method(pvals):
    """
    Stouffer's method for combining p-values.

    This is less dominated by one very small p-value than Fisher's method.
    Uses equal weights.

    Returns
    -------
    dict with:
        stat : combined z statistic
        pvalue : combined one-sided p-value
        n_tests : number of valid p-values
    """
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[~np.isnan(pvals)]

    if len(pvals) == 0:
        return {
            "stat": np.nan,
            "pvalue": np.nan,
            "n_tests": 0,
        }

    pvals = np.clip(pvals, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)

    z = norm.ppf(1.0 - pvals)
    z_combined = np.sum(z) / np.sqrt(len(z))
    p_combined = 1.0 - norm.cdf(z_combined)

    return {
        "stat": float(z_combined),
        "pvalue": float(p_combined),
        "n_tests": int(len(pvals)),
    }


# ============================================================
# Pooled chi-square tests
# ============================================================
def pooled_chi2_tests(Q, d):
    """
    Test pooled Q_i = ||R_i||^2 against chi2_d.
    """
    Q = np.asarray(Q, dtype=float)

    ks = kstest(Q, chi2(df=d).cdf)
    cvm = cramervonmises(Q, chi2(df=d).cdf)

    tail = chi2.sf(Q, df=d)

    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "cvm_stat": float(cvm.statistic),
        "cvm_pvalue": float(cvm.pvalue),
        "Q_mean": float(np.mean(Q)),
        "Q_median": float(np.median(Q)),
        "Q_max": float(np.max(Q)),
        "tail_min": float(np.min(tail)),
        "n_tail_lt_0.05": int(np.sum(tail < 0.05)),
        "n_tail_lt_0.01": int(np.sum(tail < 0.01)),
        "n_tail_lt_0.001": int(np.sum(tail < 0.001)),
    }


# ============================================================
# Mode-level summaries
# ============================================================
def mode_summaries_diag(R, Q, labels, max_resp, weights, d, min_n=20):
    """
    Summaries and chi-square diagnostics by assigned mode.
    """
    R = np.asarray(R, dtype=float)
    Q = np.asarray(Q, dtype=float)
    labels = np.asarray(labels)
    max_resp = np.asarray(max_resp, dtype=float)
    weights = np.asarray(weights, dtype=float)

    K = len(weights)
    n = len(labels)

    rows = []

    for k in range(K):
        idx = labels == k
        nk = int(np.sum(idx))

        if nk == 0:
            rows.append({
                "mode": k,
                "n": 0,
                "assigned_weight": 0.0,
                "target_weight": float(weights[k]),
                "mean_max_resp": np.nan,
                "median_max_resp": np.nan,
                "min_max_resp": np.nan,
                "Q_mean": np.nan,
                "Q_median": np.nan,
                "Q_max": np.nan,
                "ks_pvalue": np.nan,
                "cvm_pvalue": np.nan,
            })
            continue

        Qk = Q[idx]

        if nk >= min_n:
            ks = kstest(Qk, chi2(df=d).cdf)
            cvm = cramervonmises(Qk, chi2(df=d).cdf)
            ks_p = float(ks.pvalue)
            cvm_p = float(cvm.pvalue)
        else:
            ks_p = np.nan
            cvm_p = np.nan

        rows.append({
            "mode": k,
            "n": nk,
            "assigned_weight": float(nk / n),
            "target_weight": float(weights[k]),
            "mean_max_resp": float(max_resp[idx].mean()),
            "median_max_resp": float(np.median(max_resp[idx])),
            "min_max_resp": float(max_resp[idx].min()),
            "Q_mean": float(Qk.mean()),
            "Q_median": float(np.median(Qk)),
            "Q_max": float(Qk.max()),
            "ks_pvalue": ks_p,
            "cvm_pvalue": cvm_p,
        })

    return rows


# ============================================================
# Main workflow
# ============================================================
def whitened_mode_normality_workflow_diag(
    X,
    weights,
    means,
    scales,
    min_mode_n=20,
    alpha=0.05,
    verbose=True,
):
    """
    Workflow for diagonal Gaussian mixture diagnostics:

    1. Compute posterior responsibilities.
    2. Assign each observation to the most likely mode.
    3. Whiten within assigned mode:

            R_ij = (X_ij - mu_{k,j}) / sigma_{k,j}

    4. Test normality within each mode:
            d = 1: Shapiro-Wilk
            d >= 2: Henze-Zirkler

    5. Combine mode-level normality p-values with Stouffer's method.
    6. Test pooled Q_i = ||R_i||^2 against chi2_d.
    7. Return assignment certainty, mode summaries, normality results, and Q diagnostics.

    Parameters
    ----------
    X : array, shape (n,) or (n, d)
        Latent values after normalizing flow.

    weights : array, shape (K,)
        Mixture weights.

    means : array, shape (K,) or (K, d)
        Component means.

    scales : array, shape (K,) or (K, d)
        Diagonal component standard deviations.
    """
    res = whiten_by_assigned_mode_diag(X, weights, means, scales)

    R = res["R"]
    Q = res["Q"]
    labels = res["labels"]
    gamma = res["gamma"]
    max_resp = res["max_resp"]

    n = res["n"]
    d = res["d"]
    K = res["K"]

    assignment = {
        "mean_max_resp": float(np.mean(max_resp)),
        "median_max_resp": float(np.median(max_resp)),
        "min_max_resp": float(np.min(max_resp)),
        "frac_gt_0.80": float(np.mean(max_resp > 0.80)),
        "frac_gt_0.90": float(np.mean(max_resp > 0.90)),
        "frac_gt_0.95": float(np.mean(max_resp > 0.95)),
        "frac_gt_0.99": float(np.mean(max_resp > 0.99)),
        "assigned_counts": np.bincount(labels, minlength=K),
        "assigned_weights": np.bincount(labels, minlength=K) / n,
        "target_weights": res["weights"],
    }

    normality_results = normality_by_mode(
        R=R,
        labels=labels,
        min_n=min_mode_n,
        alpha=alpha,
    )

    normality_pvals = np.array([
        row["pvalue"]
        for row in normality_results
        if row["pvalue"] == row["pvalue"]
    ])

    stouffer_normality = stouffer_method(normality_pvals)

    pooled = pooled_chi2_tests(Q, d=d)

    mode_summary = mode_summaries_diag(
        R=R,
        Q=Q,
        labels=labels,
        max_resp=max_resp,
        weights=res["weights"],
        d=d,
        min_n=min_mode_n,
    )

    result = {
        "X": res["X"],
        "weights": res["weights"],
        "means": res["means"],
        "scales": res["scales"],
        "R": R,
        "Q": Q,
        "labels": labels,
        "gamma": gamma,
        "max_resp": max_resp,
        "assignment": assignment,
        "normality_by_mode": normality_results,
        "stouffer_normality": stouffer_normality,
        "pooled_chi2": pooled,
        "mode_summary": mode_summary,
        "n": n,
        "d": d,
        "K": K,
    }

    if verbose:
        print(f"--- Whitened mode-normality diagnostics (n={n}, d={d}, K={K}) ---")

        print("\n[1] Normality by assigned mode")
        for row in normality_results:
            print(f"    mode: {row['mode']}  n: {row['n']}  test: {row['test']}  pvalue: {row['pvalue']:.4f}")


        print("\n[1b] Stouffer combined normality p-value")
        print(f"     z = {stouffer_normality['stat']:.4f}")
        print(f"     p = {stouffer_normality['pvalue']:.4g}")
        print(f"     n tests = {stouffer_normality['n_tests']}")

        print("\n[2] Pooled Q against chi-square")
        print(f"    KS p-value:  {pooled['ks_pvalue']:.4f}")
        print(f"    CvM p-value: {pooled['cvm_pvalue']:.4f}")
        print(f"    Q mean:      {pooled['Q_mean']:.4f}  expected approx d={d}")
        print(f"    Q max:       {pooled['Q_max']:.4f}")
        print(f"    min tail p:  {pooled['tail_min']:.4g}")

    return result

