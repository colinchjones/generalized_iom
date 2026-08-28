from mode_assign import *

import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import kv, gamma
from scipy.integrate import quad
from scipy.optimize import brentq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import normflows as nf
from tqdm import tqdm


class ConditionalFlowFixedBase(nf.NormalizingFlow):
    """
    Normalizing flow with optional context and a fixed base distribution.

    If context is provided, the flow models z = f(x | c). Otherwise it behaves
    as an unconditional flow z = f(x). Context affects only the transformations,
    not the Gaussian-mixture base distribution.
    """

    def forward(self, z, context=None):
        """Transform latent values to the data space."""
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transform latent values and return the accumulated log-Jacobian."""
        log_det = torch.zeros(len(z), dtype=z.dtype, device=z.device)

        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d

        return z, log_det

    def inverse(self, x, context=None):
        """Transform observations from the data space to the latent space."""
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)

        return x

    def inverse_and_log_det(self, x, context=None):
        """Transform observations to latent space and return the log-Jacobian."""
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)

        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d

        return x, log_det

    def log_prob(self, x, context=None):
        """Evaluate observation-level log-density under the flow."""
        z, log_det = self.inverse_and_log_det(x, context=context)
        return self.q0.log_prob(z) + log_det

    def forward_kld(self, x, context=None):
        """Return the negative mean log-likelihood training objective."""
        return -torch.mean(self.log_prob(x, context=context))

    def sample(self, num_samples=1, context=None):
        """Draw samples from the fitted flow."""
        z, log_q = self.q0(num_samples)

        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det

        return z, log_q


class InfluentialOutlierMetric:
    """
    Compute the Influential Outlier Metric (IOM) using normalizing flows.

    The class transforms SHAP and residual quantities to Gaussian-mixture latent
    distributions, tunes Jacobian regularization using goodness-of-fit tests,
    and combines the transformed components into the final IOM.
    """

    def __init__(self):
        """Initialize the IOM object."""
        self.i_star = None
        self.results_shap = None
        self.results_resid = None

    def norm_flow(
        self,
        data,
        context=None,
        K=None,
        hidden=None,
        layers=None,
        epoch=None,
        lam=None,
        loc=None,
        scale=None,
        tail_bound=3,
        num_bins=8,
        test_size=0.2,
        random_state=42,
        torch_seed=42
    ):
        """
        Fit a neural spline flow with optional conditioning.

        Parameters
        ----------
        data : array-like
            Observations to transform.
        context : array-like, optional
            Conditioning variables aligned with `data`.
        K : int
            Number of spline transformations.
        hidden : int
            Hidden units in each spline conditioner.
        layers : int
            Hidden blocks in each spline conditioner.
        epoch : int
            Number of training epochs.
        lam : float
            Jacobian regularization strength.
        loc, scale : sequence
            Initial means and scales of the Gaussian-mixture base.
        tail_bound : float, default=3
            Spline tail boundary.
        num_bins : int, default=8
            Number of spline bins.
        test_size : float, default=0.2
            Fraction used for transport-distance evaluation.
        random_state : int, default=42
            Train/test split seed.
        torch_seed : int, default=42
            PyTorch random seed.

        Returns
        -------
        mod : ConditionalFlowFixedBase
            Fitted normalizing flow.
        z_all : ndarray
            Latent representation of all observations.
        dist_sq : ndarray
            Held-out squared transport distances.
        """

        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(device)

        data = np.asarray(data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        latent_size = data.shape[1]

        if context is not None:
            context = np.asarray(context)

            if context.ndim == 1:
                context = context.reshape(-1, 1)

            if len(data) != len(context):
                raise ValueError(
                    f"`data` and `context` must have the same number of rows; "
                    f"got {len(data)} and {len(context)}."
                )

            context_size = context.shape[1]

        idx = np.arange(len(data))

        idx_train, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state
        )

        x_train = data[idx_train]
        x_test = data[idx_test]

        x_scaler = StandardScaler(with_std=False)
        x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
        x_test_s = x_scaler.transform(x_test).astype(np.float32)
        x_all_s = x_scaler.transform(data).astype(np.float32)

        if context is not None:
            c_train = context[idx_train]
            c_test = context[idx_test]

            c_scaler = StandardScaler()
            c_train_s = c_scaler.fit_transform(c_train).astype(np.float32)
            c_test_s = c_scaler.transform(c_test).astype(np.float32)
            c_all_s = c_scaler.transform(context).astype(np.float32)

        else:
            c_scaler = None
            c_train_s = None
            c_test_s = None
            c_all_s = None

        flows = []

        for _ in range(K):
            if context is None:
                flows += [
                    nf.flows.AutoregressiveRationalQuadraticSpline(
                        latent_size,
                        layers,
                        hidden,
                        permute_mask=True,
                        init_identity=True,
                        tail_bound=tail_bound,
                        num_bins=num_bins,
                    )
                ]

            else:
                flows += [
                    nf.flows.AutoregressiveRationalQuadraticSpline(
                        latent_size,
                        layers,
                        hidden,
                        num_context_channels=context_size,
                        permute_mask=True,
                        init_identity=True,
                        tail_bound=tail_bound,
                        num_bins=num_bins,
                    )
                ]

        q0 = nf.distributions.GaussianMixture(
            n_modes=len(loc),
            dim=latent_size,
            loc=[
                np.array([loc[i]] * latent_size, dtype=np.float32)
                for i in range(len(loc))
            ],
            scale=[
                np.array([scale[i]] * latent_size, dtype=np.float32)
                for i in range(len(scale))
            ],
            trainable=True,
        )

        mod = ConditionalFlowFixedBase(q0, flows).float().to(device)

        mod.x_scaler = x_scaler
        mod.context_scaler = c_scaler

        x_train_t = torch.tensor(
            x_train_s,
            dtype=torch.float32,
            device=device
        )

        x_test_t = torch.tensor(
            x_test_s,
            dtype=torch.float32,
            device=device
        )

        x_all_t = torch.tensor(
            x_all_s,
            dtype=torch.float32,
            device=device
        )

        if context is not None:
            c_train_t = torch.tensor(
                c_train_s,
                dtype=torch.float32,
                device=device
            )

            c_test_t = torch.tensor(
                c_test_s,
                dtype=torch.float32,
                device=device
            )

            c_all_t = torch.tensor(
                c_all_s,
                dtype=torch.float32,
                device=device
            )

        else:
            c_train_t = None
            c_test_t = None
            c_all_t = None

        optimizer = torch.optim.AdamW(mod.parameters(), lr=3e-4)

        mod.train()

        for _ in tqdm(range(epoch), desc="Normalizing flow"):
            optimizer.zero_grad()

            _, jac = mod.inverse_and_log_det(
                x_train_t,
                context=c_train_t
            )

            loss = (
                mod.forward_kld(
                    x_train_t,
                    context=c_train_t
                )
                + lam * torch.mean(jac ** 2)
            )

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        mod.eval()

        with torch.no_grad():

            z_test = mod.inverse(
                x_test_t,
                context=c_test_t
            )

            dist_sq = torch.norm(
                z_test - x_test_t,
                dim=1
            ) ** 2

            z_all = mod.inverse(
                x_all_t,
                context=c_all_t
            ).cpu().numpy()

        return mod, z_all, dist_sq.cpu().numpy()

    def find_best_lambda(
        self,
        data,
        context=None,
        lambdas=None,
        K=None,
        hidden=None,
        layers=None,
        epoch=None,
        loc=None,
        scale=None,
        tail_bound=None,
        num_bins=None,
        alpha=0.15,
        torch_seed=42
    ):
        """
        Select the largest Jacobian penalty satisfying the latent
        goodness-of-fit criteria.

        Returns a dictionary containing the selected model, latent values,
        mixture parameters, preprocessing scalers, transport distances, and
        chi-square-transformed values.
        """

        if lambdas is None:
            lambdas = np.concatenate([
                np.array([0]),
                np.exp(np.linspace(-5, 1, 10))
            ])

        data = np.asarray(data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        z_last_pass = None
        mod_last_pass = None
        dist_pass = None
        lam_final = None
        test_results_last_pass = None

        for j, lam in enumerate(lambdas):
            print(f"lambda: {lam}")

            mod, z, dist = self.norm_flow(
                data,
                context,
                K,
                hidden,
                layers,
                epoch,
                lam,
                loc,
                scale,
                tail_bound=tail_bound,
                num_bins=num_bins,
                torch_seed=torch_seed
            )

            weights = np.exp(
                mod.q0.weight_scores.cpu().detach().numpy()
            )[0]

            weights /= weights.sum()

            weights = [
                weights[i]
                for i in range(len(loc))
            ]

            means = [
                mod.q0.loc.cpu().detach().numpy()[0][i]
                for i in range(len(loc))
            ]

            scales = [
                np.exp(
                    mod.q0.log_scale.cpu().detach().numpy()
                )[0, i]
                for i in range(len(scale))
            ]

            test_results = whitened_mode_normality_workflow_diag(
                z,
                weights,
                means,
                scales,
                min_mode_n=20,
                alpha=0.05,
                verbose=True,
            )

            mode_pvalues = [
                row["pvalue"]
                for row in test_results["normality_by_mode"]
                if np.isfinite(row["pvalue"])
            ]
            all_modes_tested = len(mode_pvalues) == test_results["m"]

            pval = passes = (
                all_modes_tested
                and min(mode_pvalues) > 0.01
                and test_results["stouffer_normality"]["pvalue"] > 0.05
                and test_results["pooled_chi2"]["cvm_pvalue"] > 0.05
            )

            if pval:
                z_last_pass = z
                mod_last_pass = mod
                lam_final = lam
                dist_pass = dist
                test_results_last_pass = test_results

            else:
                break

        if z_last_pass is None:
            raise RuntimeError(
                "Increase complexity of normalizing flow."
            )

        print(f"Final lambda: {lam_final:.4f}")
        print(f"Mean test distance: {dist_pass.mean():.4f}")

        weights = np.exp(
            mod_last_pass.q0.weight_scores.cpu().detach().numpy()
        )[0]

        weights /= weights.sum()

        weights = [
            np.array(weights[i])
            for i in range(len(loc))
        ]

        means = [
            mod_last_pass.q0.loc.cpu().detach().numpy()[0][i]
            for i in range(len(loc))
        ]

        scales = [
            np.exp(
                mod_last_pass.q0.log_scale.cpu().detach().numpy()
            )[0, i]
            for i in range(len(loc))
        ]

        print(
            f"Final weights: "
            f"{[np.round(x, 4) for x in weights]}"
        )

        print(
            f"Final means: "
            f"{[np.round(x, 4) for x in means]}"
        )

        print(
            f"Final scales: "
            f"{[np.round(x, 4) for x in scales]}"
        )

        return {
            "lambda": lam_final,
            "model": mod_last_pass,
            "x_scaler": mod_last_pass.x_scaler,
            "context_scaler": mod_last_pass.context_scaler,
            "latent": (
                z_last_pass
                if z_last_pass.shape[1] > 1
                else z_last_pass.reshape(-1)
            ),
            "weights": weights,
            "means": means,
            "scales": scales,
            "test_distance": dist_pass,
            "chi_z": test_results_last_pass["Q"],
        }

    @staticmethod
    def base_parameter_count(n_modes, dim):
        """
        Return the number of free parameters in a diagonal Gaussian-mixture base.
        """
        return 2 * n_modes * dim + (n_modes - 1)

    @staticmethod
    def select_flow_by_transport_modes(
        tune_results,
        dim=1,
        rho=None
    ):
        """
        Select the flow minimizing transport distance plus a complexity penalty.

        The score is M + rho * K_m, where M is mean held-out transport distance
        and K_m is the number of Gaussian-mixture base parameters.
        """

        n_val = len(tune_results)

        if rho is None:
            rho = np.log(n_val) / n_val

        table = []

        for idx, res in enumerate(tune_results):
            n_modes = len(res["weights"])
            M = float(np.mean(res["test_distance"]))

            K_m = InfluentialOutlierMetric.base_parameter_count(
                n_modes=n_modes,
                dim=dim
            )

            score = M + rho * K_m

            table.append({
                "index": idx,
                "M": M,
                "n_modes": n_modes,
                "K_m": K_m,
                "rho": rho,
                "score": score,
            })

        selected = min(
            table,
            key=lambda x: x["score"]
        )

        return {
            "selected_index": selected["index"],
            "table": table,
        }

    @staticmethod
    def find_threshold(
        p,
        alpha=[0.05, 0.01],
        bracket=(1e-8, 1000)
    ):
        """
        Compute critical IOM thresholds from the chi-square product distribution.
        """

        def chi_prod(w, m1, m2=1):
            return (
                w ** ((m1 + m2) / 4 - 1)
                / (
                    2 ** ((m1 + m2) / 2 - 1)
                    * gamma(m1 / 2)
                    * gamma(m2 / 2)
                )
                * kv(
                    (m1 - m2) / 2,
                    np.sqrt(w)
                )
            )

        def survival_prob(i, m1, m2=1):
            val, _ = quad(
                chi_prod,
                i,
                np.inf,
                args=(m1, m2)
            )

            return val

        i_star = []

        for a in alpha:
            func = lambda i: survival_prob(i, p, 1) - a

            i_star.append(
                brentq(
                    func,
                    bracket[0],
                    bracket[1]
                )
            )

        print(f"Threshold at 0.05: {i_star[0]:.3f}")
        print(f"Threshold at 0.01: {i_star[1]:.3f}")

        return i_star

    def IOM(self, results_shap, results_resid):
        """
        Compute the IOM as the product of the chi-square-transformed SHAP and
        residual components.
        """

        self.results_shap = results_shap
        self.results_resid = results_resid

        self.IOM_ = pd.Series(
            results_shap["chi_z"]
            * self.results_resid["chi_z"],
            name="IOM"
        )

        return self.IOM_
