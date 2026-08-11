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


class InfluentialOutlierMetric:
    """
    Class for computing the Influential Outlier Metric (IOM) using normalizing flows and mixture models.
    """

    def __init__(self):
        """
        Initialize the InfluentialOutlierMetric object.
        """

        self.i_star = None
        self.results_shap = None
        self.results_resid = None


    def norm_flow(
        self,
        data,
        K,
        hidden,
        layers,
        epoch,
        lam,
        loc,
        scale,
        tail_bound=3,
        num_bins=8,
        test_size=0.2,
        random_state=42,
        torch_seed=42
    ):
        """
        Trains a normalizing flow model with a Gaussian mixture base distribution.

        Parameters:
        data (np.ndarray): Input data.
        K (int): Number of mixture components.
        hidden (int): Number of hidden units.
        layers (int): Number of layers.
        epoch (int): Number of training epochs.
        lam (float): Regularization parameter.
        loc (list): Means for mixture components.
        scale (list): Scales for mixture components.
        tail_bound (float): Tail bound for spline.
        num_bins (int): Number of bins for spline.
        test_size (float): Fraction of data for test split.
        random_state (int): Random seed for train/test split.
        torch_seed (int): Random seed for torch.

        Returns:
        mod (nf.NormalizingFlow): Trained normalizing flow model.
        z_all (np.ndarray): Latent representations.
        dist_sq (np.ndarray): Squared distances in latent space.
        """
        torch.manual_seed(torch_seed)
        # torch.cuda.manual_seed_all(torch_seed)

        device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(device)
        latent_size = data.shape[1]

        x_train, x_test = train_test_split(data, test_size=test_size, random_state=random_state)

        scaler = StandardScaler(with_std=False)
        x_train_s = scaler.fit_transform(x_train).astype(np.float32)
        x_test_s = scaler.transform(x_test).astype(np.float32)
        x_all_s = scaler.transform(data).astype(np.float32)

        flows = []
        for _ in range(K):
            flows += [
                nf.flows.AutoregressiveRationalQuadraticSpline(
                    latent_size, layers, hidden, permute_mask=True, init_identity=True,
                    tail_bound=tail_bound, num_bins=num_bins
                )
            ]

        q0 = nf.distributions.GaussianMixture(
            n_modes=len(loc),
            dim=latent_size,
            loc=[np.array([loc[i]] * latent_size, dtype=np.float32) for i in range(len(loc))],
            scale=[np.array([scale[i]] * latent_size, dtype=np.float32) for i in range(len(scale))],
            trainable=True,
        )

        mod = nf.NormalizingFlow(q0, flows).float().to(device)

        x_train_t = torch.tensor(x_train_s, dtype=torch.float32, device=device)
        x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
        x_test_s_t = torch.tensor(x_test_s, dtype=torch.float32, device=device)
        x_all_t = torch.tensor(x_all_s, dtype=torch.float32, device=device)


        optimizer = torch.optim.AdamW(mod.parameters(), lr=3e-4)
        mod.train()
        for _ in tqdm(range(epoch), desc="Normalizing flow"):
            optimizer.zero_grad()
            z_train, jac = mod.inverse_and_log_det(x_train_t)
            loss = mod.forward_kld(x_train_t) + lam * torch.mean(jac ** 2)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

            mod.eval()
        with torch.no_grad():
            z_test = mod.inverse(x_test_s_t)
            dist_sq = torch.norm(z_test - x_test_s_t, dim=1) ** 2
            z_all = mod.inverse(x_all_t).cpu().numpy()

        return mod, z_all, dist_sq.cpu().numpy()

    def find_best_lambda(
        self,
        data,
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
        Find the best lambda value for the normalizing flow model based on goodness-of-fit.

        Parameters:
        lambdas (np.ndarray): Array of lambda values to try.
        K (int): Number of mixture components.
        hidden (int): Number of hidden units.
        layers (int): Number of layers.
        epoch (int): Number of training epochs.
        loc (list): Means for mixture components.
        scale (list): Scales for mixture components.
        tail_bound (float): Tail bound for spline.
        num_bins (int): Number of bins for spline.
        alpha (float): Significance threshold for goodness-of-fit.
        torch_seed (int): Random seed for torch.

        Returns:
        dict: Dictionary containing best lambda, model, latent, weights, means, covs, test_distance, chi_z, KS_p, and post.
        """
        if lambdas is None:
            lambdas = np.concatenate([np.array([0]), np.exp(np.linspace(-5, 1, 10))])

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
                data, K, hidden, layers, epoch, lam, loc, scale,
                tail_bound=tail_bound, num_bins=num_bins, torch_seed=torch_seed
            )

            weights = np.exp(mod.q0.weight_scores.cpu().detach().numpy())[0]
            weights /= weights.sum()
            weights = [weights[i] for i in range(len(loc))]
            means = [mod.q0.loc.cpu().detach().numpy()[0][i] for i in range(len(loc))]
            scales = [np.exp(mod.q0.log_scale.cpu().detach().numpy())[0, i] for i in range(len(scale))]

            test_results = whitened_mode_normality_workflow_diag(
                        z,
                        weights,
                        means,
                        scales,
                        min_mode_n=20,
                        alpha=0.05,
                        verbose=True,
                    )
       
            pval = passes = ((min(test_results['normality_by_mode'][i]['pvalue'] for i in range(test_results['K'])) > 0.01) and
                            (test_results['stouffer_normality']['pvalue'] > 0.05) and
                            (test_results['pooled_chi2']['cvm_pvalue'] > 0.05))
           
            if pval:
                z_last_pass = z
                mod_last_pass = mod
                lam_final = lam
                dist_pass = dist
                test_results_last_pass = test_results
            else:
                break

        if z_last_pass is None:
            raise RuntimeError("Increase complexity of normalizing flow.")

        print(f"Final lambda: {lam_final:.4f}")
        print(f"Mean test distance: {dist_pass.mean():.4f}")

        weights = np.exp(mod_last_pass.q0.weight_scores.cpu().detach().numpy())[0]
        weights /= weights.sum()
        weights = [np.array(weights[i]) for i in range(len(loc))]
        means = [mod_last_pass.q0.loc.cpu().detach().numpy()[0][i] for i in range(len(loc))]
        scales = [np.exp(mod_last_pass.q0.log_scale.cpu().detach().numpy())[0, i] for i in range(len(loc))]

        print(f"Final weights: {[np.round(x, 4) for x in weights]}")
        print(f"Final means: {[np.round(x, 4) for x in means]}")
        print(f"Final scales: {[np.round(x, 4) for x in scales]}")

       

        return {
            "lambda": lam_final,
            "model": mod_last_pass,
            "latent": z_last_pass.reshape(-1),
            "weights": weights,
            "means": means,
            "scales": scales,
            "test_distance": dist_pass,
            "chi_z": test_results_last_pass['Q'],
        }

    @staticmethod
    def base_parameter_count(n_modes, dim):
        """
        Parameter count for a diagonal Gaussian mixture base.

        Parameters:
        n_modes (int): Number of mixture components.
        dim (int): Dimensionality of each component.

        Returns:
        int: Total parameter count.
        """
        return 2 * n_modes * dim + (n_modes - 1)

    @staticmethod
    def select_flow_by_transport_modes(tune_results, dim=1, rho=None):
        """
        Selection rule: minimize M + rho * K_m
        where K_m = base_parameter_count(n_modes, dim).

        Parameters:
        tune_results (list): List of tuning results dictionaries.
        dim (int): Dimensionality of each component.
        rho (float): Regularization parameter.

        Returns:
        dict: Dictionary containing selected index, and table of scores.
        """
        n_val = len(tune_results)
        if rho is None:
            rho = np.log(n_val) / n_val

        table = []
        for idx, res in enumerate(tune_results):
            n_modes = len(res['weights'])
            M = float(np.mean(res['test_distance']))
            K_m = InfluentialOutlierMetric.base_parameter_count(n_modes=n_modes, dim=dim)
            score = M + rho * K_m
            table.append({
                'index': idx,
                'M': M,
                'n_modes': n_modes,
                'K_m': K_m,
                'rho': rho,
                'score': score,
            })

        selected = min(table, key=lambda x: x["score"])
        return {
            'selected_index': selected['index'],
            'table': table,
        }

    @staticmethod
    def find_threshold(p, alpha=[0.05, 0.01], bracket=(1e-8, 1000)):
        """
        Compute critical thresholds for IOM values based on χ²-product distribution.

        Parameters:
        alpha (list): List of significance levels.
        bracket (tuple): Bracket for root finding.

        Returns:
        list: List of critical thresholds for each alpha.
        """
        def chi_prod(w, m1, m2=1):
            return (w**((m1 + m2) / 4 - 1)) / \
                   (2**((m1 + m2) / 2 - 1) * gamma(m1 / 2) * gamma(m2 / 2)) * \
                   kv((m1 - m2) / 2, np.sqrt(w))

        def survival_prob(i, m1, m2=1):
            val, _ = quad(chi_prod, i, np.inf, args=(m1, m2))
            return val

        i_star = []
        for a in alpha:
            func = lambda i: survival_prob(i, p, 1) - a
            i_star.append(brentq(func, bracket[0], bracket[1]))
        print(f"Threshold at 0.05: {i_star[0]:.3f}")
        print(f"Threshold at 0.01: {i_star[1]:.3f}")
        return self.i_star

    def IOM(self, results_shap, results_resid):
        """
        Compute Influential Outlier Metric.

        Parameters:
        results_shap (dict): Results from SHAP normalizing flow.
        results_resid (dict): Results from residual normalizing flow.

        Returns:
        pd.Series: Series of IOM values.
        """
        self.results_shap = results_shap
        self.results_resid = results_resid
        self.IOM_ = pd.Series(results_shap['chi_z'] * self.results_resid['chi_z'], name="IOM")
        return self.IOM_