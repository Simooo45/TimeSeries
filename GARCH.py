import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize

from .TimeSeries import TimeSeries


class GARCH(TimeSeries):
    """
    GARCH(p, q) model for conditional variance.

    Attributes
    ----------
    p : int
        Number of GARCH lags (sigma^2 terms).
    q : int
        Number of ARCH lags (squared residual terms).
    params_ : Optional[np.ndarray]
        Estimated parameter vector in the order [omega, alpha_1..alpha_q, beta_1..beta_p].
    """

    def __init__(self, series: np.ndarray, p: int = 1, q: int = 1) -> None:
        """
        Initialize the GARCH model.

        Parameters
        ----------
        series : Sequence[float]
            The observed time series (usually mean-zero residuals or returns).
        p : int, optional
            Number of GARCH lags, by default 1.
        q : int, optional
            Number of ARCH lags, by default 1.
        """
        super().__init__(series)
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative integers.")
        self.p = int(p)
        self.q = int(q)
        self.params: Optional[np.ndarray] = None

    def _unpack_params(
        self, params: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Unpack parameter vector into omega, alphas and betas.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [omega, alpha_1..alpha_q, beta_1..beta_p].

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            omega, alphas, betas
        """
        omega = float(params[0])
        alphas = np.array(params[1 : 1 + self.q]) if self.q > 0 else np.array([])
        betas = (
            np.array(params[1 + self.q : 1 + self.q + self.p])
            if self.p > 0
            else np.array([])
        )
        return omega, alphas, betas

    def _garch_loglik(self, params: np.ndarray) -> float:
        """
        Negative conditional log-likelihood for GARCH(p, q) with Gaussian errors.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [omega, alpha_1..alpha_q, beta_1..beta_p].

        Returns
        -------
        float
            Negative log-likelihood (to be minimized).
        """
        omega, alphas, betas = self._unpack_params(params)
        y = np.asarray(self.series)
        n = len(y)

        max_lag = max(self.p, self.q, 1)
        sigma2 = np.empty(n)
        sigma2.fill(np.var(y))

        ll = 0.0
        for t in range(max_lag, n):
            arch_term = 0.0
            for i in range(1, self.q + 1):
                arch_term += alphas[i - 1] * (y[t - i] ** 2)
            garch_term = 0.0
            for j in range(1, self.p + 1):
                garch_term += betas[j - 1] * sigma2[t - j]
            sigma2[t] = omega + arch_term + garch_term
            if sigma2[t] <= 0:
                return 1e12
            ll += 0.5 * (
                np.log(2 * np.pi) + np.log(sigma2[t]) + (y[t] ** 2) / sigma2[t]
            )

        return ll

    def fit(self) -> np.ndarray:
        """
        Estimate GARCH(p, q) parameters by minimizing the negative log-likelihood.

        Returns
        -------
        np.ndarray
            Estimated parameters in the order [omega, alpha_1..alpha_q, beta_1..beta_p].

        Raises
        ------
        RuntimeError
            If optimization fails.
        """
        y = np.asarray(self.series)
        var_y = np.var(y)

        omega0 = 0.1 * var_y
        alpha0 = np.array([0.1 / max(1, self.q)] * self.q)
        beta0 = np.array([0.8 / max(1, self.p)] * self.p)

        init = (
            np.concatenate(([omega0], alpha0, beta0))
            if (self.q + self.p) > 0
            else np.array([omega0])
        )

        bounds = []
        bounds.append((1e-12, None))
        bounds.extend([(0.0, 1.0) for _ in range(self.q + self.p)])

        def constraint_sum(params: np.ndarray) -> float:
            _, alphas, betas = self._unpack_params(params)
            return 1.0 - (np.sum(alphas) + np.sum(betas)) - 1e-8

        cons = {"type": "ineq", "fun": constraint_sum}

        result = minimize(
            self._garch_loglik,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.params = np.ndarray(result.x)

        return self.params

    def forecast_variance(self, steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute in-sample conditional variances and multi-step forecasts.

        The recursion uses observed squared values where available and uses
        previously forecasted variances for future terms.

        Parameters
        ----------
        steps : int, optional
            Number of steps to forecast, by default 5.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (in_sample_sigma2, forecasted_sigma2_array)
        """
        if self.params is None:
            raise ValueError("Call fit() before forecast_variance().")

        omega, alphas, betas = self._unpack_params(self.params)
        y = np.asarray(self.series)
        n = len(y)
        max_lag = max(self.p, self.q, 1)

        sigma2 = np.empty(n)
        sigma2.fill(np.var(y))

        for t in range(max_lag, n):
            arch_term = 0.0
            for i in range(1, self.q + 1):
                arch_term += alphas[i - 1] * (y[t - i] ** 2)
            garch_term = 0.0
            for j in range(1, self.p + 1):
                garch_term += betas[j - 1] * sigma2[t - j]
            sigma2[t] = omega + arch_term + garch_term

        extended_len = n + steps
        sigma_ext = np.empty(extended_len)
        sigma_ext[:n] = sigma2
        y_ext = np.empty(extended_len)
        y_ext[:n] = y
        y_ext[n:] = 0.0

        for t in range(n, extended_len):
            arch_term = 0.0
            for i in range(1, self.q + 1):
                idx = t - i
                if idx >= 0:
                    if idx < n:
                        arch_term += alphas[i - 1] * (y_ext[idx] ** 2)
                    else:
                        arch_term += alphas[i - 1] * sigma_ext[idx]
            garch_term = 0.0
            for j in range(1, self.p + 1):
                idx = t - j
                if idx >= 0:
                    garch_term += betas[j - 1] * sigma_ext[idx]
            sigma_ext[t] = omega + arch_term + garch_term

        forecasted = sigma_ext[n:]
        return sigma2, forecasted


if __name__ == "__main__":
    np.random.seed(0)
    n = 500
    eps = np.random.normal(size=n)
    true_omega, true_alphas, true_betas = 0.1, np.array([0.1]), np.array([0.8])
    p, q = 1, 1
    sigma2 = np.empty(n)
    sigma2.fill(true_omega / (1 - true_alphas.sum() - true_betas.sum()))
    y = np.empty(n)
    y[0] = eps[0] * np.sqrt(sigma2[0])
    for t in range(1, n):
        sigma2[t] = (
            true_omega
            + true_alphas[0] * (y[t - 1] ** 2)
            + true_betas[0] * sigma2[t - 1]
        )
        y[t] = eps[t] * np.sqrt(sigma2[t])

    model = GARCH(y, p=p, q=q)
    params = model.fit()
    in_sample, forecast = model.forecast_variance(steps=5)

    print("Estimated parameters:", params)
    print("Last in-sample variances:", in_sample[-5:])
    print("Forecast variances:", forecast)
