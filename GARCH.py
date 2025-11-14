import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize
from .TimeSeries import TimeSeries


class GARCH(TimeSeries):
    """
    GARCH(p, q) model for conditional variance of a time series.

    Attributes
    ----------
    p : int
        Number of GARCH lags (sigma^2 terms).
    q : int
        Number of ARCH lags (squared residual terms).
    params : Optional[np.ndarray]
        Estimated parameters: [omega, alpha_1..alpha_q, beta_1..beta_p].
    """

    def __init__(self, series: np.ndarray, p: int = 1, q: int = 1) -> None:
        """
        Initialize GARCH(p, q) model.

        Parameters
        ----------
        series : np.ndarray
            Time series residuals (usually from ARIMA, should be mean-zero).
        p : int, optional
            Number of GARCH lags (sigma^2 terms), by default 1.
        q : int, optional
            Number of ARCH lags (squared residuals), by default 1.
        """
        super().__init__(series)
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative integers.")
        self.p = p
        self.q = q
        self.params: Optional[np.ndarray] = None

    def _unpack_params(
        self, params: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Unpack parameter vector into omega, alpha, beta."""
        omega = float(params[0])
        alphas = np.array(params[1 : 1 + self.q]) if self.q > 0 else np.array([])
        betas = (
            np.array(params[1 + self.q : 1 + self.q + self.p])
            if self.p > 0
            else np.array([])
        )
        return omega, alphas, betas

    def _garch_loglik(self, params: np.ndarray) -> float:
        """Negative log-likelihood for GARCH(p,q) with Gaussian errors."""
        omega, alphas, betas = self._unpack_params(params)
        y = np.asarray(self.series)
        n = len(y)

        max_lag = max(self.p, self.q, 1)
        sigma2 = np.empty(n)
        sigma2[:max_lag] = np.var(y)

        ll = 0.0
        for t in range(max_lag, n):
            arch_term = (
                np.sum(alphas * (y[t - np.arange(1, self.q + 1)] ** 2))
                if self.q > 0
                else 0.0
            )
            garch_term = (
                np.sum(betas * sigma2[t - np.arange(1, self.p + 1)])
                if self.p > 0
                else 0.0
            )
            sigma2[t] = omega + arch_term + garch_term
            if sigma2[t] <= 0:
                return 1e12
            ll += 0.5 * (
                np.log(2 * np.pi) + np.log(sigma2[t]) + (y[t] ** 2) / sigma2[t]
            )
        return ll

    def fit(self) -> np.ndarray:
        """Estimate GARCH(p, q) parameters by minimizing negative log-likelihood."""
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
        bounds = [(1e-12, None)] + [(0.0, 1.0) for _ in range(self.q + self.p)]

        # Constrain sum(alpha + beta) < 1 for stationarity
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
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        if not result.success:
            raise RuntimeError(f"GARCH optimization failed: {result.message}")

        self.params = np.array(result.x)
        return self.params

    def forecast_variance(self, steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute in-sample conditional variance and multi-step forecasted variance.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            sigma2: in-sample conditional variance
            sigma_forecast: multi-step forecasted variance
        """
        if self.params is None:
            raise ValueError("Call fit() before forecast_variance()")

        omega, alphas, betas = self._unpack_params(self.params)
        y = np.asarray(self.series)
        n = len(y)
        max_lag = max(self.p, self.q, 1)

        # Initialize in-sample conditional variance with sample variance
        sigma2 = np.empty(n)
        sigma2[:max_lag] = np.var(y)

        # Compute in-sample conditional variance
        for t in range(max_lag, n):
            arch_term = (
                np.sum(alphas * y[t - np.arange(1, self.q + 1)] ** 2)
                if self.q > 0
                else 0
            )
            garch_term = (
                np.sum(betas * sigma2[t - np.arange(1, self.p + 1)])
                if self.p > 0
                else 0
            )
            sigma2[t] = omega + arch_term + garch_term

        # Multi-step forecast
        sigma_forecast = np.zeros(steps)
        last_y2 = y[-self.q :] ** 2 if self.q > 0 else np.array([])
        last_sigma = sigma2[-self.p :] if self.p > 0 else np.array([])

        for t in range(steps):
            # For future unknown residuals, use the last forecasted variance for ARCH term
            arch_term = np.sum(alphas * last_y2[-self.q :]) if self.q > 0 else 0
            garch_term = np.sum(betas * last_sigma[-self.p :]) if self.p > 0 else 0

            sigma_next = omega + arch_term + garch_term
            sigma_forecast[t] = sigma_next

            # Update for next step: future squared residuals approximated by forecasted variance
            if self.q > 0:
                last_y2 = np.append(last_y2, sigma_next)
            if self.p > 0:
                last_sigma = np.append(last_sigma, sigma_next)

        return sigma2, sigma_forecast

    def aic(self) -> float:
        """Compute Akaike Information Criterion for fitted GARCH model."""
        if self.params is None:
            raise ValueError("Fit the model first.")
        n_params = 1 + self.q + self.p
        neg_loglik = self._garch_loglik(self.params)
        return 2 * n_params + 2 * neg_loglik

    def bic(self) -> float:
        """Compute Bayesian Information Criterion for fitted GARCH model."""
        if self.params is None:
            raise ValueError("Fit the model first.")
        n_params = 1 + self.q + self.p
        n = len(self.series)
        neg_loglik = self._garch_loglik(self.params)
        return np.log(n) * n_params + 2 * neg_loglik
