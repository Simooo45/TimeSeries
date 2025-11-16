import numpy as np
from typing import Tuple, Optional
from .TimeSeries import TimeSeries
from .ARIMA import ARIMA
from .GARCH import GARCH


class ARIMA_GARCH(TimeSeries):
    """
    Combined ARIMA + GARCH model for time series forecasting.

    Fits an ARIMA(p,d,q) for the mean and a GARCH(p_garch, q_garch)
    on residuals to capture time-varying volatility.
    """

    def __init__(
        self,
        series: np.ndarray,
        arima_p: int,
        arima_d: int,
        arima_q: int,
        garch_p: int = 1,
        garch_q: int = 1,
    ) -> None:
        """
        Initialize ARIMA-GARCH model.

        Parameters
        ----------
        series : np.ndarray
            The time series to model.
        arima_p : int
            AR order for ARIMA.
        arima_d : int
            Differencing order for ARIMA.
        arima_q : int
            MA order for ARIMA.
        garch_p : int, optional
            Number of GARCH lags (sigma^2 terms), by default 1.
        garch_q : int, optional
            Number of ARCH lags (squared residuals), by default 1.
        """
        super().__init__(series.copy())
        self.arima_model: Optional[ARIMA] = None
        self.garch_model: Optional[GARCH] = None
        self.resid: Optional[np.ndarray] = None
        self.arima_p = arima_p
        self.arima_d = arima_d
        self.arima_q = arima_q
        self.garch_p = garch_p
        self.garch_q = garch_q

    def fit(self) -> Tuple[dict, dict]:
        """
        Fit ARIMA on log returns for mean and GARCH on residuals for volatility.

        Returns
        -------
        Tuple[dict, dict]
            Parameters of ARIMA and GARCH models.
        """

        # 1. Fit ARIMA directly on the series
        self.arima_model = ARIMA(self.series, self.arima_p, self.arima_d, self.arima_q)
        arima_params = self.arima_model.fit()
        self.resid = np.array(arima_params["resid"])

        # 2. Fit GARCH on residuals
        self.garch_model = GARCH(self.resid, p=self.garch_p, q=self.garch_q)
        params_array = self.garch_model.fit()
        garch_params = {
            "omega": params_array[0],
            "alpha": params_array[1 : 1 + self.garch_q],
            "beta": params_array[1 + self.garch_q :],
        }

        return arima_params, garch_params

    def forecast(self, steps=5):
        # Forecast ARIMA mean
        mu_forecast = self.arima_model.forecast(steps)

        # Forecast GARCH volatility
        _, sigma_forecast = self.garch_model.forecast_variance(steps=steps)
        sigma_forecast = np.sqrt(sigma_forecast)
        return mu_forecast, sigma_forecast

    def aic(self) -> float:
        """
        Compute Akaike Information Criterion (AIC) for the ARIMA-GARCH model.

        Uses Gaussian quasi-log-likelihood:
        logL = -0.5 * sum( log(2*pi*sigma_t^2) + eps_t^2 / sigma_t^2 )

        Returns
        -------
        float
            The AIC value: 2*k - 2*logL
        """

        # Check that both models are fitted
        if self.arima_model is None or self.garch_model is None:
            raise ValueError("You must call fit() before computing AIC.")

        # Extract residuals from ARIMA
        eps = self.resid.copy()

        # Compute conditional variances from the fitted GARCH model
        # forecast_variance returns (sigma2_full, sigma2_future)
        sigma2, _ = self.garch_model.forecast_variance(steps=0)
        sigma2 = np.array(sigma2)

        # Ensure alignment
        n = min(len(eps), len(sigma2))
        eps = eps[-n:]
        sigma2 = sigma2[-n:]

        # Compute Gaussian log-likelihood
        logL = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (eps**2) / sigma2)

        # Count number of parameters:
        # ARIMA: c + phi(p) + theta(q)
        k_arima = 1 + self.arima_p + self.arima_q

        # GARCH: omega + alpha(q) + beta(p)
        k_garch = 1 + self.garch_q + self.garch_p

        k = k_arima + k_garch

        # Return AIC
        return 2 * k - 2 * logL
