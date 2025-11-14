import numpy as np
from typing import Tuple, Optional

from .TimeSeries import TimeSeries
from .ARIMA import ARIMA
from .GARCH import GARCH


class ARIMA_GARCH(TimeSeries):
    """
    Combined ARIMA + GARCH model for time series forecasting.
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
        super().__init__(series)
        self.arima_p = arima_p
        self.arima_d = arima_d
        self.arima_q = arima_q
        self.garch_p = garch_p
        self.garch_q = garch_q

        self.arima_model: Optional[ARIMA] = None
        self.garch_model: Optional[GARCH] = None
        self.resid: Optional[np.ndarray] = None
        self.arima_params: dict = {}
        self.garch_params: dict = {}

    def fit(self) -> Tuple[dict, dict]:
        """
        Fit ARIMA for the mean and GARCH for volatility.
        If GARCH orders were not provided, automatically estimate them.
        """
        # Fit ARIMA
        self.arima_model = ARIMA(self.series, self.arima_p, self.arima_d, self.arima_q)
        self.arima_params = self.arima_model.fit()

        # Compute residuals
        mu_hat = self.arima_model.forecast_in_sample()
        self.resid = self.series[self.arima_d :] - mu_hat

        # Fit GARCH
        self.garch_model = GARCH(self.resid, p=self.garch_p, q=self.garch_q)
        params_array = self.garch_model.fit()
        self.garch_params = {
            "omega": params_array[0],
            "alpha": params_array[1],
            "beta": params_array[2],
        }

        return self.arima_params, self.garch_params

    def forecast(self, steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.arima_model is None:
            raise ValueError("Fit the ARIMA model before forecasting.")
        if self.garch_model is None:
            raise ValueError("Fit the GARCH model before forecasting.")

        mu_forecast = self.arima_model.forecast(steps)
        _, sigma_forecast = self.garch_model.forecast_variance(steps=steps)
        return mu_forecast, sigma_forecast


if __name__ == "__main__":
    np.random.seed(0)
    n = 100
    eps = np.random.normal(size=n)
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    omega, alpha, beta = 0.1, 0.1, 0.8
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        y[t] = 0.5 * y[t - 1] + eps[t] * np.sqrt(sigma2[t])
    y = np.cumsum(y)  # integrated series with d=1

    # Provide ARIMA orders; GARCH orders will be automatically estimated
    model = ARIMA_GARCH(y, arima_p=1, arima_d=1, arima_q=1)
    arima_params, garch_params = model.fit()
    mu_forecast, sigma_forecast = model.forecast(steps=5)

    print("\nARIMA forecast mean:", mu_forecast)
    print("GARCH forecast volatility:", sigma_forecast)
    print("Estimated GARCH orders:", model.garch_p, model.garch_q)
