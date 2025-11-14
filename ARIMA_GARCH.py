import numpy as np
from typing import Tuple

from .TimeSeries import TimeSeries
from .ARIMA import ARIMA
from .GARCH import GARCH


class ARIMA_GARCH(TimeSeries):
    def __init__(self, series, p, d, q):
        super().__init__(series)
        self.p = p
        self.d = d
        self.q = q
        self.arima_model = None
        self.garch_model = None
        self.resid = None
        self.arima_params = {}
        self.garch_params = {}

    # ===============================
    # Fit ARIMA + GARCH
    # ===============================
    def fit(self):
        # 1. Fit ARIMA
        self.arima_model = ARIMA(self.series, self.p, self.d, self.q)
        self.arima_params = self.arima_model.fit()

        # 2. Compute ARIMA residuals
        mu_hat = self.arima_model.forecast_in_sample()
        self.resid = self.series[self.d :] - mu_hat

        # 3. Fit GARCH(1,1) on residuals
        self.garch_model = GARCH(self.resid)
        self.garch_params = self.garch_model.fit()

        return self.arima_params, self.garch_params

    def forecast(self, steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast using ARIMA (mean) and GARCH (volatility).

        Args:
            steps (int, optional): Number of forecast steps. Defaults to 5.

        Raises:
            ValueError: Raised when ARIMA or GARCH models are not fitted.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean forecast and volatility forecast.
        """

        # Check ARIMA model and forecast mean
        if self.arima_model is not None:
            mu_forecast = self.arima_model.forecast(steps)
        else:
            raise ValueError("Fit the ARIMA model before forecasting.")

        # Check GARCH model and forecast variance
        if self.garch_model is not None:
            _, sigma_forecast = self.garch_model.forecast_variance(steps=steps)
        else:
            raise ValueError("Fit the GARCH model before forecasting.")

        return mu_forecast, sigma_forecast


# ===============================
# Example usage
# ===============================
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
    y = np.cumsum(y)  # integrated series, d=1

    model = ARIMA_GARCH(y, p=1, d=1, q=1)
    arima_params, garch_params = model.fit()
    mu_forecast, sigma_forecast = model.forecast(steps=5)

    print("\nARIMA forecast mean:", mu_forecast)
    print("GARCH forecast volatility:", sigma_forecast)
