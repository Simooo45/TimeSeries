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
        print(sigma_forecast)
        return mu_forecast, sigma_forecast
