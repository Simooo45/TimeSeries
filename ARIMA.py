import numpy as np
from scipy.optimize import minimize
from .TimeSeries import TimeSeries
import matplotlib.pyplot as plt


class ARIMA(TimeSeries):
    def __init__(self, series, p, d, q):
        super().__init__(series)
        self.p = p
        self.d = d
        self.q = q
        self.params = {}

    def fit_ARMA(self, series, p, q):
        """Stima ARMA(p, q) con Conditional Least Squares (CLS)."""
        y = series
        n = len(y)
        mean_y = np.mean(y)

        def arma_residuals(params):
            c = params[0]
            phi = params[1 : 1 + p]
            theta = params[1 + p : 1 + p + q]
            eps = np.zeros(n)
            for t in range(max(p, q), n):
                ar_part = np.sum(phi * y[t - np.arange(1, p + 1)]) if p > 0 else 0
                ma_part = np.sum(theta * eps[t - np.arange(1, q + 1)]) if q > 0 else 0
                eps[t] = y[t] - (c + ar_part + ma_part)
            return eps[max(p, q) :]

        def objective(params):
            res = arma_residuals(params)
            return np.sum(res**2)

        init = np.zeros(1 + p + q)
        init[0] = mean_y

        result = minimize(objective, init, method="BFGS")
        params = result.x
        c = params[0]
        phi = params[1 : 1 + p]
        theta = params[1 + p : 1 + p + q]
        eps = arma_residuals(params)

        return {"c": c, "phi": phi, "theta": theta, "sigma2": np.var(eps), "resid": eps}

    def fit(self):
        diff_ts = self.first_difference(self.d)
        y_diff = diff_ts.series

        params = self.fit_ARMA(y_diff, self.p, self.q)
        self.params = params
        return params

    def forecast_in_sample(self):
        """Previsioni ARIMA sui dati esistenti (per residui)."""
        diff_ts = self.first_difference(self.d)
        y_diff = diff_ts.series
        if not {"c", "phi", "theta"} <= set(self.params.keys()):
            raise ValueError('"c", "phi" or "theta" not in self.params.keys()')

        c = self.params["c"]
        phi = self.params["phi"]
        theta = self.params["theta"]
        p, q = len(phi), len(theta)
        n = len(y_diff)
        eps = np.zeros(n)
        y_hat = np.zeros(n)
        for t in range(max(p, q), n):
            ar_part = np.sum(phi * y_diff[t - np.arange(1, p + 1)]) if p > 0 else 0
            ma_part = np.sum(theta * eps[t - np.arange(1, q + 1)]) if q > 0 else 0
            y_hat[t] = c + ar_part + ma_part
            eps[t] = y_diff[t] - y_hat[t]
        return y_hat[max(p, q) :]

    def forecast(self, steps=5):
        """Prevede i prossimi valori di ARIMA(p,d,q)."""
        if self.params is None or len(self.params.items()) == 0:
            raise ValueError("Devi chiamare .fit() prima di prevedere.")

        if not {"c", "phi", "theta"} <= set(self.params.keys()):
            raise ValueError('"c", "phi" or "theta" not in self.params.keys()')

        c = self.params["c"]
        phi = self.params["phi"]
        theta = self.params["theta"]
        p, q = len(phi), len(theta)

        # Serie differenziata (d volte)
        diff_ts = self.first_difference(self.d)
        y_diff = diff_ts.series
        n = len(y_diff)

        eps = np.zeros(n + steps)
        forecasts = []

        # Previsioni sulla serie differenziata
        for h in range(steps):
            t = n + h
            ar_part = np.sum(phi * y_diff[t - np.arange(1, p + 1)]) if t - p >= 0 else 0
            ma_part = np.sum(theta * eps[t - np.arange(1, q + 1)]) if t - q >= 0 else 0
            y_pred = c + ar_part + ma_part
            forecasts.append(y_pred)
            y_diff = np.append(y_diff, y_pred)

        forecasts = np.array(forecasts)

        # "Reintegrazione" se d > 0
        y_orig = self.series.copy()
        if self.d == 1:
            last_val = y_orig[-1]
            forecasts = np.cumsum(forecasts) + last_val
        elif self.d == 2:
            last_two = y_orig[-2:]
            forecasts = (
                np.cumsum(np.cumsum(forecasts))
                + last_two[-1]
                + (last_two[-1] - last_two[-2])
            )

        return forecasts

    def plot_series(self):
        """Plot the original time series."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.series, label="Original series", color="steelblue")
        plt.title("Original Time Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_fitted_vs_actual(self):
        """Plot in-sample fitted values vs actual series (differenced)."""
        diff_ts = self.first_difference(self.d)
        y_true = diff_ts.series[max(self.p, self.q) :]
        y_pred = self.forecast_in_sample()
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual (differenced)", color="black")
        plt.plot(y_pred, label="Fitted (in-sample)", color="red", linestyle="--")
        plt.title("Fitted vs Actual (Differenced Series)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals(self):
        """Plot residuals for model diagnostics."""
        if "resid" not in self.params:
            raise ValueError("Fit the model before plotting residuals.")
        resid = self.params["resid"]
        plt.figure(figsize=(10, 4))
        plt.plot(resid, color="gray")
        plt.title("Model Residuals")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.show()

    def plot_forecast(self, steps=10):
        """Plot the forecast together with the historical series."""
        forecast_values = self.forecast(steps)
        plt.figure(figsize=(10, 4))
        plt.plot(self.series, label="Observed", color="black")
        plt.plot(
            np.arange(len(self.series), len(self.series) + steps),
            forecast_values,
            label="Forecast",
            color="red",
            linestyle="--",
        )
        plt.title(f"ARIMA({self.p},{self.d},{self.q}) Forecast ({steps} steps ahead)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
