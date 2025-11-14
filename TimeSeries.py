import numpy as np
import matplotlib.pyplot as plt


class TimeSeries:
    """Generic time series container with basic analysis tools."""

    def __init__(self, series: np.ndarray):
        self.series = series

    def first_difference(self, d: int):
        """Apply d-th order differencing (supports d=0,1,2)."""
        if d == 0:
            return TimeSeries(self.series)
        elif d == 1:
            series = self.series[1:] - self.series[:-1]
            return TimeSeries(series)
        elif d == 2:
            series = self.series[2:] - 2 * self.series[1:-1] + self.series[:-2]
            return TimeSeries(series)
        else:
            raise NotImplementedError("This method supports only d = 0, 1, or 2.")

    def autocorrelation(self, k: int) -> float:
        """Compute autocorrelation at lag k."""
        y = self.series
        y_mean = np.mean(y)
        N = len(y)
        if k >= N:
            return np.nan
        num = float(np.sum((y[k:] - y_mean) * (y[:-k] - y_mean)))
        den = float(np.sum((y - y_mean) ** 2))
        return num / den

    def compute_acf(self, max_lag: int) -> np.ndarray:
        """Compute ACF up to max_lag."""
        acf_vals = [1.0]
        for k in range(1, max_lag + 1):
            acf_vals.append(self.autocorrelation(k))
        return np.array(acf_vals)

    def compute_pacf(self, max_lag: int) -> np.ndarray:
        """Compute PACF using Durbin-Levinson algorithm."""
        acf = self.compute_acf(max_lag)
        pacf = np.zeros(max_lag + 1)
        phi = np.zeros((max_lag + 1, max_lag + 1))
        pacf[0] = 1

        for k in range(1, max_lag + 1):
            if k == 1:
                phi[k, k] = acf[1]
            else:
                num = acf[k] - np.sum(phi[k - 1, 1:k] * acf[k - 1 : 0 : -1])
                den = 1 - np.sum(phi[k - 1, 1:k] * acf[1:k])
                phi[k, k] = num / den
                for j in range(1, k):
                    phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            pacf[k] = phi[k, k]

        return pacf

    def plot_ACF(self, max_lag: int = 20):
        """Plot ACF with confidence interval."""
        acf_vals = self.compute_acf(max_lag)
        lags = np.arange(0, max_lag + 1)
        plt.figure(figsize=(8, 4))
        plt.bar(lags, acf_vals, width=0.3, color="skyblue", edgecolor="k")
        plt.axhline(y=0, color="black", linewidth=0.8)
        conf = 1.96 / np.sqrt(len(self.series))
        plt.axhline(y=conf, color="red", linestyle="--", linewidth=1)
        plt.axhline(y=-conf, color="red", linestyle="--", linewidth=1)
        plt.title("Autocorrelation Function (ACF)")
        plt.xlabel("Lag")
        plt.ylabel("ρ(k)")
        plt.show()

    def plot_PACF(self, max_lag: int = 20):
        """Plot PACF with confidence interval."""
        pacf_vals = self.compute_pacf(max_lag)
        lags = np.arange(0, max_lag + 1)
        plt.figure(figsize=(8, 4))
        plt.bar(lags, pacf_vals, width=0.3, color="lightgreen", edgecolor="k")
        plt.axhline(y=0, color="black", linewidth=0.8)
        conf = 1.96 / np.sqrt(len(self.series))
        plt.axhline(y=conf, color="red", linestyle="--", linewidth=1)
        plt.axhline(y=-conf, color="red", linestyle="--", linewidth=1)
        plt.title("Partial Autocorrelation Function (PACF)")
        plt.xlabel("Lag")
        plt.ylabel("φ(k,k)")
        plt.show()

    def cutoff(self, vals, conf, max_lag):
        """Find first lag where value falls below confidence interval."""
        for k in range(1, len(vals)):
            if abs(vals[k]) < conf:
                return k - 1
        return max_lag

    def estimate_arma_p_q(self, max_lag: int = 20):
        """Suggest AR and MA orders based on ACF and PACF of the series."""
        N = len(self.series)
        conf = 1.96 / np.sqrt(N)
        acf = self.compute_acf(max_lag)
        pacf = self.compute_pacf(max_lag)
        q = self.cutoff(acf, conf, max_lag)
        p = self.cutoff(pacf, conf, max_lag)
        return p, q


if __name__ == "__main__":
    # Example AR(2) simulation
    np.random.seed(0)
    n = 200
    eps = np.random.normal(size=n)
    series = np.zeros(n)
    for t in range(2, n):
        series[t] = 0.6 * series[t - 1] - 0.3 * series[t - 2] + eps[t]

    ts = TimeSeries(series)
    ts.plot_ACF(max_lag=20)
    ts.plot_PACF(max_lag=20)
    p, q = ts.estimate_arma_p_q(20)
    print("Suggested ARMA orders:", p, q)
