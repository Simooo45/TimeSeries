import numpy as np
from itertools import product
from ARIMA import ARIMA
from GARCH import GARCH
from ARIMA_GARCH import ARIMA_GARCH


class TimeSeriesModel:
    def __init__(self, series):
        self.series = series
        self.best_arima_params = None
        self.best_garch_params = None
        self.best_arima_garch_params = None

    # ===============================
    # Selezione ordine ARIMA automatico
    # ===============================
    def select_arima(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        best_aic = np.inf
        best_params = None
        for p, d, q in product(
            range(p_range[0], p_range[1] + 1),
            range(d_range[0], d_range[1] + 1),
            range(q_range[0], q_range[1] + 1),
        ):
            if p == 0 and d == 0 and q == 0:
                continue
            model = ARIMA(self.series, p, d, q)
            params = model.fit()
            # Calcolo residui allineando correttamente la lunghezza
            y_pred = model._forecast_in_sample()
            y_true = model.series[model.d + max(model.p, model.q) :]
            resid = y_true - y_pred

            # Calcolo AIC: 2k + n*log(RSS/n)
            n = len(resid)
            k = p + q + 1  # numero parametri stimati
            RSS = np.sum(resid**2)
            aic = 2 * k + n * np.log(RSS / n)

            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)
        self.best_arima_params = best_params
        print(
            f"ARIMA ottimale: p={best_params[0]}, d={best_params[1]}, q={best_params[2]}, AIC={best_aic:.2f}"
        )
        return best_params

    # ===============================
    # Stima GARCH(1,1)
    # ===============================
    def fit_garch(self):
        garch_model = GARCH(self.series)
        params = garch_model.fit()
        self.best_garch_params = params
        return params

    # ===============================
    # Stima ARIMA-GARCH
    # ===============================
    def fit_arima_garch(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        best_aic = np.inf
        best_params = None
        for p, d, q in product(
            range(p_range[0], p_range[1] + 1),
            range(d_range[0], d_range[1] + 1),
            range(q_range[0], q_range[1] + 1),
        ):
            if p == 0 and d == 0 and q == 0:
                continue
            try:
                model = ARIMA_GARCH(self.series, p, d, q)
                arima_params, garch_params = model.fit()
                resid = model.resid_
                n = len(resid)
                k = p + q + 1 + 3  # ARIMA params + GARCH params
                RSS = np.sum(resid**2)
                aic = 2 * k + n * np.log(RSS / n)
                if aic < best_aic:
                    best_aic = aic
                    best_params = {
                        "p": p,
                        "d": d,
                        "q": q,
                        "arima": arima_params,
                        "garch": garch_params,
                    }
            except:
                continue
        self.best_arima_garch_params = best_params
        print(
            f"ARIMA-GARCH ottimale: p={best_params['p']}, d={best_params['d']}, q={best_params['q']}, AIC={best_aic:.2f}"
        )
        return best_params


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
    y = np.cumsum(y)  # integrazione per d>0

    ts_model = TimeSeriesModel(y)
    ts_model.select_arima(p_range=(0, 2), d_range=(0, 1), q_range=(0, 2))
    garch_params = ts_model.fit_garch()
    arima_garch_params = ts_model.fit_arima_garch(
        p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)
    )
