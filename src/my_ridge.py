import numpy as np

#https://www.kaggle.com/datasets/hamidmieh/superconductivity-data

class MyRidge:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.intercept_ = None
        self.X_mean_ = None
        self.y_mean_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D (muestras, características); se recibió {X.ndim}D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X tiene {X.shape[0]} filas y y tiene {y.shape[0]} valores; deben coincidir."
            )
        if self.alpha < 0:
            raise ValueError(f"alpha no puede ser negativo; se recibió {self.alpha}.")

        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()

        Xc = X - self.X_mean_
        yc = y - self.y_mean_

        n_caracteristicas = X.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_caracteristicas)
        b = Xc.T @ yc

        self.coef_ = np.linalg.solve(A, b)

        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a 'fit' primero.")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X tiene {X.shape[1]} características, pero el modelo se entrenó "
                f"con {self.coef_.shape[0]}."
            )

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        predicciones = self.predict(X)

        ss_res = np.sum((y - predicciones) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1.0 - ss_res / ss_tot

    def rmse(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        residuales = self.predict(X) - y
        return float(np.sqrt(np.mean(residuales ** 2)))
