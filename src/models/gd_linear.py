import numpy as np


class GDLinearRegression:
    def __init__(self, lr=0.05, n_iter=500, scale=True):
        self.lr = lr
        self.n_iter = n_iter
        self.scale = scale

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        if self.scale:
            X = (X - self.mu_) / self.sd_
        return X

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-12
        Xs = self._prep(X)
        n, d = Xs.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.loss_ = []
        for _ in range(self.n_iter):
            err = Xs @ self.w_ + self.b_ - y
            loss = float(np.mean(err ** 2) / 2)
            self.loss_.append(loss)
            if not np.isfinite(loss) or loss > 1e12:
                self.diverged_ = True
                break
            self.w_ -= self.lr * (Xs.T @ err) / n
            self.b_ -= self.lr * err.mean()
        else:
            self.diverged_ = False
        return self

    def predict(self, X):
        return self._prep(X) @ self.w_ + self.b_
