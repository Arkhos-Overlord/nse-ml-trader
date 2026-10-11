"""Self-contained Gaussian HMM (3-state) — no hmmlearn dependency.

Implements Baum-Welch (EM) for a Gaussian-emission HMM with full covariance, plus
Viterbi decode and state posteriors. Same model class as hmmlearn's GaussianHMM,
written in pure NumPy so it runs anywhere without a native build. Seeded and
deterministic; enough for regime detection on OHLCV features.
"""
from __future__ import annotations

import numpy as np


def _log_gaussian(X, mean, cov):
    d = X.shape[1]
    diff = X - mean
    cov = cov + 1e-6 * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.full(len(X), -1e18)
    sol = np.linalg.solve(cov, diff.T).T
    maha = np.einsum("ij,ij->i", diff, sol)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


class GaussianHMM:
    def __init__(self, n_components: int = 3, n_iter: int = 100, tol: float = 1e-4,
                 random_state: int = 42):
        self.k = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None

    def _init_params(self, X):
        n, d = X.shape
        # k-means++-ish init: pick k spread-out points as means
        idx = [self.rng.integers(n)]
        for _ in range(self.k - 1):
            dists = np.min(
                [np.linalg.norm(X - X[i], axis=1) for i in idx], axis=0)
            probs = dists / (dists.sum() + 1e-9)
            idx.append(int(self.rng.choice(n, p=probs)))
        self.means_ = X[idx].copy()
        self.covars_ = np.array([np.cov(X.T) + 1e-3 * np.eye(d) for _ in range(self.k)])
        self.transmat_ = np.full((self.k, self.k), 1.0 / self.k)
        self.startprob_ = np.full(self.k, 1.0 / self.k)

    def _emission(self, X):
        return np.column_stack([_log_gaussian(X, self.means_[j], self.covars_[j])
                                for j in range(self.k)])

    def _forward(self, logB):
        n = len(logB)
        la = np.zeros((n, self.k))
        la[0] = np.log(self.startprob_ + 1e-300) + logB[0]
        trans = np.log(self.transmat_ + 1e-300)
        for t in range(1, n):
            la[t] = _logsumexp(la[t - 1][:, None] + trans, axis=0) + logB[t]
        return la

    def _backward(self, logB):
        n = len(logB)
        lb = np.zeros((n, self.k))
        trans = np.log(self.transmat_ + 1e-300)
        for t in range(n - 2, -1, -1):
            lb[t] = _logsumexp(trans + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
        return lb

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._init_params(X)
        prev = -np.inf
        for _ in range(self.n_iter):
            logB = self._emission(X)
            la = self._forward(logB)
            lb = self._backward(logB)
            ll = _logsumexp(la[-1])
            # E-step: posteriors
            log_post = la + lb
            log_post -= _logsumexp(log_post, axis=1, keepdims=True)
            gamma = np.exp(log_post)
            xi_sum = np.zeros((self.k, self.k))
            trans = np.log(self.transmat_ + 1e-300)
            for t in range(len(X) - 1):
                m = la[t][:, None] + trans + (logB[t + 1] + lb[t + 1])[None, :]
                xi_sum += np.exp(m - _logsumexp(m))
            # M-step
            self.startprob_ = gamma[0] + 1e-300
            self.startprob_ /= self.startprob_.sum()
            self.transmat_ = xi_sum + 1e-300
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            for j in range(self.k):
                w = gamma[:, j]
                sw = w.sum() + 1e-300
                self.means_[j] = (w[:, None] * X).sum(0) / sw
                diff = X - self.means_[j]
                self.covars_[j] = (diff * w[:, None]).T @ diff / sw + 1e-3 * np.eye(X.shape[1])
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict(self, X):
        logB = self._emission(np.asarray(X, dtype=float))
        la = self._forward(logB)
        return np.argmax(la, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        logB = self._emission(X)
        la = self._forward(logB)
        lb = self._backward(logB)
        log_post = la + lb
        log_post -= _logsumexp(log_post, axis=1, keepdims=True)
        return np.exp(log_post)


def _logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300) + m
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out
