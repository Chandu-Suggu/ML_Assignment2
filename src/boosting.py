"""
From-scratch implementation of Multiclass AdaBoost (SAMME Algorithm)
as specified by Zhu, Zou, Rosset, and Hastie (2009).
Supports K classes, explicit handling of weak learners not beating random chance,
normalized weighted voting confidence q_k(x) = S_k(x) / sum_t alpha_t,
and complete tracking of sample weight dynamics across boosting rounds.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class SAMMEClassifier:
    """
    Multi-class AdaBoost using the SAMME (Stagewise Additive Modeling using
    a Multi-class Exponential loss) algorithm.
    """

    def __init__(self, max_depth=3, n_estimators=100, n_classes=40, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        self.random_state = random_state

        self.estimators_ = []
        self.estimator_weights_ = []
        self.err_history_ = []
        self.alpha_history_ = []
        self.max_weight_history_ = []
        self.top10_weight_history_ = []
        self.sample_weights_history_ = []
        self.final_weights_ = None

    def fit(self, X, y):
        N = len(X)
        K = self.n_classes
        w = np.ones(N, dtype=np.float64) / N

        self.estimators_ = []
        self.estimator_weights_ = []
        self.err_history_ = []
        self.alpha_history_ = []
        self.max_weight_history_ = []
        self.top10_weight_history_ = []
        self.sample_weights_history_ = []

        random_cutoff = 1.0 - (1.0 / K)

        for t in range(self.n_estimators):
            # Seed each round deterministically
            round_seed = (self.random_state + t * 9973) % (2**31 - 1)
            dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=round_seed)
            dt.fit(X, y, sample_weight=w)
            preds = dt.predict(X)

            mistakes = (preds != y).astype(np.float64)
            err_t = np.sum(w * mistakes)

            # Explicit check: weak learner must be strictly better than random guessing (1 - 1/K)
            if err_t >= random_cutoff:
                # If learner is no better than random guessing, stop boosting early
                break

            # Handle perfect learner (zero error) to avoid division by zero
            if err_t < 1e-12:
                alpha_t = np.log((1.0 - 1e-12) / 1e-12) + np.log(K - 1)
            else:
                alpha_t = np.log((1.0 - err_t) / err_t) + np.log(K - 1)

            # Exponential weight update
            w_new = w * np.exp(alpha_t * mistakes)
            sum_w = np.sum(w_new)
            if sum_w <= 0 or np.isnan(sum_w):
                break
            w = w_new / sum_w

            self.estimators_.append(dt)
            self.estimator_weights_.append(alpha_t)

            self.err_history_.append(float(err_t))
            self.alpha_history_.append(float(alpha_t))
            self.max_weight_history_.append(float(np.max(w)))
            self.top10_weight_history_.append(float(np.sum(np.sort(w)[-10:])))
            self.sample_weights_history_.append(w.copy())

            # If error is zero, dataset is fully shattered; stop
            if err_t < 1e-12:
                break

        self.estimator_weights_ = np.array(self.estimator_weights_, dtype=np.float64)
        self.final_weights_ = w.copy()
        return self

    def predict_scores(self, X, n_rounds=None):
        """
        Computes cumulative weighted score S_k(x) = sum_{t} alpha_t 1[h_t(x) = k]
        Returns shape (N_samples, n_classes).
        """
        n_rounds = len(self.estimators_) if n_rounds is None else min(n_rounds, len(self.estimators_))
        N = len(X)
        K = self.n_classes
        scores = np.zeros((N, K), dtype=np.float64)

        for t in range(n_rounds):
            dt = self.estimators_[t]
            alpha = self.estimator_weights_[t]
            preds = dt.predict(X)
            for i in range(N):
                scores[i, preds[i]] += alpha

        return scores

    def predict_proba(self, X, n_rounds=None):
        """
        Normalized weighted vote vector: q_k(x) = S_k(x) / sum_t alpha_t.
        Returns shape (N_samples, n_classes).
        """
        n_rounds = len(self.estimators_) if n_rounds is None else min(n_rounds, len(self.estimators_))
        scores = self.predict_scores(X, n_rounds=n_rounds)
        total_alpha = np.sum(self.estimator_weights_[:n_rounds])
        if total_alpha <= 0:
            return np.ones((len(X), self.n_classes)) / self.n_classes
        return scores / total_alpha

    def predict(self, X, n_rounds=None):
        """Returns argmax_k S_k(x)."""
        scores = self.predict_scores(X, n_rounds=n_rounds)
        return np.argmax(scores, axis=1)

    def predict_with_confidence(self, X, n_rounds=None):
        """
        Returns:
            preds: (N,) predicted classes
            confidences: (N,) maximum normalized weighted vote max_k q_k(x)
        """
        proba = self.predict_proba(X, n_rounds=n_rounds)
        preds = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        return preds, confidences
