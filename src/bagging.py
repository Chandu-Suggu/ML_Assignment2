"""
From-scratch Bagging Ensemble for Multiclass Face Recognition.
Implements bootstrap sampling, out-of-bag (OOB) evaluation,
pairwise tree disagreement, vote margin computation, and prediction confidence.
"""

from collections import Counter
import itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class BaggedDecisionTreeEnsemble:
    """
    Bagged Decision Tree Ensemble built from scratch using bootstrap sampling.
    """

    def __init__(self, base_tree_params=None, n_estimators=100, random_state=42):
        self.base_tree_params = base_tree_params or {"max_depth": None, "min_samples_leaf": 2}
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.distinct_sample_counts_ = []

    def fit(self, X, y):
        N = len(X)
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.distinct_sample_counts_ = []

        for b in range(self.n_estimators):
            boot_idx = rng.choice(N, size=N, replace=True)
            self.bootstrap_indices_.append(boot_idx)
            self.distinct_sample_counts_.append(len(np.unique(boot_idx)))

            # Seed each tree deterministically
            tree_seed = (self.random_state + b * 10007) % (2**31 - 1)
            dt = DecisionTreeClassifier(**self.base_tree_params, random_state=tree_seed)
            dt.fit(X[boot_idx], y[boot_idx])
            self.estimators_.append(dt)

        return self

    def predict_raw_matrix(self, X, n_trees=None):
        """Returns predictions matrix of shape (n_trees, N_samples)."""
        trees = self.estimators_[:n_trees] if n_trees is not None else self.estimators_
        return np.array([tree.predict(X) for tree in trees])

    def predict(self, X, n_trees=None):
        """Predicts class via majority voting across the ensemble."""
        raw_preds = self.predict_raw_matrix(X, n_trees=n_trees)
        B, N = raw_preds.shape
        maj_preds = np.zeros(N, dtype=int)
        for i in range(N):
            counts = Counter(raw_preds[:, i])
            maj_preds[i] = counts.most_common(1)[0][0]
        return maj_preds

    def predict_proba(self, X, n_trees=None, n_classes=40):
        """Returns empirical vote fraction distribution over classes of shape (N, n_classes)."""
        raw_preds = self.predict_raw_matrix(X, n_trees=n_trees)
        B, N = raw_preds.shape
        proba = np.zeros((N, n_classes))
        for i in range(N):
            for p in raw_preds[:, i]:
                proba[i, p] += 1.0
        return proba / B

    def predict_with_confidence(self, X, n_trees=None):
        """
        Returns:
            predictions: (N,) majority predicted classes
            confidences: (N,) winning vote fraction
            vote_counters: list of Counter objects
            margins: (N,) winning vote fraction minus runner-up vote fraction
        """
        raw_preds = self.predict_raw_matrix(X, n_trees=n_trees)
        B, N = raw_preds.shape
        preds = np.zeros(N, dtype=int)
        confidences = np.zeros(N)
        margins = np.zeros(N)
        vote_counters = []

        for i in range(N):
            counts = Counter(raw_preds[:, i])
            vote_counters.append(counts)
            top = counts.most_common(2)
            top1_class, top1_count = top[0]
            preds[i] = top1_class
            confidences[i] = top1_count / B

            if len(top) > 1:
                top2_count = top[1][1]
                margins[i] = (top1_count - top2_count) / B
            else:
                margins[i] = 1.0

        return preds, confidences, vote_counters, margins

    def compute_oob_accuracy(self, X_train, y_train, n_trees=None, n_classes=40):
        """
        Computes out-of-bag (OOB) accuracy on the training data.
        Only estimators that did not include sample i in their bootstrap sample vote for sample i.
        """
        B = n_trees if n_trees is not None else len(self.estimators_)
        N = len(X_train)
        oob_votes = np.zeros((N, n_classes))
        oob_counts = np.zeros(N)

        for b_idx in range(B):
            in_bag = set(self.bootstrap_indices_[b_idx])
            oob_indices = [i for i in range(N) if i not in in_bag]
            if not oob_indices:
                continue

            preds = self.estimators_[b_idx].predict(X_train[oob_indices])
            for idx, p in zip(oob_indices, preds):
                oob_votes[idx, p] += 1
                oob_counts[idx] += 1

        valid_mask = oob_counts > 0
        if not np.any(valid_mask):
            return 0.0

        oob_preds = np.argmax(oob_votes[valid_mask], axis=1)
        return float(accuracy_score(y_train[valid_mask], oob_preds))

    def compute_pairwise_disagreement(self, X, n_trees=None):
        """
        Computes average pairwise disagreement between distinct tree pairs on X:
        (1 / C(B, 2)) sum_{j < k} (1 / N) sum_{i=1}^N 1[h_j(x_i) != h_k(x_i)]
        Vectorized using class count frequencies:
        agreeing_pairs = sum_k C(c_k, 2), disagreement = 1 - agreeing_pairs / C(B, 2).
        """
        B = n_trees if n_trees is not None else len(self.estimators_)
        if B <= 1:
            return 0.0

        raw_preds = self.predict_raw_matrix(X, n_trees=B)  # shape (B, N)
        N = raw_preds.shape[1]
        total_pairs = B * (B - 1) / 2.0

        sample_disagreements = []
        for i in range(N):
            counts = Counter(raw_preds[:, i])
            agreeing_pairs = sum(c * (c - 1) / 2.0 for c in counts.values())
            disagree_fraction = 1.0 - (agreeing_pairs / total_pairs)
            sample_disagreements.append(disagree_fraction)

        return float(np.mean(sample_disagreements))

    def compute_mean_vote_margin(self, X, n_trees=None):
        """
        Computes mean vote margin: winning vote fraction minus runner-up vote fraction.
        """
        _, _, _, margins = self.predict_with_confidence(X, n_trees=n_trees)
        return float(np.mean(margins))
