import numpy as np
from base_model import Model
from decision_tree import DecisionTree

class AdaBoost(Model):
    """
    AdaBoost ensemble classifier using decision stumps as weak learners.
    """

    def __init__(self, n_estimators=50, max_depth=1):
        """
        Initialize the AdaBoost ensemble.

        Args:
            n_estimators: Number of boosting rounds (weak learners)
            max_depth: Maximum depth for weak learners (default 1 for stumps)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        self.estimator_weights = []

    def _fit_stump(self, X, y, sample_weights):
        """
        Fit one decision stump on (X, y) respecting sample_weights.

        Args:
            X:              np.ndarray (n_samples, n_features)
            y:              np.ndarray of {0, 1} labels (n_samples,)
            sample_weights: np.ndarray of non-negative weights (n_samples,)

        Returns:
            Trained DecisionTree stump.
        """
        n_samples = X.shape[0]

        # Normalize weights to a probability distribution
        probs = sample_weights / sample_weights.sum()

        # Weighted bootstrap: sample indices according to weight distribution
        indices = np.random.choice(n_samples, size=n_samples, replace=True, p=probs)
        X_boot = X[indices]
        y_boot = y[indices]

        # Build the stump directly (bypass grid-search tuning by calling the
        # internal _build_tree method and storing the result ourselves).
        stump = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=2,
        )
        stump.tree = stump._build_tree(
            X_boot, y_boot,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=2,
        )
        return stump

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the AdaBoost ensemble on the training data.
        1. Initialize uniform sample weights.
        2. For t = 1 … T:
           a. Fit a weak learner h_t on the weighted training set.
           b. Compute weighted error  ε_t.
           c. Compute learner weight  α_t = 0.5 * ln((1-ε_t) / ε_t).
           d. Update & re-normalize sample weights, upweighting
              mis-classified examples.
        3. Optionally use X_val / y_val to select the best number of
           estimators (early stopping on validation F1).

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix for hyperparameter tuning
            y_val: Optional validation target vector for hyperparameter tuning
        """
        from sklearn.metrics import f1_score

        X = np.array(X_train)
        y = np.array(y_train)
        n_samples = X.shape[0]

        # Map labels to {-1, +1} for standard AdaBoost maths
        y_ada = np.where(y == 1, 1, -1).astype(float)

        if X_val is not None and y_val is not None:
            X_v = np.array(X_val)
            y_v = np.array(y_val)
        else:
            X_v = y_v = None

        # Initialize uniform weights
        weights = np.full(n_samples, 1.0 / n_samples)

        self.estimators = []
        self.estimator_weights = []

        # For optional val-based early stopping
        best_val_f1 = -1.0
        best_estimators = []
        best_alphas = []

        print(f"\n  AdaBoost training ({self.n_estimators} rounds, max_depth={self.max_depth})")
        col = (8, 12, 12, 14)
        print(f"  {'Round':<{col[0]}} {'epsilon':<{col[1]}} {'alpha':<{col[2]}} "
              f"{'val_f1' if X_v is not None else '':<{col[3]}}")
        print("  " + "-" * sum(col))

        for t in range(self.n_estimators):
            # ── Step 1: fit weak learner ──────────────────────────────
            stump = self._fit_stump(X, y, weights)

            # ── Step 2: weighted error ────────────────────────────────
            y_pred_01 = stump.predict(X)  # {0, 1}
            y_pred = np.where(y_pred_01 == 1, 1, -1).astype(float)

            incorrect = (y_pred != y_ada).astype(float)
            epsilon = np.dot(weights, incorrect)  # weighted error

            # Guard against degenerate stumps
            epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)

            # ── Step 3: learner weight ────────────────────────────────
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)

            # ── Step 4: update sample weights ────────────────────────
            weights = weights * np.exp(-alpha * y_ada * y_pred)
            weights /= weights.sum()  # re-normalise

            self.estimators.append(stump)
            self.estimator_weights.append(alpha)

            # ── Optional: evaluate on validation set ──────────────────
            val_f1_str = ""
            if X_v is not None:
                val_preds = self.predict(X_v)
                val_f1 = f1_score(y_v, val_preds, average='binary', zero_division=0)
                val_f1_str = f"{val_f1:.4f}"

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_estimators = list(self.estimators)
                    best_alphas = list(self.estimator_weights)

            print(f"  {t + 1:<{col[0]}} {epsilon:<{col[1]}.4f} {alpha:<{col[2]}.4f} "
                  f"{val_f1_str:<{col[3]}}")

        print("  " + "-" * sum(col))

        # If validation was provided, keep the round count with best val F1
        if X_v is not None and best_estimators:
            self.estimators = best_estimators
            self.estimator_weights = best_alphas
            print(f"  Best val_f1={best_val_f1:.4f} "
                  f"at {len(self.estimators)} estimators\n")
        else:
            print(f"  Training complete – {len(self.estimators)} estimators\n")

    def predict(self, X):
        """
        Make predictions using the AdaBoost ensemble (weighted majority vote).

        The ensemble score for sample x is:
            H(x) = sign( Σ_t  α_t * h_t(x) )
        where h_t(x) ∈ {-1, +1}.

        Args:
            X: Feature matrix for prediction

        Returns:
            np.ndarray of predicted class labels in {0, 1}.
        """
        X_arr = np.array(X)

        # Accumulate weighted votes in {-1, +1} space
        ensemble_score = np.zeros(X_arr.shape[0])

        for stump, alpha in zip(self.estimators, self.estimator_weights):
            y_pred_01 = stump.predict(X_arr)  # {0, 1}
            y_pred = np.where(y_pred_01 == 1, 1, -1).astype(float)
            ensemble_score += alpha * y_pred

        # Map sign back to {0, 1}
        return (ensemble_score >= 0).astype(int)