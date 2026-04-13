import numpy as np
from base_model import Model
from decision_tree import DecisionTree


class RandomForest(Model):
    """
    Random Forest = Bagging + random feature subsets at each node split.
    max_features per node is set to floor(sqrt(n_features)), per the standard.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = []
        self.max_features = None   # set in fit() once we know n_features

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        n_samples, n_features = X_train.shape
        self.max_features = int(np.floor(np.sqrt(n_features)))  # √p rule
        self.estimators = []

        for i in range(self.n_estimators):
            # Bootstrap sample (same as Bagging)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X_train[indices]
            y_sample = y_train[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,   # ← key difference vs Bagging
            )

            # Trees in RF don't need per-tree val tuning;
            # pass val set so DecisionTree.fit() doesn't raise.
            tree.fit(X_sample, y_sample, X_val, y_val)
            self.estimators.append(tree)

    def predict(self, X):
        X = np.array(X)
        # Shape: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators])

        # Majority vote across estimators for each sample
        return np.apply_along_axis(
            lambda col: np.bincount(col.astype(int)).argmax(),
            axis=0,
            arr=all_preds,
        )