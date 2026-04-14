import numpy as np
from base_model import Model
from decision_tree import DecisionTree


class Bagging(Model):

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples = X_train.shape[0]
        self.estimators = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_sample = X_train.iloc[indices]
            y_sample = y_train.iloc[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            tree.fit(x_sample, y_sample, X_val, y_val)
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        predictions = predictions.T

        results = []
        for prediction in predictions:
            values, counts = np.unique(prediction, return_counts=True)
            results.append(values[np.argmax(counts)])

        return np.array(results)