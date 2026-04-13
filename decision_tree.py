import copy
import numpy as np
from base_model import Model
from sklearn.metrics import f1_score


class DecisionTree(Model):
    """
    Decision Tree classifier using Information Gain.
    Hyperparameters are tuned automatically via grid search on the validation set.
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            raise ValueError("Validation set (X_val, y_val) is required for hyperparameter tuning")
        self._tune_hyperparameters(np.array(X_train), np.array(y_train),
                                   np.array(X_val),   np.array(y_val))

    # Tree construction
    def _build_tree(self, X, y, depth, max_depth, min_samples_split):
        """Recursively build the tree. Hyperparameters are explicit arguments
        so grid search remains stateless — no mutation of self required."""
        if depth == max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
            return {'is_leaf': True, 'class': self._majority_class(y)}

        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            return {'is_leaf': True, 'class': self._majority_class(y)}

        left_mask = X[:, feature] <= threshold
        return {
            'is_leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left':  self._build_tree(X[left_mask],  y[left_mask],  depth + 1, max_depth, min_samples_split),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples_split),
        }

    def _extend_tree(self, node, X, y, depth, target_depth, min_samples_split):
        """Extend an existing tree by one depth level.

        Walks to every current leaf, routing the training data along the way.
        Leaves stopped only by the previous max_depth are split; leaves stopped
        by purity or min_samples_split are left untouched. All internal nodes
        (and their splits) are reused without recomputation."""
        if not node['is_leaf']:
            left_mask = X[:, node['feature']] <= node['threshold']
            node['left']  = self._extend_tree(node['left'],  X[left_mask],  y[left_mask],  depth + 1, target_depth, min_samples_split)
            node['right'] = self._extend_tree(node['right'], X[~left_mask], y[~left_mask], depth + 1, target_depth, min_samples_split)
            return node

        if depth < target_depth and len(y) >= min_samples_split and len(np.unique(y)) > 1:
            feature, threshold = self._find_best_split(X, y)
            if feature is not None:
                left_mask = X[:, feature] <= threshold
                return {
                    'is_leaf': False,
                    'feature': feature,
                    'threshold': threshold,
                    'left':  {'is_leaf': True, 'class': self._majority_class(y[left_mask])},
                    'right': {'is_leaf': True, 'class': self._majority_class(y[~left_mask])},
                }
        return node

    # Split search
    def _find_best_split(self, X, y):
        best_ig, best_feature, best_threshold = -1, None, None

        # Random Forest: sample a subset of features at each node
        n_features = X.shape[1]
        if self.max_features is not None:
            features = np.random.choice(n_features,
                                        size=min(self.max_features, n_features),
                                        replace=False)
        else:
            features = range(n_features)

        for feature in features:
            values = X[:, feature]
            unique = np.unique(values)
            if len(unique) < 2:
                continue

            if len(unique) > 20:
                thresholds = np.unique(np.percentile(values, np.linspace(10, 90, 10)))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2

            for threshold in thresholds:
                left_mask = values <= threshold
                if left_mask.all() or (~left_mask).all():
                    continue
                ig = self._information_gain(y, y[left_mask], y[~left_mask])
                if ig > best_ig:
                    best_ig, best_feature, best_threshold = ig, feature, threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left, right):
        n = len(parent)
        weighted_child_entropy = (len(left) / n) * self._entropy(left) + (len(right) / n) * self._entropy(right)
        return self._entropy(parent) - weighted_child_entropy

    def _entropy(self, y):
        p1 = np.sum(y == 1) / len(y)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0
        return -(p0 * np.log2(p0) + p1 * np.log2(p1))

    def _majority_class(self, y):
        return 1 if np.sum(y) > len(y) / 2 else 0

    # Hyperparameter tuning
    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Grid search over max_depth x min_samples_split.

        Loop order is intentional:
        - Outer loop (min_samples_split): controls which nodes attempt to split,
          so trees across different values are structurally incompatible — rebuild.
        - Inner loop (max_depth): a depth-N tree is a true subtree of depth-(N+1),
          so we grow incrementally via _extend_tree, reusing all earlier splits.
        """
        max_depths         = [2, 3, 4, 5, 6, 7, 8]
        min_samples_splits = [2, 3, 5, 10, 15]

        best_val_f1, best_tree = -1, None

        col = (12, 20, 10)
        print(f"\n  {'max_depth':<{col[0]}} {'min_samples_split':<{col[1]}} {'val_f1':<{col[2]}}")
        print("  " + "-" * sum(col))

        for min_samples_split in min_samples_splits:
            tree = self._build_tree(X_train, y_train, depth=0,
                                    max_depth=max_depths[0],
                                    min_samples_split=min_samples_split)

            for max_depth in max_depths:
                tree = self._extend_tree(tree, X_train, y_train,
                                         depth=0, target_depth=max_depth,
                                         min_samples_split=min_samples_split)

                val_f1 = f1_score(y_val, self.predict(X_val, tree=tree), average='binary')
                print(f"  {max_depth:<{col[0]}} {min_samples_split:<{col[1]}} {val_f1:<{col[2]}.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.max_depth, self.min_samples_split = max_depth, min_samples_split
                    best_tree = copy.deepcopy(tree)

            print()

        self.tree = best_tree
        print("  " + "-" * sum(col))
        print(f"  Best: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, val_f1={best_val_f1:.4f}\n")

    # Prediction
    def predict(self, X, tree=None):
        tree = tree if tree is not None else self.tree
        return np.array([self._predict_sample(sample, tree) for sample in np.array(X)])

    def _predict_sample(self, sample, node):
        if node['is_leaf']:
            return node['class']
        child = node['left'] if sample[node['feature']] <= node['threshold'] else node['right']
        return self._predict_sample(sample, child)