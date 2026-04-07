import numpy as np
from base_model import Model

class DecisionTree(Model):
    """
    Decision Tree classifier implementation using Information Gain.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize the Decision Tree.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the decision tree on the training data.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix for hyperparameter tuning
            y_val: Optional validation target vector for hyperparameter tuning
        """
        # TODO: Implement decision tree training using Information Gain
        # Use X_val and y_val for hyperparameter tuning if provided
        pass

    def predict(self, X):
        """
        Make predictions using the trained decision tree.

        Args:
            X: Feature matrix for prediction

        Returns:
            Predicted class labels
        """
        # TODO: Implement prediction traversal through the tree
        pass