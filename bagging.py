import numpy as np
from base_model import Model
from decision_tree import DecisionTree

class Bagging(Model):
    """
    Bagging ensemble classifier using Decision Trees as base learners.
    """

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        """
        Initialize the Bagging ensemble.

        Args:
            n_estimators: Number of base learners (decision trees)
            max_depth: Maximum depth for each decision tree
            min_samples_split: Minimum samples to split for each tree
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the bagging ensemble on the training data.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix for hyperparameter tuning
            y_val: Optional validation target vector for hyperparameter tuning
        """
        # TODO: Implement bagging by training multiple decision trees
        # on bootstrap samples and aggregating predictions
        # Use X_val and y_val for hyperparameter tuning if provided
        pass

    def predict(self, X):
        """
        Make predictions using the bagging ensemble.

        Args:
            X: Feature matrix for prediction

        Returns:
            Predicted class labels (majority vote)
        """
        # TODO: Implement prediction by aggregating votes from all estimators
        pass