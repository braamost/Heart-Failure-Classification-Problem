from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """
    Abstract base class for machine learning models.
    All models must implement fit and predict methods.
    """

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on the given data.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix for hyperparameter tuning
            y_val: Optional validation target vector for hyperparameter tuning
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on the given data.

        Args:
            X: Feature matrix (numpy array or pandas DataFrame)

        Returns:
            Predictions (numpy array)
        """
        pass