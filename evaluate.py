import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: Accuracy score
    """
    return accuracy_score(y_true, y_pred)

def compute_f1_score(y_true, y_pred, average='binary'):
    """
    Compute F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'macro', 'micro', 'weighted')

    Returns:
        float: F1 score
    """
    return f1_score(y_true, y_pred, average=average)

def compute_confusion_matrix(y_true, y_pred, normalize=None):
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', None)

    Returns:
        numpy.ndarray: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    """
    Plot confusion matrix using seaborn.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model on test data.

    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = compute_accuracy(y_test, y_pred)
    f1 = compute_f1_score(y_test, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n{model_name} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, title=f'{model_name} Confusion Matrix')

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

def compare_models(results_list):
    """
    Compare multiple models based on their evaluation results.

    Args:
        results_list: List of result dictionaries from evaluate_model
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)

    # Create summary table
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 40)

    for result in results_list:
        print(f"{result['model_name']:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f}")

    # Find best models
    best_accuracy = max(results_list, key=lambda x: x['accuracy'])
    best_f1 = max(results_list, key=lambda x: x['f1_score'])

    print(f"\nBest Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"Best F1-Score: {best_f1['model_name']} ({best_f1['f1_score']:.4f})")

    # Analyze confusion matrices for most confusing classes
    print("\nConfusion Matrix Analysis:")
    for result in results_list:
        cm = result['confusion_matrix']
        print(f"\n{result['model_name']}:")
        print(f"True Positives: {cm[1,1]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Negatives: {cm[0,0]}")

        # Calculate error rates
        total = np.sum(cm)
        error_rate = (cm[0,1] + cm[1,0]) / total
        print(f"Error Rate: {error_rate:.4f}")

def find_most_confusing_classes(results_list):
    """
    Analyze confusion matrices to find the most confusing class pairs.

    Args:
        results_list: List of result dictionaries
    """
    print("\n" + "="*50)
    print("MOST CONFUSING CLASSES ANALYSIS")
    print("="*50)

    for result in results_list:
        cm = result['confusion_matrix']
        model_name = result['model_name']

        print(f"\n{model_name}:")
        if cm.shape[0] == 2:  # Binary classification
            fp = cm[0, 1]  # False Positives (predicted 1, actual 0)
            fn = cm[1, 0]  # False Negatives (predicted 0, actual 1)

            print(f"Class 0 misclassified as Class 1: {fp} times")
            print(f"Class 1 misclassified as Class 0: {fn} times")

            if fp > fn:
                print("More likely to misclassify Class 0 as Class 1")
            elif fn > fp:
                print("More likely to misclassify Class 1 as Class 0")
            else:
                print("Equal misclassification rates")
        else:
            # For multi-class, find the most confused pairs
            max_confusion = 0
            confused_pair = None
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j and cm[i, j] > max_confusion:
                        max_confusion = cm[i, j]
                        confused_pair = (i, j)
            if confused_pair:
                print(f"Most confusing: Class {confused_pair[0]} misclassified as Class {confused_pair[1]} ({max_confusion} times)")