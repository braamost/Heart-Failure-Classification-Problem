import numpy as np
import argparse
from data import prepare_data
from decision_tree import DecisionTree
from bagging import Bagging
from adaboost import AdaBoost
from evaluate import evaluate_model, compare_models, find_most_confusing_classes

def get_available_models():
    """Get dictionary of available models."""
    return {
        'decision_tree': DecisionTree(max_depth=10, min_samples_split=5),
        'bagging': Bagging(n_estimators=10, max_depth=10, min_samples_split=5),
        'adaboost': AdaBoost(n_estimators=50, max_depth=1)
    }

def run_experiment(selected_models=None):
    """
    Main experiment runner that:
    1. Loads and prepares data
    2. Trains selected models (or all if none specified)
    3. Evaluates models on test set
    4. Compares and logs results
    """
    print("Starting Heart Failure Prediction Experiment")
    print("=" * 50)

    # Load and prepare data
    print("1. Loading and preparing data...")
    data = prepare_data('dataset/heart.csv')
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Get available models and select which ones to train
    available_models = get_available_models()

    if selected_models is None or len(selected_models) == 0:
        # Train all models
        models_to_train = available_models
        print("\n2. Training all models...")
    else:
        # Train only selected models
        models_to_train = {}
        for model_name in selected_models:
            if model_name in available_models:
                models_to_train[model_name] = available_models[model_name]
            else:
                print(f"Warning: Model '{model_name}' not found. Available models: {list(available_models.keys())}")

        if not models_to_train:
            print("No valid models selected. Training all models instead.")
            models_to_train = available_models

        print(f"\n2. Training selected models: {list(models_to_train.keys())}...")

    # Train models
    trained_models = []

    for name, model in models_to_train.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train, X_val, y_val)
            trained_models.append((name, model))
            print(f"✓ {name} trained successfully")
        except NotImplementedError:
            print(f"✗ {name} implementation not complete - skipping")
        except Exception as e:
            print(f"✗ Error training {name}: {str(e)}")

    # Evaluate models on test set
    print("\n3. Evaluating models on test set...")
    evaluation_results = []

    for name, model in trained_models:
        try:
            result = evaluate_model(model, X_test, y_test, name.replace('_', ' ').title())
            evaluation_results.append(result)
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")

    # Compare models (only if multiple models were evaluated)
    if len(evaluation_results) > 1:
        print("\n4. Comparing model performance...")
        compare_models(evaluation_results)
        find_most_confusing_classes(evaluation_results)
    elif len(evaluation_results) == 1:
        print(f"\nSingle model evaluation completed for {evaluation_results[0]['model_name']}")
    else:
        print("No models were successfully trained and evaluated.")

    print("\n" + "=" * 50)
    print("Experiment completed!")
    print("=" * 50)

    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heart Failure Prediction Experiment Runner')
    parser.add_argument('--models', nargs='*', choices=['decision_tree', 'bagging', 'adaboost'])

    args = parser.parse_args()

    # Run the main experiment
    selected_models = args.models if args.models else None
    results = run_experiment(selected_models)


# Example usage:
# To run all models:
# python main.py
#
# To run specific models:
# python main.py --models decision_tree bagging