
import numpy as np
import pandas as pd
import time
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data_augmentation import load_augmented_dataset
import joblib


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Gradient Boosting"""
    print("Starting hyperparameter tuning for Gradient Boosting...")
    
    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    gb = GradientBoostingClassifier(random_state=42)
    
    print(f"Using full dataset with {len(X_train):,} samples for hyperparameter tuning...")
    
    random_search = RandomizedSearchCV(
        gb, 
        param_distributions, 
        n_iter=30,
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    print("Performing fine-tuning with GridSearchCV...")
    best_n_estimators = random_search.best_params_['n_estimators']
    best_learning_rate = random_search.best_params_['learning_rate']
    
    fine_tune_grid = {
        'n_estimators': [max(50, best_n_estimators-25), best_n_estimators, best_n_estimators+25],
        'learning_rate': [best_learning_rate],
        'max_depth': [random_search.best_params_['max_depth']],
        'min_samples_split': [random_search.best_params_['min_samples_split']],
        'min_samples_leaf': [random_search.best_params_['min_samples_leaf']],
        'subsample': [random_search.best_params_['subsample']],
        'max_features': [random_search.best_params_['max_features']]
    }
    
    grid_search = GridSearchCV(
        gb,
        fine_tune_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Fine-tuned best parameters: {grid_search.best_params_}")
    print(f"Fine-tuned best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_

def main():
    print("="*70)
    print("Gradient Boosting Training with Hyperparameter Tuning and Data Augmentation")
    print("="*70)
    
    # use heavy augmentation
    from data_augmentation import load_augmented_dataset
    
    print("Loading heavy augmented training data...")
    X_train_aug, y_train_aug = load_augmented_dataset('2x_heavy')
    
    if X_train_aug is None:
        print("Augmented data not found. Please run data_augmentation.py first.")
        return
    
    print("Loading test data...")
    df_test = pd.read_csv('../data/mnist_test.csv')
    X_test = df_test.iloc[:, 1:].values / 255.0
    y_test = df_test.iloc[:, 0].values
    
    print(f"Training data shape: {X_train_aug.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Using full augmented dataset with {len(X_train_aug):,} samples")
    
    best_params, best_cv_score = hyperparameter_tuning(X_train_aug, y_train_aug)
    
    print("Training final Gradient Boosting model with best parameters...")
    start_time = time.time()
    
    best_gb = GradientBoostingClassifier(**best_params, random_state=42, verbose=1)
    best_gb.fit(X_train_aug, y_train_aug)
    
    training_time = time.time() - start_time
    
    print("Evaluating model...")
    train_pred = best_gb.predict(X_train_aug)
    test_pred = best_gb.predict(X_test)
    
    train_accuracy = accuracy_score(y_train_aug, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best CV score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    # Save model
    joblib.dump(best_gb, '../models/best_gradient_boosting.joblib')
    
    # Save best parameters
    results = {
        'model_name': 'Gradient Boosting',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'augmentation_factor': 2,
        'train_samples': len(X_train_aug)
    }
    
    with open('../models/gradient_boosting_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nModel saved as '../models/best_gradient_boosting.joblib'")
    print(f"Best parameters saved as '../models/gradient_boosting_best_params.json'")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    print(f"\nTop 20 Most Important Pixels:")
    feature_importance = best_gb.feature_importances_
    top_features = np.argsort(feature_importance)[-20:][::-1]
    for i, feature_idx in enumerate(top_features):
        row = feature_idx // 28
        col = feature_idx % 28
        print(f"{i+1:2d}. Pixel at position ({row:2d}, {col:2d}): {feature_importance[feature_idx]:.6f}")
    
    print(f"\nTraining Loss Progress:")
    print(f"Initial Loss: {best_gb.train_score_[0]:.4f}")
    print(f"Final Loss: {best_gb.train_score_[-1]:.4f}")
    print(f"Loss Reduction: {(best_gb.train_score_[0] - best_gb.train_score_[-1]):.4f}")

if __name__ == "__main__":
    main()
