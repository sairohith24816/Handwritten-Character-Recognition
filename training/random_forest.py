
import numpy as np
import pandas as pd
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data_augmentation import load_augmented_dataset
import joblib


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    print("Starting hyperparameter tuning for Random Forest...")
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None, 0.5],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    print(f"Using full dataset with {len(X_train):,} samples for hyperparameter tuning...")
    
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions, 
        n_iter=50,
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
    best_max_depth = random_search.best_params_['max_depth']
    
    fine_tune_grid = {
        'n_estimators': [max(50, best_n_estimators-50), best_n_estimators, best_n_estimators+50],
        'max_depth': [best_max_depth] if best_max_depth else [None],
        'min_samples_split': [random_search.best_params_['min_samples_split']],
        'min_samples_leaf': [random_search.best_params_['min_samples_leaf']],
        'max_features': [random_search.best_params_['max_features']],
        'bootstrap': [random_search.best_params_['bootstrap']],
        'criterion': [random_search.best_params_['criterion']]
    }
    
    if best_max_depth is not None:
        fine_tune_grid['max_depth'] = [max(10, best_max_depth-10), best_max_depth, best_max_depth+10]
    
    grid_search = GridSearchCV(
        rf,
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
    print("Random Forest Training with Hyperparameter Tuning and Data Augmentation")
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
    
    print("Training final Random Forest model with best parameters...")
    start_time = time.time()
    
    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train_aug, y_train_aug)
    
    training_time = time.time() - start_time
    
    print("Evaluating model...")
    train_pred = best_rf.predict(X_train_aug)
    test_pred = best_rf.predict(X_test)
    
    train_accuracy = accuracy_score(y_train_aug, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best CV score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    joblib.dump(best_rf, '../models/best_random_forest.joblib')
    
    results = {
        'model_name': 'Random Forest',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'augmentation_factor': 2,
        'train_samples': len(X_train_aug)
    }
    
    with open('../models/random_forest_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nModel saved as '../models/best_random_forest.joblib'")
    print(f"Best parameters saved as '../models/random_forest_best_params.json'")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    print(f"\nTop 20 Most Important Pixels:")
    feature_importance = best_rf.feature_importances_
    top_features = np.argsort(feature_importance)[-20:][::-1]
    for i, feature_idx in enumerate(top_features):
        row = feature_idx // 28
        col = feature_idx % 28
        print(f"{i+1:2d}. Pixel at position ({row:2d}, {col:2d}): {feature_importance[feature_idx]:.6f}")

if __name__ == "__main__":
    main()
