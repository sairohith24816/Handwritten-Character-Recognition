
import numpy as np
import pandas as pd
import time
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from data_augmentation import load_augmented_dataset
import joblib

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for KNN"""
    print("Starting hyperparameter tuning for KNN...")
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # Only used when metric='minkowski'
    }
    
    knn = KNeighborsClassifier()
    print(f"Using full dataset with {len(X_train):,} samples for grid search...")
    
    grid_search = GridSearchCV(
        knn, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_

def main():
    print("="*60)
    print("KNN Training with Hyperparameter Tuning and Data Augmentation")
    print("="*60)
    
    # use moderate augmentation
    from data_augmentation import load_augmented_dataset
    
    print("Loading moderate augmented training data...")
    X_train_aug, y_train_aug = load_augmented_dataset('2x_moderate')
    
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

    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    best_params, best_cv_score = hyperparameter_tuning(X_train_scaled, y_train_aug)

    print("Training final KNN model with best parameters...")
    start_time = time.time()

    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(X_train_scaled, y_train_aug)

    training_time = time.time() - start_time

    print("Evaluating model...")
    train_pred = best_knn.predict(X_train_scaled)
    test_pred = best_knn.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train_aug, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best CV score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")

    # Save model and scaler
    joblib.dump(best_knn, '../models/best_knn.joblib')
    joblib.dump(scaler, '../models/knn_scaler.joblib')

    # Save best parameters
    results = {
        'model_name': 'KNN',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'augmentation_factor': 2,
        'train_samples': len(X_train_scaled)
    }

    with open('../models/knn_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nModel saved as '../models/best_knn.joblib'")
    print(f"Scaler saved as '../models/knn_scaler.joblib'")
    print(f"Best parameters saved as '../models/knn_best_params.json'")

if __name__ == "__main__":
    main()
