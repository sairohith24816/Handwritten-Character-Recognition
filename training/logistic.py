
import numpy as np
import pandas as pd
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from data_augmentation import load_augmented_dataset
import joblib

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Logistic Regression"""
    print("Starting hyperparameter tuning for Logistic Regression...")
    
    param_distributions = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'max_iter': [1000, 2000, 3000],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Only for elasticnet
    }
    
    logreg = LogisticRegression(random_state=42)
    print(f"Using full dataset with {len(X_train):,} samples for hyperparameter tuning...")
    
    random_search = RandomizedSearchCV(
        logreg, 
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
    best_C = random_search.best_params_['C']
    
    fine_tune_grid = {
        'C': [best_C/2, best_C, best_C*2],
        'solver': [random_search.best_params_['solver']],
        'penalty': [random_search.best_params_['penalty']],
        'max_iter': [random_search.best_params_['max_iter']]
    }
    
    if random_search.best_params_['penalty'] == 'elasticnet':
        fine_tune_grid['l1_ratio'] = [random_search.best_params_['l1_ratio']]
    
    grid_search = GridSearchCV(
        logreg,
        fine_tune_grid,
        cv=5,
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
    print("Logistic Regression Training with Hyperparameter Tuning and Data Augmentation")
    print("="*70)
    
    # light augmentation 
    from data_augmentation import load_augmented_dataset
    
    print("Loading light augmented training data...")
    X_train_aug, y_train_aug = load_augmented_dataset('2x_light')
    
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
    
    print("Training final Logistic Regression model with best parameters...")
    start_time = time.time()
    
    best_logreg = LogisticRegression(**best_params, random_state=42)
    best_logreg.fit(X_train_scaled, y_train_aug)
    
    training_time = time.time() - start_time
    
    print("Evaluating model...")
    train_pred = best_logreg.predict(X_train_scaled)
    test_pred = best_logreg.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train_aug, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best CV score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    # Save model
    joblib.dump(best_logreg, '../models/best_logistic_regression.joblib')
    joblib.dump(scaler, '../models/logistic_scaler.joblib')
    
    # Save best parameters
    results = {
        'model_name': 'Logistic Regression',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'augmentation_factor': 2,
        'train_samples': len(X_train_scaled)
    }
    
    with open('../models/logistic_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nModel saved as '../models/best_logistic_regression.joblib'")
    print(f"Scaler saved as '../models/logistic_scaler.joblib'")
    print(f"Best parameters saved as '../models/logistic_best_params.json'")

if __name__ == "__main__":
    main()
