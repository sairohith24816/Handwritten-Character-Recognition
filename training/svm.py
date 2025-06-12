
import numpy as np
import pandas as pd
import time
import json
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from data_augmentation import load_augmented_dataset
import joblib


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for SVM"""
    print("Starting hyperparameter tuning for SVM...")
    
    param_distributions = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4, 5],  
        'coef0': [0.0, 0.1, 0.5, 1.0]  
    }
    
    svm = SVC(probability=True, random_state=42)
    print(f"Using full dataset with {len(X_train):,} samples for hyperparameter tuning...")
    
    random_search = RandomizedSearchCV(
        svm, 
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
    best_C = random_search.best_params_['C']
    best_kernel = random_search.best_params_['kernel']
    
    fine_tune_grid = {
        'C': [best_C/2, best_C, best_C*2],
        'kernel': [best_kernel],
        'gamma': [random_search.best_params_['gamma']]
    }
    
    if best_kernel == 'poly':
        fine_tune_grid['degree'] = [random_search.best_params_['degree']]
        fine_tune_grid['coef0'] = [random_search.best_params_['coef0']]
    elif best_kernel == 'sigmoid':
        fine_tune_grid['coef0'] = [random_search.best_params_['coef0']]
    
    grid_search = GridSearchCV(
        svm,
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
    print("SVM Training with Hyperparameter Tuning and Data Augmentation")
    print("="*70)
    
    # used light augmentation for SVM
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
    
    print("Training final SVM model with best parameters...")
    start_time = time.time()
    
    best_svm = SVC(**best_params, probability=True, random_state=42)
    best_svm.fit(X_train_scaled, y_train_aug)
    
    training_time = time.time() - start_time
    
    print("Evaluating model...")
    train_pred = best_svm.predict(X_train_scaled)
    test_pred = best_svm.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train_aug, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best CV score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    joblib.dump(best_svm, '../models/best_svm.joblib')
    joblib.dump(scaler, '../models/svm_scaler.joblib')
    
    results = {
        'model_name': 'SVM',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'augmentation_factor': 2,
        'train_samples': len(X_train_scaled)
    }
    
    with open('../models/svm_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nModel saved as '../models/best_svm.joblib'")
    print(f"Scaler saved as '../models/svm_scaler.joblib'")
    print(f"Best parameters saved as '../models/svm_best_params.json'")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred))

if __name__ == "__main__":
    main()
