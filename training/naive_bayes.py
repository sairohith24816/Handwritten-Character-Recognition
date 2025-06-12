
import numpy as np
import pandas as pd
import time
import json
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Naive Bayes"""
    print("Starting hyperparameter tuning for Naive Bayes...")
    
    models_to_test = [
        {
            'name': 'Gaussian NB',
            'model': GaussianNB(),
            'params': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
            'scaler': StandardScaler()
        },
        {
            'name': 'Multinomial NB',
            'model': MultinomialNB(),
            'params': {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
            'scaler': MinMaxScaler()  # Ensures non-negative values
        },
        {
            'name': 'Bernoulli NB',
            'model': BernoulliNB(),
            'params': {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0], 'binarize': [0.0, 0.5, 0.8]},
            'scaler': None  # No scaling needed
        }
    ]
    
    best_model = None
    best_score = 0
    best_params = None
    best_model_name = None
    best_scaler = None
    print(f"Using full dataset with {len(X_train):,} samples for hyperparameter tuning...")
    
    for model_config in models_to_test:
        print(f"\nTesting {model_config['name']}...")
        
        if model_config['scaler'] is not None:
            scaler = model_config['scaler']
            X_scaled = scaler.fit_transform(X_train)
        else:
            X_scaled = X_train
            scaler = None
        
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1        )
        
        start_time = time.time()
        grid_search.fit(X_scaled, y_train)
        tuning_time = time.time() - start_time
        
        print(f"{model_config['name']} tuning completed in {tuning_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_model_name = model_config['name']
            best_scaler = scaler
    
    print(f"\nBest Naive Bayes variant: {best_model_name}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    return best_model, best_params, best_score, best_model_name, best_scaler

def main():
    print("="*70)
    print("Naive Bayes Training with Hyperparameter Tuning and Data Augmentation")
    print("="*70)

    # moderate augmentation 
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
    
    best_model, best_params, best_cv_score, best_model_name, best_scaler = hyperparameter_tuning(X_train_aug, y_train_aug)
    
    print(f"Training final {best_model_name} model with best parameters...")
    start_time = time.time()
    
    if best_scaler is not None:
        X_train_final = best_scaler.fit_transform(X_train_aug)
        X_test_final = best_scaler.transform(X_test)
    else:
        X_train_final = X_train_aug
        X_test_final = X_test
    
    best_model.fit(X_train_final, y_train_aug)
    
    training_time = time.time() - start_time
    
    print("Evaluating model...")
    train_pred = best_model.predict(X_train_final)
    test_pred = best_model.predict(X_test_final)
    
    train_accuracy = accuracy_score(y_train_aug, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"Best model type: {best_model_name}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best CV score: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    joblib.dump(best_model, '../models/best_naive_bayes.joblib')
    if best_scaler is not None:
        joblib.dump(best_scaler, '../models/naive_bayes_scaler.joblib')
    
    results = {
        'model_name': f'Naive Bayes ({best_model_name})',
        'model_type': best_model_name,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'augmentation_factor': 3,
        'train_samples': len(X_train_final),
        'uses_scaler': best_scaler is not None
    }
    
    with open('../models/naive_bayes_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nModel saved as '../models/best_naive_bayes.joblib'")
    if best_scaler is not None:
        print(f"Scaler saved as '../models/naive_bayes_scaler.joblib'")
    print(f"Best parameters saved as '../models/naive_bayes_best_params.json'")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    print(f"\nModel Information:")
    print(f"Number of features: {best_model.n_features_in_}")
    print(f"Number of classes: {len(best_model.classes_)}")
    print(f"Classes: {best_model.classes_}")
    
    if hasattr(best_model, 'class_prior_'):
        print(f"Class priors: {best_model.class_prior_}")
    if hasattr(best_model, 'var_'):
        print(f"Feature variances (first 10 features): {best_model.var_[0][:10]}")

if __name__ == "__main__":
    main()
