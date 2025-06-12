
import numpy as np
import pandas as pd
import time
import json
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_augmentation import load_augmented_dataset
import joblib
import warnings
warnings.filterwarnings('ignore')

models_dir = '../models'
results = []
trained_models = {}
X_test = None
y_test = None

def initialize():
    """Initialize global variables and load test data"""
    global X_test, y_test
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading test data...")
    df_test = pd.read_csv('../data/mnist_test.csv')
    X_test = df_test.iloc[:, 1:].values / 255.0
    y_test = df_test.iloc[:, 0].values
    print(f"Test data shape: {X_test.shape}")

def load_best_params(model_name):
    """Load best parameters for a specific model"""
    params_file = f'{models_dir}/{model_name}_best_params.json'
    try:
        with open(params_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {params_file} not found. Using default parameters.")
        return None

def train_svm():
    """Train SVM with best parameters"""
    print("\n" + "="*60)
    print("Training SVM")
    print("="*60)
    
    X_train, y_train = load_augmented_dataset('2x_light')
    if X_train is None:
        print("Augmented data not found. Skipping SVM.")
        return
    
    best_params_data = load_best_params('svm')
    if best_params_data:
        best_params = best_params_data['best_params']
    else:
        best_params = {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    model = SVC(**best_params, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    joblib.dump(scaler, f'{models_dir}/svm_scaler.joblib')
    
    results.append({
        'model_name': 'SVM',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'best_params': best_params
    })
    
    trained_models['SVM'] = {'model': model, 'scaler': scaler}
    
    print(f"SVM - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}, Time: {training_time:.2f}s")

def train_random_forest():
    """Train Random Forest with best parameters"""
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    X_train, y_train = load_augmented_dataset('2x_heavy')
    if X_train is None:
        print("Augmented data not found. Skipping Random Forest.")
        return
    
    best_params_data = load_best_params('random_forest')
    if best_params_data:
        best_params = best_params_data['best_params']
    else:
        best_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
    
    start_time = time.time()
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    joblib.dump(model, f'{models_dir}/best_random_forest.joblib')
    
    results.append({
        'model_name': 'Random Forest',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'best_params': best_params
    })
    
    trained_models['Random Forest'] = {'model': model, 'scaler': None}
    
    print(f"Random Forest - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}, Time: {training_time:.2f}s")

def train_naive_bayes():
    """Train Naive Bayes with best parameters"""
    print("\n" + "="*60)
    print("Training Naive Bayes")
    print("="*60)
    
    X_train, y_train = load_augmented_dataset('2x_moderate')
    if X_train is None:
        print("Augmented data not found. Skipping Naive Bayes.")
        return
    
    best_params_data = load_best_params('naive_bayes')
    if best_params_data:
        best_params = best_params_data['best_params']
        model_type = best_params_data.get('model_type', 'Gaussian NB')
    else:
        best_params = {'var_smoothing': 1e-9}
        model_type = 'Gaussian NB'
    
    scaler = None
    if model_type == 'Gaussian NB':
        model = GaussianNB(**best_params)
        scaler = StandardScaler()
    elif model_type == 'Multinomial NB':
        model = MultinomialNB(**best_params)
        scaler = MinMaxScaler()
    else: 
        model = BernoulliNB(**best_params)
    
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    if scaler:
        joblib.dump(scaler, f'{models_dir}/naive_bayes_scaler.joblib')
    
    results.append({
        'model_name': f'Naive Bayes ({model_type})',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'best_params': best_params
    })
    
    trained_models['Naive Bayes'] = {'model': model, 'scaler': scaler}
    
    print(f"Naive Bayes ({model_type}) - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}, Time: {training_time:.2f}s")

def train_logistic_regression():
    """Train Logistic Regression with best parameters"""
    print("\n" + "="*60)
    print("Training Logistic Regression")
    print("="*60)
    
    X_train, y_train = load_augmented_dataset('2x_light')
    if X_train is None:
        print("Augmented data not found. Skipping Logistic Regression.")
        return
    
    best_params_data = load_best_params('logistic')
    if best_params_data:
        best_params = best_params_data['best_params']
    else:
        best_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    model = LogisticRegression(**best_params, random_state=42)
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    joblib.dump(scaler, f'{models_dir}/logistic_regression_scaler.joblib')
    
    results.append({
        'model_name': 'Logistic Regression',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'best_params': best_params
    })
    
    trained_models['Logistic Regression'] = {'model': model, 'scaler': scaler}
    
    print(f"Logistic Regression - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}, Time: {training_time:.2f}s")

def train_knn():
    """Train KNN with best parameters"""
    print("\n" + "="*60)
    print("Training K-Nearest Neighbors")
    print("="*60)
    
    X_train, y_train = load_augmented_dataset('2x_moderate')
    if X_train is None:
        print("Augmented data not found. Skipping KNN.")
        return
    
    best_params_data = load_best_params('knn')
    if best_params_data:
        best_params = best_params_data['best_params']
    else:
        best_params = {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    model = KNeighborsClassifier(**best_params)
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    joblib.dump(scaler, f'{models_dir}/knn_scaler.joblib')
    
    results.append({
        'model_name': 'K-Nearest Neighbors',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'best_params': best_params
    })
    
    trained_models['KNN'] = {'model': model, 'scaler': scaler}
    
    print(f"KNN - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}, Time: {training_time:.2f}s")

def train_gradient_boosting():
    """Train Gradient Boosting with best parameters"""
    print("\n" + "="*60)
    print("Training Gradient Boosting")
    print("="*60)
    
    X_train, y_train = load_augmented_dataset('2x_heavy')
    if X_train is None:
        print("Augmented data not found. Skipping Gradient Boosting.")
        return
    
    best_params_data = load_best_params('gradient_boosting')
    if best_params_data:
        best_params = best_params_data['best_params']
    else:
        best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
    
    start_time = time.time()
    model = GradientBoostingClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    joblib.dump(model, f'{models_dir}/best_gradient_boosting.joblib')
    
    results.append({
        'model_name': 'Gradient Boosting',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'best_params': best_params
    })
    
    trained_models['Gradient Boosting'] = {'model': model, 'scaler': None}
    
    print(f"Gradient Boosting - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}, Time: {training_time:.2f}s")

def train_all_models():
    """Train all models with their best parameters"""
    print("="*80)
    print("TRAINING ALL MODELS WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    start_time = time.time()
    
    train_svm()
    train_random_forest()
    train_naive_bayes()
    train_logistic_regression()
    train_knn()
    train_gradient_boosting()
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ALL MODELS TRAINED IN {total_time:.2f} SECONDS")
    print("="*80)
    
    return total_time

def save_results():
    """Save comprehensive results to CSV and JSON"""
    if not results:
        print("No results to save.")
        return None, None
    
    df_results = pd.DataFrame(results)
    
    df_results['overfitting'] = df_results['train_accuracy'] - df_results['test_accuracy']
    df_results['efficiency'] = df_results['test_accuracy'] / df_results['training_time']
    
    df_results = df_results.sort_values('test_accuracy', ascending=False)
    
    csv_path = f'{models_dir}/model_comparison_results.csv'
    df_results.to_csv(csv_path, index=False)
    
    detailed_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': results,
        'summary': {
            'best_model': df_results.iloc[0]['model_name'],
            'best_test_accuracy': df_results.iloc[0]['test_accuracy'],
            'total_models': len(df_results),
            'avg_test_accuracy': df_results['test_accuracy'].mean(),
            'avg_training_time': df_results['training_time'].mean()
        }
    }
    
    json_path = f'{models_dir}/model_comparison_detailed.json'
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"\nResults saved:")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    
    return df_results, detailed_results

def print_summary():
    """Print a summary of all model results"""
    if not results:
        print("No results to summarize.")
        return
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_accuracy', ascending=False)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Rank':<4} {'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'Time (s)':<10} {'Samples':<10}")
    print("-"*100)
    
    for i, (_, row) in enumerate(df_results.iterrows()):
        print(f"{i+1:<4} {row['model_name']:<20} {row['train_accuracy']:<10.4f} "
              f"{row['test_accuracy']:<10.4f} {row['training_time']:<10.2f} {row['train_samples']:<10,}")
    
    print("-"*100)
    print(f"Best Model: {df_results.iloc[0]['model_name']} (Test Accuracy: {df_results.iloc[0]['test_accuracy']:.4f})")
    print(f"Average Test Accuracy: {df_results['test_accuracy'].mean():.4f}")
    print(f"Total Training Time: {df_results['training_time'].sum():.2f} seconds")

def main():
    """Main function to train all models and save results"""
    initialize()
    
    total_training_time = train_all_models()
    
    df_results, detailed_results = save_results()
    
    print_summary()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Total time: {total_training_time:.2f} seconds")
    print(f"All models and results saved in: {models_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
