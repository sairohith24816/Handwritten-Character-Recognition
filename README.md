# 🎯 Handwritten Character Recognition

A comprehensive machine learning project for recognizing handwritten digits using the MNIST dataset. This project implements multiple classification algorithms with hyperparameter tuning, data augmentation, and an interactive web interface for real-time digit recognition.

## 🚀 Features

- **6 ML Models**: Naive Bayes, Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting
- **Interactive Web Interface**: Streamlit app with drawable canvas for testing models
- **Hyperparameter Tuning**: Full dataset optimization using GridSearchCV with 5-fold CV
- **Data Augmentation**: Enhanced training with rotation, translation, scaling, and shearing

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install -r requirements.txt
```

### Setup
1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies: `pip install -r requirements.txt`
4. Download MNIST dataset and place CSV files in the `data/` folder

## 🎮 Quick Start

### Option 1: Use the Interactive Web Interface (Recommended)
```bash
streamlit run app.py
```
This opens a web browser with an interactive drawing canvas where you can:
- Draw digits with your mouse/touchscreen
- Get instant predictions from trained models
- See prediction confidence scores
- Test different drawing styles

### Option 2: Train Models from Scratch
```bash
cd training
python data_augmentation.py    # Generate augmented datasets 
python train_all_models.py     # Train all models (1-4 hours depending on system)
```

## 📊 Dataset

This project uses the MNIST handwritten digit dataset:
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28x28 pixels (784 features)
- **Classes**: 10 digits (0-9)

## 🚀 Usage

### 1. Interactive Web Interface (Recommended)
```bash
streamlit run app.py
```
Features:
- **Drawing Canvas**: Draw digits directly in your browser
- **Real-time Prediction**: Instant results as you draw
- **Model Selection**: Choose from different trained models
- **Confidence Scores**: See prediction probabilities
- **Image Processing**: Automatic preprocessing for optimal recognition

### 2. Data Augmentation
```bash
cd training
python data_augmentation.py
```
This creates augmented datasets with different augmentation levels:
- **2x_light**: 120,000 samples with light transformations (for SVM)
- **2x_moderate**: 120,000 samples with moderate transformations (for KNN, Logistic, Random Forest, Gradient Boosting)
- **2x_heavy**: 120,000 samples with heavy transformations (for Naive Bayes)

### 3. Train Individual Models
```bash
cd training

# Train specific models
python naive_bayes.py  (Fastest)
python logistic.py         
python knn.py              
python svm.py             
python random_forest.py    
python gradient_boosting.py 
```

### 4. Train all Models with best parameters at Once
```bash
cd training
python train_all_models.py       
```

### 5. Model Evaluation
All trained models are automatically evaluated with:
- Training accuracy
- Test accuracy  
- 5-fold cross-validation scores
- Classification reports
- Confusion matrices
- Training time metrics

### Loading and Using Saved Models
```python
import joblib
import numpy as np

# Load any trained model
model = joblib.load('models/best_svm.joblib')

# Load scaler if it exists (for models that use scaling)
try:
    scaler = joblib.load('models/svm_scaler.joblib')
    # Preprocess new data
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
except FileNotFoundError:
    # No scaler needed (Random Forest, Gradient Boosting)
    predictions = model.predict(X_new)

# Get prediction probabilities
probabilities = model.predict_proba(X_new)
```

### Quick Prediction Function
```python
def predict_digit(image_array, model_name='svm'):
    """
    Predict digit from 28x28 image array
    model_name: 'naive_bayes', 'logistic_regression', 'knn', 'svm', 
                'random_forest', 'gradient_boosting'
    """
    # Load model
    model = joblib.load(f'models/best_{model_name}.joblib')
    
    # Preprocess image
    image_data = image_array.flatten() / 255.0
    image_data = image_data.reshape(1, -1)
    
    # Apply scaling if needed
    try:
        scaler = joblib.load(f'models/{model_name}_scaler.joblib')
        image_data = scaler.transform(image_data)
    except FileNotFoundError:
        pass  # No scaler needed
    
    # Predict
    prediction = model.predict(image_data)[0]
    confidence = model.predict_proba(image_data)[0]
    
    return prediction, confidence
```

### Model-Specific Optimizations
- **SVM**: Uses light augmentation (2x_light) for faster training while maintaining accuracy
- **Naive Bayes**: Uses heavy augmentation (2x_heavy) for better generalization
- **Tree-based models**: Use moderate augmentation (2x_moderate) for balanced performance
- **Linear models**: Benefit from StandardScaler preprocessing
- **Ensemble methods**: No scaling required, handle raw features well

### Streamlit Interface Features
- **Real-time drawing**: Smooth canvas with configurable stroke width
- **Automatic preprocessing**: Image resizing, centering, and normalization
- **Multi-model prediction**: Switch between different trained models
- **Confidence visualization**: Bar charts showing prediction probabilities
- **Performance metrics**: Display accuracy and training time for each model
- **Export functionality**: Save drawn images for testing

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 🐛 Troubleshooting

### Performance Optimization Tips

#### For Faster Experimentation:
1. **Train individual models**: Start with Naive Bayes (~15 min) and Logistic Regression (~30 min)
2. **Modify augmentation**: Edit factors in `data_augmentation.py` for smaller datasets
3. **Use subset training**: Modify training scripts to use smaller samples during development

#### For Better Accuracy:
1. **Use heavy augmentation**: Generates more diverse training examples
2. **Full hyperparameter search**: Let complete GridSearchCV finish
3. **Ensemble predictions**: Combine multiple model predictions

#### For Memory Optimization:
1. **Train models individually**: Avoid `train_all_models.py` if RAM is limited
2. **Close other applications**: Especially browsers and IDEs during training
3. **Use smaller batch sizes**: Modify GridSearchCV `n_jobs` parameter

### System Requirements

#### Minimum:
- **RAM**: 4GB (for Naive Bayes, Logistic Regression)
- **Storage**: 2GB free space
- **CPU**: Dual-core processor

#### Recommended:
- **RAM**: 8GB+ (for SVM, ensemble methods)
- **Storage**: 5GB free space  
- **CPU**: Quad-core processor
- **GPU**: Not required (scikit-learn uses CPU)

