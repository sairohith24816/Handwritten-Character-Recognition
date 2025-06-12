import numpy as np
import pandas as pd
import os
from scipy.ndimage import rotate, shift
import time

AUGMENTATION_CONFIGS = {
    'light': {'angle': 5, 'shift': 1.0, 'noise': 0.002},
    'moderate': {'angle': 8, 'shift': 1.5, 'noise': 0.005},
    'heavy': {'angle': 12, 'shift': 2.0, 'noise': 0.01}
}

def augment_single_image(img_flat, intensity='moderate'):
    """Augment a single flattened image"""
    config = AUGMENTATION_CONFIGS.get(intensity, AUGMENTATION_CONFIGS['moderate'])
    img_2d = img_flat.reshape(28, 28)
    
    # Random rotation
    angle = np.random.uniform(-config['angle'], config['angle'])
    rotated = rotate(img_2d, angle, reshape=False, cval=0)
    
    # Random shift
    shift_x = np.random.uniform(-config['shift'], config['shift'])
    shift_y = np.random.uniform(-config['shift'], config['shift'])
    shifted = shift(rotated, [shift_x, shift_y], cval=0)
    
    # Add noise
    noise = np.random.normal(0, config['noise'], shifted.shape)
    noisy = np.clip(shifted + noise, 0, 1)
    
    return noisy.flatten()

def augment_data(X, y, augmentation_factor=2, intensity='moderate'):
    """Apply data augmentation to training data"""
    if augmentation_factor <= 1:
        return X, y
        
    print(f"Applying {intensity} augmentation (factor: {augmentation_factor})...")
    start_time = time.time()
    
    augmented_X = [X]
    augmented_y = [y]
    
    for i in range(augmentation_factor - 1):
        print(f"Generating batch {i+1}/{augmentation_factor-1}...")
        X_aug = np.array([augment_single_image(img, intensity) for img in X])
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
    
    final_X = np.vstack(augmented_X)
    final_y = np.hstack(augmented_y)
    
    print(f"Augmentation completed in {time.time() - start_time:.1f}s")
    print(f"Shape: {X.shape} -> {final_X.shape}")
    
    return final_X, final_y

def prepare_all_datasets(augmentation_factor=2):
    """Prepare augmented datasets with specified factor for all intensity levels"""
    print("="*60)
    print(f"Preparing Augmented Datasets (Factor: {augmentation_factor})")
    print("="*60)
    
    aug_data_dir = '../data/augmented'
    os.makedirs(aug_data_dir, exist_ok=True)

    print("Loading MNIST training data...")
    df_train = pd.read_csv('../data/mnist_train.csv')
    X_train = df_train.iloc[:, 1:].values / 255.0
    y_train = df_train.iloc[:, 0].values
    print(f"Original data shape: {X_train.shape}")
    
    print("Using full dataset - no subsets taken")
    
    if augmentation_factor == 1:
        datasets = [(len(X_train), 1, 'moderate', "original")]
    else:
        datasets = [
            (len(X_train), augmentation_factor, 'light', f"{augmentation_factor}x_light"),
            (len(X_train), augmentation_factor, 'moderate', f"{augmentation_factor}x_moderate"),
            (len(X_train), augmentation_factor, 'heavy', f"{augmentation_factor}x_heavy"),
        ]
    
    total_start = time.time()
    
    for subset_size, aug_factor, intensity, name in datasets:
        print(f"\n--- Creating {name} ---")
        
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
        
        if aug_factor == 1:
            X_final, y_final = X_subset, y_subset
        else:
            X_final, y_final = augment_data(X_subset, y_subset, aug_factor, intensity)
          # Save dataset
        filepath = os.path.join(aug_data_dir, f'mnist_{name}.npz')
        np.savez_compressed(filepath, X=X_final, y=y_final)
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[OK] Saved: mnist_{name}.npz ({size_mb:.1f} MB)")
    
    print(f"\n{'='*60}")
    print(f"Datasets ready in {time.time() - total_start:.1f}s")
    if augmentation_factor == 1:
        print("Available dataset: original")
    else:
        print(f"Available datasets ({augmentation_factor}x): {augmentation_factor}x_light, {augmentation_factor}x_moderate, {augmentation_factor}x_heavy")

def load_augmented_dataset(dataset_name):
    """Load a pre-computed augmented dataset"""
    filepath = f'../data/augmented/mnist_{dataset_name}.npz'
    try:
        data = np.load(filepath)
        return data['X'], data['y']
    except FileNotFoundError:
        print(f"Dataset not found: {filepath}")
        print("Run prepare_all_datasets() first")
        return None, None

def get_augmentation_config(intensity='moderate'):
    """Get augmentation configuration for specific intensity"""
    return AUGMENTATION_CONFIGS.get(intensity, AUGMENTATION_CONFIGS['moderate'])


if __name__ == "__main__":
    import sys
    
    augmentation_factor = 2
    if len(sys.argv) > 1:
        try:
            augmentation_factor = int(sys.argv[1])
        except ValueError:
            print("Invalid augmentation factor. Using default value of 2.")
    
    prepare_all_datasets(augmentation_factor)
