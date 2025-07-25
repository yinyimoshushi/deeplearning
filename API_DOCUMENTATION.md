# Deep Learning Algorithms - API Documentation

This repository contains implementations of various deep learning and machine learning algorithms. This document provides comprehensive documentation for all public APIs, functions, and components.

## Table of Contents

1. [K-Nearest Neighbors (KNN) Module](#k-nearest-neighbors-knn-module)
2. [HarDNet-MSEG Medical Image Segmentation](#hardnet-mseg-medical-image-segmentation)
3. [Dependencies and Requirements](#dependencies-and-requirements)
4. [Installation Guide](#installation-guide)
5. [Usage Examples](#usage-examples)

---

## K-Nearest Neighbors (KNN) Module

### File: `knn.py`

This module implements a K-Nearest Neighbors classifier for predicting user check-in locations based on geographical and temporal features.

### Public Functions

#### `knncls()`

**Description**: Implements a complete K-nearest neighbors workflow for predicting user check-in locations using geographical coordinates and temporal features.

**Parameters**: None

**Returns**: None

**Functionality**:
- Loads and preprocesses location data from CSV
- Performs feature engineering on temporal data
- Applies data filtering and cleaning
- Implements standardization for numerical features
- Performs hyperparameter tuning using GridSearchCV
- Evaluates model performance

**Algorithm Workflow**:
1. **Data Loading**: Reads training data from `train.csv`
2. **Data Filtering**: Filters data to specific geographical bounds (x: 1.0-1.25, y: 2.5-2.75)
3. **Feature Engineering**: 
   - Converts Unix timestamps to datetime features
   - Extracts day, hour, and weekday features
   - Removes original timestamp column
4. **Data Cleaning**: Removes locations with fewer than 4 check-ins
5. **Data Splitting**: Creates train/test split (75%/25%)
6. **Feature Scaling**: Applies StandardScaler normalization
7. **Model Training**: Uses GridSearchCV for hyperparameter optimization
8. **Evaluation**: Reports accuracy scores and best parameters

**Usage Example**:
```python
# Ensure you have train.csv in the same directory
from knn import knncls

# Run the complete KNN pipeline
knncls()
```

**Expected Data Format**:
The function expects a CSV file named `train.csv` with the following columns:
- `x`: Longitude coordinate (float)
- `y`: Latitude coordinate (float)
- `time`: Unix timestamp (int)
- `place_id`: Target location identifier (int/string)
- `row_id`: Unique identifier for each record (int)

**Output**:
```
预测的准确率: 0.85
在测试集上准确率： 0.87
在交叉验证当中最好的结果： 0.83
选择最好的模型是： KNeighborsClassifier(n_neighbors=5)
每个超参数每次交叉验证的结果： {...}
```

### Dependencies Used

| Library | Purpose |
|---------|---------|
| `sklearn.datasets` | Data loading utilities |
| `sklearn.model_selection` | Train/test splitting and cross-validation |
| `sklearn.neighbors` | KNN classifier implementation |
| `sklearn.preprocessing` | Feature scaling and normalization |
| `sklearn.feature_extraction` | Text and feature vectorization |
| `sklearn.naive_bayes` | Naive Bayes classifier |
| `sklearn.metrics` | Model evaluation metrics |
| `sklearn.tree` | Decision tree algorithms |
| `sklearn.ensemble` | Ensemble methods |
| `pandas` | Data manipulation and analysis |

---

## HarDNet-MSEG Medical Image Segmentation

### File: `day01/HarDNet_MSEG.ipynb`

This Jupyter notebook implements a medical image segmentation pipeline using HarDNet-MSEG architecture for polyp segmentation in endoscopic images.

### Overview

**Purpose**: Implements state-of-the-art medical image segmentation using HarDNet (Harmonic DenseNet) architecture specifically designed for multi-scale segmentation tasks.

**Application Domain**: Medical imaging, specifically endoscopic polyp segmentation using the Kvasir-SEG dataset.

### Key Components

#### 1. Environment Setup
```python
# Clone the HarDNet-MSEG repository
!git clone https://github.com/james128333/HarDNet-MSEG

# Install required packages
!pip install thop torchstat
```

#### 2. Data Preparation
- **Dataset**: Kvasir-SEG (Kvasir Segmentation Dataset)
- **Training Set**: 880 images with corresponding masks
- **Validation Set**: 120 images with corresponding masks
- **Format**: PNG images for both input and ground truth masks

#### 3. Model Architecture
- **Base Architecture**: HarDNet (Harmonic DenseNet)
- **Task**: Multi-scale segmentation (MSEG)
- **Pre-trained Weights**: Uses `hardnet85.pth` as backbone

#### 4. Key Features
- **Multi-scale Processing**: Handles objects at different scales
- **Efficient Architecture**: Optimized for memory and computational efficiency
- **Medical Domain Optimization**: Specifically designed for medical image analysis

### Usage Instructions

#### Prerequisites
```bash
# Ensure you have the following files:
# - hardnet85.pth (pre-trained weights)
# - Kvasir_SEG_Training_880.zip (training dataset)
# - Kvasir_SEG_Validation_120.zip (validation dataset)
```

#### Step-by-Step Execution

1. **Setup Environment**:
```python
import torch
print(f"CUDA Device: {torch.cuda.get_device_name()}")
```

2. **Clone Repository**:
```python
!git clone https://github.com/james128333/HarDNet-MSEG
%cd HarDNet-MSEG/
```

3. **Install Dependencies**:
```python
!pip install thop torchstat
```

4. **Prepare Data**:
```python
# Copy pre-trained weights
!cp /path/to/hardnet85.pth ./lib/

# Copy and extract datasets
!cp /path/to/Kvasir_SEG_Training_880.zip ./data/
!cp /path/to/Kvasir_SEG_Validation_120.zip ./data/
!unzip -d ./data ./data/Kvasir_SEG_Training_880.zip
!unzip -d ./data ./data/Kvasir_SEG_Validation_120.zip
```

5. **Run Training/Inference**:
```python
# The notebook continues with model training and evaluation
# Specific commands depend on the HarDNet-MSEG implementation
```

### Expected Results

- **Task**: Polyp segmentation in endoscopic images
- **Metrics**: Typically evaluated using Dice coefficient, IoU, and pixel accuracy
- **Performance**: State-of-the-art results on medical segmentation benchmarks

### Data Structure

```
HarDNet-MSEG/
├── data/
│   ├── Kvasir_SEG_Training_880/
│   │   ├── image/          # Training images
│   │   └── mask/           # Ground truth masks
│   └── Kvasir_SEG_Validation_120/
│       ├── images/         # Validation images
│       └── masks/          # Validation masks
├── lib/
│   └── hardnet85.pth       # Pre-trained weights
└── ...
```

---

## Dependencies and Requirements

### Core Dependencies

#### Python Libraries
```
torch>=1.7.0           # Deep learning framework
torchvision>=0.8.0      # Computer vision utilities
scikit-learn>=0.24.0    # Machine learning library
pandas>=1.2.0           # Data manipulation
numpy>=1.19.0           # Numerical computing
```

#### Additional Packages
```
thop                    # PyTorch operation profiler
torchstat              # Model statistics
jupyter                # Notebook environment
matplotlib             # Plotting (implied)
```

### System Requirements
- **Python**: 3.7 or higher
- **CUDA**: Compatible GPU for deep learning (recommended)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large datasets
- **Storage**: Sufficient space for datasets (Kvasir-SEG ~500MB)

---

## Installation Guide

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yinyimoshushi/deeplearning.git
cd deeplearning
```

2. **Install Python dependencies**:
```bash
pip install torch torchvision scikit-learn pandas numpy thop torchstat jupyter
```

3. **Prepare data for KNN**:
```bash
# Place your train.csv file in the root directory
# Ensure it contains columns: x, y, time, place_id, row_id
```

4. **For HarDNet-MSEG setup**:
```bash
cd day01/
jupyter notebook HarDNet_MSEG.ipynb
# Follow the notebook cells for complete setup
```

### Development Setup

For development and experimentation:

```bash
# Create virtual environment
python -m venv deeplearning_env
source deeplearning_env/bin/activate  # Linux/Mac
# or
deeplearning_env\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt       # If available
pip install jupyter matplotlib seaborn plotly  # Additional visualization tools
```

---

## Usage Examples

### Example 1: Basic KNN Location Prediction

```python
"""
Basic usage of KNN for location prediction
"""
from knn import knncls
import pandas as pd

# Ensure your data file exists and has the correct format
# Example data structure:
data_example = {
    'x': [1.1, 1.15, 1.2, 1.05],
    'y': [2.6, 2.65, 2.7, 2.55], 
    'time': [1591776000, 1591779600, 1591783200, 1591786800],
    'place_id': ['loc_1', 'loc_2', 'loc_1', 'loc_3'],
    'row_id': [1, 2, 3, 4]
}

# Save as CSV (if creating test data)
df = pd.DataFrame(data_example)
df.to_csv('train.csv', index=False)

# Run the KNN pipeline
knncls()
```

### Example 2: Custom KNN Implementation

```python
"""
Custom implementation using the same approach as knncls()
but with more control over parameters
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def custom_knn_predictor(data_path, x_range=(1.0, 1.25), y_range=(2.5, 2.75), 
                        test_size=0.25, k_values=[3, 5, 10], cv_folds=2):
    """
    Custom KNN predictor with configurable parameters
    
    Args:
        data_path (str): Path to CSV data file
        x_range (tuple): Longitude range for filtering
        y_range (tuple): Latitude range for filtering  
        test_size (float): Proportion of data for testing
        k_values (list): K values to test in grid search
        cv_folds (int): Number of cross-validation folds
    
    Returns:
        dict: Results including accuracy, best parameters, and model
    """
    
    # Load and filter data
    data = pd.read_csv(data_path)
    data = data.query(f"x > {x_range[0]} & x < {x_range[1]} & "
                     f"y > {y_range[0]} & y < {y_range[1]}")
    
    # Feature engineering
    time_value = pd.to_datetime(data['time'], unit='s')
    time_value = pd.DatetimeIndex(time_value)
    
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    data = data.drop(['time'], axis=1)
    
    # Remove locations with insufficient data
    place_count = data.groupby('place_id').count()
    valid_places = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(valid_places.place_id)]
    
    # Prepare features and targets
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Grid search
    knn = KNeighborsClassifier()
    param_grid = {"n_neighbors": k_values}
    grid_search = GridSearchCV(knn, param_grid=param_grid, cv=cv_folds)
    
    grid_search.fit(x_train_scaled, y_train)
    
    # Results
    results = {
        'test_accuracy': grid_search.score(x_test_scaled, y_test),
        'best_cv_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }
    
    return results

# Usage
results = custom_knn_predictor('train.csv')
print(f"Test Accuracy: {results['test_accuracy']:.3f}")
print(f"Best Parameters: {results['best_params']}")
```

### Example 3: Medical Image Segmentation Setup

```python
"""
Setting up HarDNet-MSEG for medical image segmentation
"""

# In Jupyter notebook environment:
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Check GPU availability
if torch.cuda.is_available():
    device = torch.cuda.get_device_name()
    print(f"GPU available: {device}")
else:
    print("Using CPU")

# Example of loading and preprocessing medical images
def load_medical_image(image_path, mask_path=None):
    """
    Load and preprocess medical images for segmentation
    
    Args:
        image_path (str): Path to input image
        mask_path (str): Path to ground truth mask (optional)
    
    Returns:
        tuple: Preprocessed image and mask (if provided)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Load mask if provided
    if mask_path:
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8)  # Binary threshold
        return image, mask
    
    return image

# Example usage (after setting up the environment)
# image, mask = load_medical_image('path/to/image.png', 'path/to/mask.png')
```

### Example 4: Data Validation and Preprocessing

```python
"""
Data validation utilities for both KNN and segmentation tasks
"""

import pandas as pd
import numpy as np
from pathlib import Path

def validate_knn_data(csv_path):
    """
    Validate KNN input data format and quality
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        dict: Validation results and statistics
    """
    try:
        data = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['x', 'y', 'time', 'place_id', 'row_id']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return {'valid': False, 'error': f'Missing columns: {missing_cols}'}
        
        # Check data types and ranges
        validation_results = {
            'valid': True,
            'total_records': len(data),
            'unique_places': data['place_id'].nunique(),
            'x_range': (data['x'].min(), data['x'].max()),
            'y_range': (data['y'].min(), data['y'].max()),
            'time_range': (data['time'].min(), data['time'].max()),
            'missing_values': data.isnull().sum().to_dict(),
            'place_distribution': data['place_id'].value_counts().head(10).to_dict()
        }
        
        return validation_results
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def validate_segmentation_data(data_dir):
    """
    Validate medical image segmentation dataset
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        dict: Validation results
    """
    data_path = Path(data_dir)
    
    # Check directory structure
    required_dirs = ['image', 'mask']  # or 'images', 'masks'
    available_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    validation_results = {
        'valid': True,
        'data_directory': str(data_path),
        'available_directories': available_dirs,
        'errors': []
    }
    
    # Find image and mask directories (flexible naming)
    image_dir = None
    mask_dir = None
    
    for dir_name in available_dirs:
        if 'image' in dir_name.lower():
            image_dir = data_path / dir_name
        elif 'mask' in dir_name.lower():
            mask_dir = data_path / dir_name
    
    if not image_dir or not mask_dir:
        validation_results['valid'] = False
        validation_results['errors'].append('Could not find image and mask directories')
        return validation_results
    
    # Count files
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
    
    validation_results.update({
        'image_count': len(image_files),
        'mask_count': len(mask_files),
        'image_directory': str(image_dir),
        'mask_directory': str(mask_dir)
    })
    
    # Check for matching files
    image_names = {f.stem for f in image_files}
    mask_names = {f.stem for f in mask_files}
    
    unmatched_images = image_names - mask_names
    unmatched_masks = mask_names - image_names
    
    if unmatched_images or unmatched_masks:
        validation_results['errors'].append({
            'unmatched_images': list(unmatched_images)[:10],  # Show first 10
            'unmatched_masks': list(unmatched_masks)[:10]
        })
    
    validation_results['matched_pairs'] = len(image_names & mask_names)
    
    return validation_results

# Usage examples
print("Validating KNN data:")
knn_validation = validate_knn_data('train.csv')
print(knn_validation)

print("\nValidating segmentation data:")
seg_validation = validate_segmentation_data('./data/Kvasir_SEG_Training_880/')
print(seg_validation)
```

---

## Error Handling and Troubleshooting

### Common Issues and Solutions

#### KNN Module Issues

1. **FileNotFoundError: train.csv not found**
   ```python
   # Solution: Ensure CSV file exists in the correct location
   import os
   if not os.path.exists('train.csv'):
       print("train.csv not found. Please provide the data file.")
   ```

2. **Empty dataset after filtering**
   ```python
   # Solution: Adjust filtering parameters
   # Modify the query ranges in knncls() function
   data = data.query("x > 0.5 & x < 1.5 & y > 2.0 & y < 3.0")  # Wider range
   ```

3. **Insufficient data for places**
   ```python
   # Solution: Reduce minimum place count requirement
   tf = place_count[place_count.row_id > 2].reset_index()  # Reduce from 3 to 2
   ```

#### HarDNet-MSEG Issues

1. **CUDA out of memory**
   ```python
   # Solution: Reduce batch size or use CPU
   torch.cuda.empty_cache()
   # Or modify batch size in training script
   ```

2. **Missing dataset files**
   ```bash
   # Solution: Verify file paths and download datasets
   ls -la ./data/
   # Ensure Kvasir datasets are properly extracted
   ```

3. **Package compatibility issues**
   ```bash
   # Solution: Use specific package versions
   pip install torch==1.9.0 torchvision==0.10.0
   ```

### Performance Optimization Tips

#### For KNN:
- Use smaller geographical ranges for faster processing
- Implement early stopping in grid search
- Consider using approximate KNN algorithms for large datasets

#### For HarDNet-MSEG:
- Use mixed precision training to reduce memory usage
- Implement gradient checkpointing for large models
- Use data loading optimization with multiple workers

---

## Contributing Guidelines

When extending this codebase:

1. **Follow the existing code structure** and documentation format
2. **Add comprehensive docstrings** for all new functions
3. **Include usage examples** for new features
4. **Update this documentation** when adding new components
5. **Test your code** with the provided validation functions

### Code Style
- Follow PEP 8 for Python code formatting
- Use descriptive variable names
- Include type hints where applicable
- Add comments for complex logic

### Documentation Requirements
- Document all public functions with parameters, returns, and examples
- Include error handling information
- Provide performance considerations
- Add troubleshooting guidance for common issues

---

This documentation provides comprehensive coverage of all public APIs, functions, and components in the deep learning algorithms repository. For specific implementation details, refer to the source code and inline comments.