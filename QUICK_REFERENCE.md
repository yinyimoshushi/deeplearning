# Quick Reference Guide

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/yinyimoshushi/deeplearning.git
cd deeplearning
pip install -r requirements.txt
```

### Basic Usage
```python
# KNN Location Prediction
from knn import knncls
knncls()  # Requires train.csv in directory

# Medical Image Segmentation  
cd day01/
jupyter notebook HarDNet_MSEG.ipynb
```

---

## 📁 Project Structure

```
deeplearning/
├── knn.py                    # KNN location prediction
├── day01/
│   └── HarDNet_MSEG.ipynb   # Medical image segmentation
├── train.csv                # Data file (user-provided)
├── requirements.txt         # Dependencies
├── API_DOCUMENTATION.md     # Complete API docs
├── FUNCTION_REFERENCE.md    # Detailed function specs
└── QUICK_REFERENCE.md       # This file
```

---

## 🔧 Core APIs

### KNN Module (`knn.py`)

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `knncls()` | Complete KNN pipeline | None (reads train.csv) | None (prints results) |

**Required Data Format (train.csv):**
```csv
x,y,time,place_id,row_id
1.1,2.6,1591776000,loc_1,1
1.15,2.65,1591779600,loc_2,2
```

### HarDNet-MSEG Notebook

| Cell Type | Purpose | Key Commands |
|-----------|---------|--------------|
| Setup | Environment initialization | `!git clone`, `!pip install` |
| Data | Dataset preparation | `!unzip`, `!cp` |
| Training | Model training | Execute notebook cells |

---

## ⚡ Quick Commands

### KNN Execution
```python
# Basic usage
python -c "from knn import knncls; knncls()"

# With custom data validation
python -c "
import pandas as pd
data = pd.read_csv('train.csv')
print(f'Data shape: {data.shape}')
print(f'Columns: {list(data.columns)}')
from knn import knncls
knncls()
"
```

### Environment Check
```python
# Check dependencies
import sys
import pkg_resources

required = ['scikit-learn', 'pandas', 'numpy', 'torch']
for package in required:
    try:
        pkg_resources.get_distribution(package)
        print(f"✅ {package}")
    except:
        print(f"❌ {package} - run: pip install {package}")
```

---

## 🛠️ Common Usage Patterns

### Data Preparation
```python
# Create sample data for testing
import pandas as pd
import numpy as np

# Generate sample location data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'x': np.random.uniform(1.0, 1.25, n_samples),
    'y': np.random.uniform(2.5, 2.75, n_samples),
    'time': np.random.randint(1591776000, 1591876000, n_samples),
    'place_id': np.random.choice(['loc_1', 'loc_2', 'loc_3'], n_samples),
    'row_id': range(n_samples)
})

data.to_csv('train.csv', index=False)
print("Sample data created: train.csv")
```

### Custom KNN Parameters
```python
# Modify knncls() for custom parameters
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def custom_knn(csv_path='train.csv', k_values=[3, 5, 7, 10]):
    data = pd.read_csv(csv_path)
    # ... (follow knncls() workflow with custom k_values)
    param = {"n_neighbors": k_values}
    # ... rest of implementation
```

---

## 🐛 Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: train.csv` | Missing data file | Create/place train.csv in directory |
| `KeyError: 'x'` | Wrong CSV format | Ensure columns: x,y,time,place_id,row_id |
| `Empty DataFrame` | Geographical filtering | Adjust filter ranges in knncls() |
| `CUDA out of memory` | GPU memory limit | Reduce batch size or use CPU |

### Quick Fixes
```python
# Check data file
import os
print("✅ train.csv exists" if os.path.exists('train.csv') else "❌ train.csv missing")

# Validate data format
if os.path.exists('train.csv'):
    df = pd.read_csv('train.csv')
    required_cols = ['x', 'y', 'time', 'place_id', 'row_id']
    missing = [col for col in required_cols if col not in df.columns]
    print(f"Missing columns: {missing}" if missing else "✅ All columns present")

# Memory usage check
import psutil
memory_usage = psutil.virtual_memory().percent
print(f"Memory usage: {memory_usage:.1f}%")
```

---

## 📊 Performance Tips

### KNN Optimization
- **Smaller regions**: Tighter geographical bounds = faster processing
- **Fewer neighbors**: Lower k values = faster prediction
- **Data size**: Use data sampling for large datasets

### Memory Management
```python
# Monitor memory usage
import tracemalloc
tracemalloc.start()
# ... run your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

---

## 🔗 Related Resources

### Documentation
- **Complete API**: `API_DOCUMENTATION.md`
- **Function Details**: `FUNCTION_REFERENCE.md`
- **Dependencies**: `requirements.txt`

### External Links
- [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [HarDNet Paper](https://arxiv.org/abs/1909.00218)
- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)

### Code Examples
```python
# Complete minimal example
import pandas as pd
import numpy as np

# 1. Create test data
data = pd.DataFrame({
    'x': [1.1, 1.15, 1.2],
    'y': [2.6, 2.65, 2.7],
    'time': [1591776000, 1591779600, 1591783200],
    'place_id': ['A', 'B', 'A'],
    'row_id': [1, 2, 3]
})
data.to_csv('train.csv', index=False)

# 2. Run KNN
from knn import knncls
knncls()

# 3. Check results
print("KNN execution completed!")
```

---

## 📞 Support

For issues and questions:
1. Check this quick reference
2. Review `API_DOCUMENTATION.md`
3. Examine `FUNCTION_REFERENCE.md`
4. Create GitHub issue with error details

**Happy coding! 🎉**