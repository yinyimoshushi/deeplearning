# Deep Learning Algorithms Repository

A comprehensive collection of deep learning and machine learning algorithms with implementations for location prediction and medical image segmentation.

## 🚀 Quick Start

```bash
git clone https://github.com/yinyimoshushi/deeplearning.git
cd deeplearning
pip install -r requirements.txt
```

## 📋 Components

### 1. K-Nearest Neighbors Location Prediction (`knn.py`)
- Predicts user check-in locations using geographical and temporal features
- Implements complete ML pipeline with feature engineering and hyperparameter optimization
- **Usage**: `python -c "from knn import knncls; knncls()"`

### 2. HarDNet-MSEG Medical Image Segmentation (`day01/HarDNet_MSEG.ipynb`)
- State-of-the-art medical image segmentation for endoscopic polyp detection
- Uses HarDNet (Harmonic DenseNet) architecture with multi-scale processing
- **Usage**: `jupyter notebook day01/HarDNet_MSEG.ipynb`

## 📚 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[🔗 Quick Reference](QUICK_REFERENCE.md)** | Fast overview and common commands | All users |
| **[📖 API Documentation](API_DOCUMENTATION.md)** | Comprehensive API guide with examples | Developers |
| **[🔧 Function Reference](FUNCTION_REFERENCE.md)** | Detailed function specifications | Advanced users |

## 🛠️ Installation

### System Requirements
- Python 3.7+
- 8GB+ RAM (16GB recommended)
- CUDA-compatible GPU (recommended for deep learning)

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages: `scikit-learn`, `pandas`, `torch`, `torchvision`, `jupyter`

## ⚡ Quick Examples

### KNN Location Prediction
```python
# Create sample data
import pandas as pd
data = pd.DataFrame({
    'x': [1.1, 1.15, 1.2], 'y': [2.6, 2.65, 2.7],
    'time': [1591776000, 1591779600, 1591783200],
    'place_id': ['A', 'B', 'A'], 'row_id': [1, 2, 3]
})
data.to_csv('train.csv', index=False)

# Run prediction
from knn import knncls
knncls()
```

### Medical Image Segmentation
```bash
cd day01/
jupyter notebook HarDNet_MSEG.ipynb
# Follow notebook cells for complete setup
```

## 🎯 Use Cases

- **Location Analytics**: User behavior prediction, location recommendation
- **Medical Imaging**: Polyp detection, medical diagnosis assistance
- **Research**: Algorithm comparison, feature engineering studies
- **Education**: Learning ML/DL concepts with practical examples

## 🤝 Contributing

1. Fork the repository
2. Follow the documentation standards in existing files
3. Add comprehensive tests and examples
4. Update relevant documentation files

## 📄 License

Open source project for educational and research purposes.

---

**For detailed usage instructions, see [📖 API Documentation](API_DOCUMENTATION.md)**

**For quick commands, see [🔗 Quick Reference](QUICK_REFERENCE.md)**
