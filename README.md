# Air Quality ML Analysis

Use satellite imagery + weather data + urban metrics to predict air pollution levels (regression) and classify air quality risk (classification) into categories like Good, Moderate, Unhealthy, etc.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ (TensorFlow requires Python < 3.12)
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd air-quality-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   # Option 1: Run the Python script
   python air_quality_analysis.py
   
   # Option 2: Use Jupyter notebook
   jupyter notebook air_quality_analysis.ipynb
   ```

## 📊 Dataset

This project uses the **UCI Air Quality Dataset** which contains:
- **Time period**: 2004-2005
- **Location**: City of Rome, Italy
- **Variables**: 
  - Air pollutants (NO2, CO, NMHC, etc.)
  - Weather conditions (Temperature, Humidity, etc.)
  - Temporal features (Date, Time)

### Data Cleaning Steps
1. Replace missing values (-200) with NaN
2. Drop rows with missing data
3. Convert Date/Time to datetime index
4. Handle data type conversions

## 🎯 Analysis Features

### Regression Analysis
- **Target**: NO2(GT) levels (µg/m³)
- **Features**: Temperature (T), Relative Humidity (RH), Absolute Humidity (AH)
- **Models**: 
  - Linear Regression
  - Random Forest
  - Deep Neural Network (TensorFlow/Keras)
- **Metrics**: R² Score, RMSE

### Classification Analysis
- **Target**: Air Quality Index (AQI) categories
- **Categories**: Good, Moderate, Unhealthy for Sensitive, Unhealthy
- **Models**: 
  - Random Forest Classifier
  - Deep Neural Network (TensorFlow/Keras)
- **Metrics**: Classification Report, Confusion Matrix

### Deep Learning Models
- **Architecture**: Multi-layer perceptron (MLP)
- **Regression**: 2 hidden layers (64, 32 neurons) with ReLU activation
- **Classification**: 2 hidden layers with softmax output
- **Training**: Adam optimizer, 50 epochs, batch size 32
- **Data Preprocessing**: StandardScaler normalization

### Visualizations
- Time series plots of air quality metrics
- Correlation heatmaps
- Regression results comparison (all models)
- Feature importance analysis
- AQI category distribution
- Seasonal patterns analysis
- Deep learning training history (when available)

## 📁 Project Structure

```
air-quality-ml/
├── air+quality/
│   ├── AirQualityUCI.csv          # Main dataset
│   └── AirQualityUCI.xlsx         # Excel version
├── air_quality_analysis.ipynb     # Jupyter notebook
├── air_quality_analysis.py        # Python script
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 🔍 Key Insights

### Data Characteristics
- **Total observations**: ~9,000+ after cleaning
- **Time span**: ~1 year of hourly measurements
- **Missing data**: Handled by removing rows with -200 values

### Model Performance
- **Regression**: Linear Regression typically performs best on this dataset
- **Classification**: High accuracy in predicting air quality categories
- **Feature importance**: Temperature and humidity are key predictors
- **Deep Learning**: Provides additional modeling option when TensorFlow is available

### Air Quality Patterns
- **Seasonal variations**: Clear patterns across months
- **Daily patterns**: Hourly fluctuations in pollution levels
- **Weather correlation**: Strong relationship with temperature and humidity

## 🧠 Deep Learning Setup

### TensorFlow Installation
```bash
# For Python < 3.12
pip install tensorflow

# For macOS with Apple Silicon
pip install tensorflow-macos

# For GPU support
pip install tensorflow-gpu
```

### Model Architecture
```python
# Regression Model
Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Classification Model
Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
```

## 🛠️ Customization

### Adding New Features
```python
# Add more features to the analysis
features = ['T', 'RH', 'AH', 'CO(GT)', 'PT08.S1(CO)']
```

### Modifying AQI Categories
```python
def custom_aqi_label(no2):
    if no2 <= 30:
        return 'Excellent'
    elif no2 <= 60:
        return 'Good'
    # ... add more categories
```

### Using Different Models
```python
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Add your preferred models
```

### Customizing Deep Learning
```python
# Modify neural network architecture
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Change training parameters
dl_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)
```

## 📈 Output Files

- `air_quality_analysis_results.png`: Comprehensive visualization dashboard
- Console output: Detailed analysis results and metrics
- Model performance comparisons

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- [Air Quality Index Standards](https://www.epa.gov/air-quality-index)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy analyzing! 🌬️📊🧠**
