# Air Quality ML Analysis

Use satellite imagery + weather data + urban metrics to predict air pollution levels (regression) and classify air quality risk (classification) into categories like Good, Moderate, Unhealthy, etc.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
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
- **Models**: Linear Regression, Random Forest
- **Metrics**: R² Score, RMSE

### Classification Analysis
- **Target**: Air Quality Index (AQI) categories
- **Categories**: Good, Moderate, Unhealthy for Sensitive, Unhealthy
- **Model**: Random Forest Classifier
- **Metrics**: Classification Report, Confusion Matrix

### Visualizations
- Time series plots of air quality metrics
- Correlation heatmaps
- Regression results comparison
- Feature importance analysis
- AQI category distribution
- Seasonal patterns analysis

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
- **Regression**: Random Forest typically outperforms Linear Regression
- **Classification**: High accuracy in predicting air quality categories
- **Feature importance**: Temperature and humidity are key predictors

### Air Quality Patterns
- **Seasonal variations**: Clear patterns across months
- **Daily patterns**: Hourly fluctuations in pollution levels
- **Weather correlation**: Strong relationship with temperature and humidity

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

## 📈 Output Files

- `air_quality_analysis_results.png`: Comprehensive visualization dashboard
- Console output: Detailed analysis results and metrics

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

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy analyzing! 🌬️📊**
