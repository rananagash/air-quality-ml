# Air Quality Analysis with Machine Learning

A comprehensive analysis of the UCI Air Quality dataset using multiple machine learning approaches, including **both PyTorch and TensorFlow** for deep learning models.

## Features

- **Data Preprocessing**: Clean and prepare air quality data
- **Regression Analysis**: Predict NO₂ levels using multiple algorithms
- **Classification**: Categorize air quality into risk levels
- **Deep Learning**: 
  - **PyTorch** for regression (NO₂ prediction)
  - **TensorFlow/Keras** for classification (AQI categories)
- **Visualization**: Interactive plots and analysis results
- **Model Comparison**: Compare performance across different algorithms

## Dual Framework Approach

This project demonstrates versatility by using both major deep learning frameworks:

- **PyTorch**: Used for regression tasks (predicting continuous NO₂ values)
- **TensorFlow/Keras**: Used for classification tasks (categorizing air quality levels)

This approach showcases:
- Adaptability to different frameworks
- Understanding of framework-specific strengths
- Ability to compare and choose appropriate tools for specific tasks

## Dataset

UCI Air Quality Dataset containing:
- Temperature, Humidity, Absolute Humidity
- NO₂, CO, NOx levels
- Time series data from 2004-2005

## 🛠Installation

```bash
# Clone the repository
git clone <repository-url>
cd air-quality-ml

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- **PyTorch** (for regression models)
- **TensorFlow** (for classification models)

## Usage

### Run the complete analysis:
```bash
python air_quality_analysis.py
```

### Or use Jupyter Notebook:
```bash
jupyter notebook air_quality_analysis.ipynb
```

## Results

The analysis provides:
- Regression models (Linear, Random Forest, PyTorch Deep Learning)
- Classification models (Random Forest, TensorFlow Deep Learning)
- Feature importance analysis
- Performance comparisons
- Interactive visualizations

## Learning Outcomes

This project demonstrates:
- **Data Science Workflow**: From data loading to model deployment
- **Multiple ML Approaches**: Traditional ML and deep learning
- **Framework Versatility**: PyTorch and TensorFlow implementation
- **Model Evaluation**: Comprehensive performance analysis
- **Visualization Skills**: Creating informative plots and dashboards

## Project Structure

```
air-quality-ml/
├── air_quality_analysis.py      # Main analysis script
├── air_quality_analysis.ipynb   # Jupyter notebook version
├── requirements.txt             # Dependencies
├── README.md                   # This file
└── air+quality/                # Dataset directory
    └── AirQualityUCI.csv       # UCI Air Quality dataset
```

## Customization

- Modify feature selection in the script
- Adjust model hyperparameters
- Add new visualization types
- Extend with additional ML algorithms

## Sample Output

The script generates:
- Console output with model performance metrics
- `air_quality_analysis_results.png` with comprehensive visualizations
- Detailed analysis summary
