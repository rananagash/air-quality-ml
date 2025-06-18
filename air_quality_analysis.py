#!/usr/bin/env python3
"""
Air Quality Analysis with UCI Dataset
=====================================

This script demonstrates how to:
- Load and clean the UCI Air Quality dataset
- Perform regression to predict NO2 levels
- Classify air quality into risk categories
- Visualize the data and results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Air Quality Analysis...\n")
    
    # Step 1: Load the dataset
    print("📊 Loading dataset...")
    df = pd.read_csv('air+quality/AirQualityUCI.csv', sep=';', decimal=',', engine='python')
    df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'])
    print(f"Dataset shape: {df.shape}")
    
    # Step 2: Clean the data
    print("\n🧹 Cleaning data...")
    df.replace(-200, np.nan, inplace=True)
    print(f"Rows before dropping NaN: {len(df)}")
    
    df.dropna(inplace=True)
    print(f"Rows after dropping NaN: {len(df)}")
    
    # Convert Date/Time to datetime
    df['Time'] = df['Time'].str.replace('.', ':', regex=False)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.drop(columns=['Date', 'Time'])
    df = df.set_index('Datetime')
    
    # Step 3: Data exploration
    print("\n📈 Data exploration...")
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        start = pd.Timestamp(df.index.min()).strftime('%Y-%m-%d')  # type: ignore
        end = pd.Timestamp(df.index.max()).strftime('%Y-%m-%d')  # type: ignore
        print(f"   • Time period: {start} to {end}")
    else:
        print("   • Time period: N/A")
    print(f"Average NO2 level: {df['NO2(GT)'].mean():.1f} µg/m³")
    
    # Step 4: Prepare features and target
    target = 'NO2(GT)'
    features = ['T', 'RH', 'AH']
    
    X = df[features]
    y = df[target]
    
    # Step 5: Regression analysis
    print("\n🎯 Running regression analysis...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    print(f"Linear Regression R²: {r2_score(y_test, y_pred_lr):.3f}")
    print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
    
    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    print(f"Random Forest R²: {r2_score(y_test, y_pred_rf):.3f}")
    print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
    
    # Step 6: Classification analysis
    print("\n🏷️ Running classification analysis...")
    
    def label_aqi(no2):
        if no2 <= 50:
            return 'Good'
        elif no2 <= 100:
            return 'Moderate'
        elif no2 <= 150:
            return 'Unhealthy for Sensitive'
        else:
            return 'Unhealthy'
    
    df['AQI_Label'] = df['NO2(GT)'].apply(label_aqi)
    
    X_class = df[features]
    y_class = df['AQI_Label']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    
    print("Classification Report:")
    print(classification_report(y_test_c, y_pred_c))
    
    # Step 7: Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Air Quality Analysis Results', fontsize=16)
    
    # 1. Time series of NO2
    axes[0, 0].plot(df.index, df['NO2(GT)'], alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('NO2 Levels Over Time')
    axes[0, 0].set_ylabel('NO2 (µg/m³)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. AQI distribution
    aqi_counts = df['AQI_Label'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    bars = axes[0, 1].bar(aqi_counts.index, aqi_counts.values, color=colors[:len(aqi_counts)])
    axes[0, 1].set_title('Distribution of Air Quality Categories')
    axes[0, 1].set_ylabel('Number of Observations')
    
    # Add count labels on bars
    for bar, count in zip(bars, aqi_counts.values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                       str(count), ha='center', va='bottom')
    
    # 3. Regression results comparison
    axes[1, 0].scatter(y_test, y_pred_lr, alpha=0.6, label='Linear Regression', s=20)
    axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.6, label='Random Forest', s=20)
    axes[1, 0].plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual NO2')
    axes[1, 0].set_ylabel('Predicted NO2')
    axes[1, 0].set_title('Regression Results Comparison')
    axes[1, 0].legend()
    
    # 4. Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1, 1].set_title('Feature Importance (Random Forest)')
    axes[1, 1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('air_quality_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 8: Summary
    print("\n" + "="*50)
    print("📋 ANALYSIS SUMMARY")
    print("="*50)
    print(f"📊 Dataset Information:")
    print(f"   • Total observations: {len(df):,}")
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        start_date = df.index.min()
        end_date = df.index.max()
        if bool(pd.isna(start_date)) == False and bool(pd.isna(end_date)) == False:
            start = pd.Timestamp(start_date).strftime('%Y-%m-%d')  # type: ignore
            end = pd.Timestamp(end_date).strftime('%Y-%m-%d')  # type: ignore
            print(f"   • Time period: {start} to {end}")
        else:
            print("   • Time period: N/A")
    else:
        print("   • Time period: N/A")
    print(f"   • Features used: {', '.join(features)}")
    print(f"   • Target variable: {target}")
    
    print(f"\n🎯 Regression Results:")
    print(f"   • Linear Regression R²: {r2_score(y_test, y_pred_lr):.3f}")
    print(f"   • Random Forest R²: {r2_score(y_test, y_pred_rf):.3f}")
    print(f"   • Best model: {'Random Forest' if r2_score(y_test, y_pred_rf) > r2_score(y_test, y_pred_lr) else 'Linear Regression'}")
    
    print(f"\n🏷️ Classification Results:")
    print(f"   • Air quality categories: {', '.join(clf.classes_)}")
    print(f"   • Most common category: {df['AQI_Label'].mode().iloc[0]}")
    
    print(f"\n🔍 Key Insights:")
    print(f"   • Most important feature: {feature_importance.iloc[-1]['feature']}")
    print(f"   • Average NO2 level: {df['NO2(GT)'].mean():.1f} µg/m³")
    print(f"   • NO2 range: {df['NO2(GT)'].min():.1f} - {df['NO2(GT)'].max():.1f} µg/m³")
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 Results saved to: air_quality_analysis_results.png")

if __name__ == "__main__":
    main() 