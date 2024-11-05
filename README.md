# CHEG47_MLWorkshop2

# Hydrogen Storage Material Analysis Using Machine Learning

## Objective
To analyze and predict hydrogen uptake/adsorption and desorption performance of carbon nanotube-based materials using various material properties and operating conditions.

## Dataset Overview
- 539 samples
- Features:
  - Material properties: Average crystal size, Crystallinity index, BET surface area, Pore volume
  - Operating conditions: Temperature, Pressure
  - Material types: Various functionalized carbon nanotubes and composites
- Target variables:
  - Hydrogen uptake/adsorption (wt%)
  - Hydrogen desorption (wt%)

## Data Processing
1. Data cleaning:
   - No duplicates found
   - Filled missing values with mean values
   - Outlier analysis performed using Z-score method
2. Feature engineering:
   - One-hot encoding for material types
   - Standard scaling for numerical features

## Model Performance
### Hydrogen Uptake/Adsorption:
- Random Forest: R² = 0.4079, MSE = 0.1131, CV score = 0.6674
- XGBoost: R² = 0.2614, MSE = 0.1411, CV score = 0.6603
- Decision Tree: R² = 0.1991, MSE = 0.1530, CV score = 0.4803

### Hydrogen Desorption:
- Random Forest: R² = 0.2743, MSE = 0.1146, CV score = 0.0268
- XGBoost: R² = 0.1872, MSE = 0.1284, CV score = 0.0806
- Decision Tree: R² = 0.1590, MSE = 0.1328, CV score = -0.6214

## Key Findings
1. Most influential features:
   - Temperature and Pressure for uptake
   - Pressure and Crystallinity index for desorption
2. Material characteristics:
   - Pore volume shows moderate importance
   - BET surface area has lower impact than expected

## Usage
```python
from sklearn.pipeline import Pipeline
model = joblib.load('best_model.joblib')
predictions = model.predict(X_new)
```

## Dependencies
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- seaborn
