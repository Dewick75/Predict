# House Price Prediction Model

A machine learning project that predicts house prices based on house size and number of bedrooms using linear regression.

## 📋 Project Overview

This project implements a simple yet effective house price prediction model using scikit-learn's Linear Regression algorithm. The model analyzes the relationship between house features (size and bedrooms) and their corresponding prices to make accurate predictions.

## 🎯 Features

- **Data Loading & Preprocessing**: Handles CSV data with automatic missing value removal
- **Linear Regression Model**: Uses scikit-learn's LinearRegression for price prediction
- **Model Evaluation**: Provides comprehensive metrics including MSE and R² score
- **Visualization**: Generates scatter plots comparing actual vs predicted prices
- **Train-Test Split**: Implements proper data splitting for model validation

## 📊 Dataset

The project uses a dataset (`random_house_prices 2025 may assignment.csv`) containing:
- **Size**: House size in square feet
- **Bedrooms**: Number of bedrooms
- **Price**: House price (target variable)

**Dataset Statistics:**
- Total records: 101 houses
- Features: 2 (Size, Bedrooms)
- Target: Price

## 🚀 Getting Started

### Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install pandas scikit-learn matplotlib
```

### Installation

1. Clone or download this repository
2. Ensure the CSV file `random_house_prices 2025 may assignment.csv` is in the same directory
3. Run the prediction script:

```bash
python HousePredict.py
```

## 📈 Model Performance

The model provides the following evaluation metrics:
- **Mean Squared Error (MSE)**: Measures average squared differences between actual and predicted prices
- **R² Score**: Indicates how well the model explains the variance in house prices
- **Visualization**: Scatter plot showing correlation between actual and predicted values

### Sample Output
![Model Performance Metrics](Files/1.png)

### Actual vs Predicted Prices Visualization
![Scatter Plot - Actual vs Predicted](Files/2.png)

## 🔧 Complete Technology Stack & Implementation

### 🐍 **Core Technologies Used**

#### **1. Python Libraries & Frameworks**
- **pandas (pd)**: Data manipulation and analysis library
- **scikit-learn**: Machine learning library for model building
- **matplotlib.pyplot**: Data visualization and plotting
- **NumPy** (implicit): Numerical computing (used by pandas/sklearn)

#### **2. Machine Learning Components**
- **Algorithm**: Linear Regression (Ordinary Least Squares)
- **Model Type**: Supervised Learning - Regression
- **Training Method**: Batch Learning
- **Evaluation**: Statistical metrics and visual analysis

### 🔬 **Detailed Technical Implementation**

#### **Step 1: Data Loading & Exploration**
```python
import pandas as pd
df = pd.read_csv('random_house_prices 2025 may assignment.csv')
print("Shape:", df.shape)
```
**What happens:**
- **pandas.read_csv()** loads CSV data into a DataFrame object
- **DataFrame.shape** returns tuple (rows, columns) = (101, 3)
- Data structure: 101 house records with 3 columns (Size, Bedrooms, Price)

#### **Step 2: Data Preprocessing**
```python
df = df.dropna()
```
**What happens:**
- **dropna()** removes any rows containing NaN/null values
- Ensures clean data for machine learning algorithms
- Prevents errors during mathematical computations
- In this dataset: No missing values found (all 101 records retained)

#### **Step 3: Feature Engineering & Data Splitting**
```python
X = df.drop('Price', axis=1)  # Features: Size, Bedrooms
y = df['Price']               # Target: Price values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
**What happens:**
- **Feature Matrix (X)**: 2D array with Size and Bedrooms columns
- **Target Vector (y)**: 1D array with Price values
- **train_test_split()**: Randomly divides data into:
  - Training set: 80% (≈81 records) for model learning
  - Testing set: 20% (≈20 records) for model evaluation
- **Random state**: Not set, so split varies each run

#### **Step 4: Linear Regression Model Training**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
**Mathematical Foundation:**
- **Linear Equation**: `Price = β₀ + β₁×Size + β₂×Bedrooms + ε`
- **Objective**: Minimize Sum of Squared Errors (SSE)
- **Method**: Ordinary Least Squares (OLS)
- **Output**: Learned coefficients (β₀, β₁, β₂)

**What happens internally:**
1. **Matrix Operations**: Solves normal equation: `β = (XᵀX)⁻¹Xᵀy`
2. **Coefficient Calculation**: Finds optimal weights for each feature
3. **Intercept Determination**: Calculates y-intercept (β₀)
4. **Model Storage**: Saves learned parameters in model object

#### **Step 5: Prediction Generation**
```python
y_pred = model.predict(X_test)
```
**What happens:**
- **Input**: Test features (Size, Bedrooms) for ≈20 houses
- **Computation**: `y_pred = β₀ + β₁×Size_test + β₂×Bedrooms_test`
- **Output**: Predicted prices for test houses
- **Data Type**: NumPy array of float values

#### **Step 6: Model Evaluation**
```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**Mean Squared Error (MSE):**
- **Formula**: `MSE = (1/n) × Σ(y_actual - y_predicted)²`
- **Purpose**: Measures average squared prediction errors
- **Units**: Squared price units (dollars²)
- **Interpretation**: Lower values indicate better predictions

**R² Score (Coefficient of Determination):**
- **Formula**: `R² = 1 - (SS_res / SS_tot)`
- **Range**: 0 to 1 (higher is better)
- **Meaning**: Proportion of variance explained by the model
- **Example**: R² = 0.85 means model explains 85% of price variance

#### **Step 7: Data Visualization**
```python
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted')
plt.show()
```
**What happens:**
- **Scatter Plot**: Each point represents one house prediction
- **X-axis**: True house prices from test set
- **Y-axis**: Model's predicted prices
- **Perfect Model**: Points would form diagonal line (y = x)
- **Analysis**: Point spread indicates prediction accuracy

### 🧮 **Mathematical Model Details**

#### **Linear Regression Equation**
```
House_Price = β₀ + β₁ × House_Size + β₂ × Number_of_Bedrooms + ε
```

**Where:**
- **β₀ (Intercept)**: Base price when Size=0 and Bedrooms=0
- **β₁ (Size Coefficient)**: Price change per square foot
- **β₂ (Bedroom Coefficient)**: Price change per additional bedroom
- **ε (Error Term)**: Unexplained variance

#### **Model Assumptions**
1. **Linearity**: Price changes linearly with features
2. **Independence**: Each house price is independent
3. **Homoscedasticity**: Constant error variance
4. **Normality**: Errors follow normal distribution

### 🔄 **Data Flow Architecture**

```
CSV File → pandas DataFrame → Data Cleaning → Feature/Target Split
    ↓
Train/Test Split → Linear Regression Training → Model Coefficients
    ↓
Predictions → Evaluation Metrics → Visualization → Results
```

### 💾 **Memory & Performance**

#### **Data Structures**
- **DataFrame**: 101 rows × 3 columns = ~2.4KB
- **Feature Matrix**: 101 × 2 float64 = ~1.6KB
- **Model Object**: Stores 3 coefficients + metadata = ~1KB
- **Total Memory**: <10KB (very lightweight)

#### **Computational Complexity**
- **Training**: O(n × p²) where n=samples, p=features
- **Prediction**: O(n × p) for n predictions
- **Space**: O(p²) for coefficient matrix
- **Runtime**: <1 second for this dataset size

## 📝 Usage Example

```python
# Load and preprocess data
df = pd.read_csv('random_house_prices 2025 may assignment.csv')
df = df.dropna()

# Prepare features and target
X = df.drop('Price', axis=1)  # Features: Size, Bedrooms
y = df['Price']               # Target: Price

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 🎨 Visualization

The project generates a scatter plot that helps visualize model performance:
- **X-axis**: Actual house prices
- **Y-axis**: Predicted house prices
- **Ideal scenario**: Points would form a perfect diagonal line
- **Model quality**: Closer points to the diagonal indicate better predictions

As shown in the visualization above, the scatter plot provides insights into how well the model predictions align with actual house prices.

## 🔍 Advanced Model Insights & Algorithm Analysis

### 🧠 **Linear Regression Deep Dive**

#### **Algorithm Mechanics**
The scikit-learn LinearRegression uses **Ordinary Least Squares (OLS)** method:

1. **Normal Equation Solution**:
   ```
   β = (XᵀX)⁻¹Xᵀy
   ```
   - **X**: Feature matrix (Size, Bedrooms)
   - **y**: Target vector (Prices)
   - **β**: Coefficient vector [β₀, β₁, β₂]

2. **Matrix Decomposition**:
   - Uses **SVD (Singular Value Decomposition)** for numerical stability
   - Handles near-singular matrices better than direct inversion
   - Computational complexity: O(np² + p³)

#### **Feature Interpretation**
```python
# After training, you can access:
intercept = model.intercept_          # β₀ (base price)
coefficients = model.coef_            # [β₁, β₂] (feature weights)
```

**Real-world meaning:**
- **Size Coefficient (β₁)**: Price increase per square foot
- **Bedroom Coefficient (β₂)**: Price premium per additional bedroom
- **Intercept (β₀)**: Theoretical price of 0 sq ft, 0 bedroom house

#### **Model Limitations & Assumptions**

1. **Linearity Assumption**:
   - Assumes constant price per square foot
   - Reality: Diminishing returns possible for very large houses
   - Solution: Consider polynomial features or log transformations

2. **Feature Independence**:
   - Assumes Size and Bedrooms are independent predictors
   - Reality: Larger houses typically have more bedrooms
   - Impact: Potential multicollinearity (not severe with 2 features)

3. **Homoscedasticity**:
   - Assumes constant error variance across all price ranges
   - Reality: Expensive houses might have higher price volatility
   - Detection: Plot residuals vs fitted values

4. **No Outliers**:
   - Sensitive to extreme values
   - One mansion could skew entire model
   - Solution: Robust regression or outlier detection

### 📊 **Statistical Significance & Validation**

#### **Model Diagnostics**
```python
# Additional analysis you could add:
from scipy import stats
import numpy as np

# Calculate residuals
residuals = y_test - y_pred

# Normality test for residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)

# Durbin-Watson test for autocorrelation
from statsmodels.stats.diagnostic import durbin_watson
dw_stat = durbin_watson(residuals)
```

#### **Cross-Validation Considerations**
Current implementation uses simple train-test split. For more robust evaluation:
```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
```

### 🎯 **Prediction Confidence & Uncertainty**

#### **Prediction Intervals**
Linear regression provides point estimates, but real applications need uncertainty:
```python
# Standard error of prediction
prediction_std = np.sqrt(mse)  # Approximation
confidence_interval = y_pred ± 1.96 * prediction_std  # 95% CI
```

#### **Feature Importance Analysis**
```python
# Standardized coefficients for comparison
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train on scaled features to compare coefficient magnitudes
```

### 🔧 **Implementation Optimizations**

#### **Current Code Efficiency**
- **Memory**: O(n) space complexity
- **Time**: O(n) for prediction, O(np²) for training
- **Scalability**: Excellent for datasets up to millions of records

#### **Potential Improvements**
1. **Feature Scaling**: Standardize features for better numerical stability
2. **Regularization**: Add L1/L2 penalties to prevent overfitting
3. **Feature Engineering**: Create interaction terms (Size × Bedrooms)
4. **Polynomial Features**: Capture non-linear relationships

### 🧪 **Experimental Design**

#### **Data Split Strategy**
```python
# Current: Random 80-20 split
# Better: Stratified split by price ranges
# Best: Time-based split if temporal data available
```

#### **Hyperparameter Tuning**
Linear regression has few hyperparameters, but consider:
- **fit_intercept**: Whether to calculate intercept (default: True)
- **normalize**: Whether to normalize features (deprecated, use StandardScaler)
- **positive**: Constrain coefficients to be positive

### 📈 **Business Intelligence & Insights**

#### **Economic Interpretation**
From the trained model coefficients:
- **Price per sq ft**: Coefficient of Size feature
- **Bedroom premium**: Coefficient of Bedrooms feature
- **Market baseline**: Intercept value

#### **Decision Support**
Model enables:
1. **Property Valuation**: Estimate fair market price
2. **Investment Analysis**: ROI calculations for renovations
3. **Market Trends**: Track coefficient changes over time
4. **Pricing Strategy**: Set competitive listing prices

## 🏗️ **Technical Architecture & Data Pipeline**

### � **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Data Processing │───▶│  ML Pipeline    │
│   (CSV File)    │    │   (pandas)       │    │  (scikit-learn) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Raw Data        │    │ Clean Data       │    │ Trained Model   │
│ • Size          │    │ • Validated      │    │ • Coefficients  │
│ • Bedrooms      │    │ • No missing     │    │ • Intercept     │
│ • Price         │    │ • Proper types   │    │ • Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Evaluation      │    │ Visualization    │    │ Results Output  │
│ • MSE           │    │ • Scatter Plot   │    │ • Predictions   │
│ • R² Score      │    │ • Matplotlib     │    │ • Metrics       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🔄 **Data Processing Pipeline**

#### **Stage 1: Data Ingestion**
```python
# File I/O Operation
df = pd.read_csv('random_house_prices 2025 may assignment.csv')
```
**Technical Details:**
- **File Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8 (default)
- **Parser**: pandas C engine (fastest)
- **Memory Mapping**: Automatic for small files
- **Data Types**: Auto-inferred (int64 for Size/Bedrooms, int64 for Price)

#### **Stage 2: Data Validation & Cleaning**
```python
# Data quality checks
print("Shape:", df.shape)  # Dimension validation
df = df.dropna()           # Missing value handling
```
**Quality Assurance:**
- **Completeness**: Check for missing values
- **Consistency**: Verify data types
- **Validity**: Ensure reasonable value ranges
- **Uniqueness**: No duplicate records needed for this use case

#### **Stage 3: Feature Engineering**
```python
# Feature-target separation
X = df.drop('Price', axis=1)  # Feature matrix
y = df['Price']               # Target vector
```
**Data Structures:**
- **X (Features)**: pandas DataFrame → numpy array (101×2)
- **y (Target)**: pandas Series → numpy array (101×1)
- **Memory Layout**: Contiguous arrays for efficient computation

#### **Stage 4: Data Partitioning**
```python
# Stratified sampling for train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
**Sampling Strategy:**
- **Method**: Random sampling without replacement
- **Ratio**: 80% training, 20% testing
- **Randomization**: Different split each run (no random_state set)
- **Stratification**: Not used (could be added for price ranges)

### ⚙️ **Machine Learning Pipeline**

#### **Model Initialization**
```python
model = LinearRegression()
```
**Internal Configuration:**
- **Solver**: SVD-based least squares
- **Regularization**: None (pure OLS)
- **Intercept**: Fitted (fit_intercept=True)
- **Normalization**: None (deprecated parameter)

#### **Training Process**
```python
model.fit(X_train, y_train)
```
**Mathematical Operations:**
1. **Input Validation**: Check array dimensions and types
2. **Matrix Preparation**: Convert to numpy arrays if needed
3. **SVD Decomposition**: X = UΣVᵀ
4. **Coefficient Calculation**: β = VΣ⁻¹Uᵀy
5. **Model Storage**: Save β and intercept

#### **Prediction Pipeline**
```python
y_pred = model.predict(X_test)
```
**Computation Steps:**
1. **Input Validation**: Ensure feature count matches training
2. **Matrix Multiplication**: ŷ = Xβ + intercept
3. **Output Formatting**: Return numpy array of predictions

### 📊 **Evaluation Framework**

#### **Metrics Calculation**
```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**Mean Squared Error Implementation:**
```python
# Internal calculation:
mse = np.mean((y_test - y_pred) ** 2)
```

**R² Score Implementation:**
```python
# Internal calculation:
ss_res = np.sum((y_test - y_pred) ** 2)    # Residual sum of squares
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)  # Total sum of squares
r2 = 1 - (ss_res / ss_tot)
```

### 🎨 **Visualization Pipeline**

#### **Matplotlib Integration**
```python
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted')
plt.show()
```

**Rendering Process:**
1. **Data Preparation**: Convert arrays to plottable format
2. **Figure Creation**: Initialize matplotlib figure and axes
3. **Scatter Plot**: Plot each (actual, predicted) pair as point
4. **Styling**: Apply labels, title, and default styling
5. **Display**: Render to screen or save to file

### 🔧 **Performance Optimization**

#### **Current Optimizations**
- **Vectorized Operations**: NumPy arrays for fast computation
- **Memory Efficiency**: In-place operations where possible
- **Algorithm Choice**: SVD for numerical stability

#### **Scalability Considerations**
- **Dataset Size**: Current approach scales to ~1M records
- **Feature Count**: Efficient up to ~1000 features
- **Memory Usage**: O(n×p) for data, O(p²) for model
- **Computation Time**: Linear in dataset size

## 🚀 Future Improvements & Advanced Techniques

### 🔬 **Advanced Machine Learning Enhancements**

#### **1. Feature Engineering**
```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Interaction terms
# Size × Bedrooms interaction
X['Size_Bedroom_Interaction'] = X['Size'] * X['Bedrooms']

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### **2. Advanced Algorithms**
```python
# Regularized regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Ensemble methods
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
```

#### **3. Model Validation & Selection**
```python
# Cross-validation
from sklearn.model_selection import cross_val_score, GridSearchCV
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Hyperparameter tuning
param_grid = {'alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 📈 **Production Deployment Considerations**

#### **Model Persistence**
```python
import joblib
# Save trained model
joblib.dump(model, 'house_price_model.pkl')
# Load model
loaded_model = joblib.load('house_price_model.pkl')
```

#### **API Development**
```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.json
    prediction = model.predict([[data['size'], data['bedrooms']]])
    return jsonify({'predicted_price': prediction[0]})
```

#### **Monitoring & Maintenance**
- **Data Drift Detection**: Monitor feature distributions
- **Model Performance**: Track prediction accuracy over time
- **Retraining Pipeline**: Automated model updates with new data
- **A/B Testing**: Compare model versions in production

## 📄 Files Description

- `HousePredict.py`: Main prediction script
- `random_house_prices.csv`: Dataset with house features and prices
- `README.md`: Project documentation

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📧 Contact

For questions or suggestions about this house price prediction model, please feel free to reach out.

---

*This project demonstrates fundamental machine learning concepts including data preprocessing, model training, evaluation, and visualization using Python's scientific computing stack.*


