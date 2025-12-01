# Multiple Linear Regression

## Overview
This folder contains a comprehensive implementation of Multiple Linear Regression, an extension of simple linear regression that handles multiple independent variables. The implementation uses an economic index dataset to predict index prices based on multiple economic indicators.

## Algorithm Description
**Multiple Linear Regression** extends simple linear regression to model the relationship between a dependent variable and multiple independent variables. It's one of the most widely used regression techniques in statistics and machine learning.

### Mathematical Foundation
Multiple Linear Regression models the relationship as:

**y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε**

Where:
- **y**: Dependent variable (target)
- **x₁, x₂, ..., xₙ**: Independent variables (features)
- **β₀**: Y-intercept (bias term)
- **β₁, β₂, ..., βₙ**: Coefficients (slopes)
- **ε**: Error term

### Key Characteristics
- **Multiple Features**: Can handle multiple independent variables simultaneously
- **Linear Relationship**: Assumes linear relationships between each feature and target
- **Interpretable**: Coefficients show the effect of each feature
- **Feature Interaction**: Can capture relationships between multiple features
- **Widely Applicable**: Used in various domains (economics, science, engineering)

## Dataset: Economic Index

### Dataset Overview
- **File**: `economic_index.csv`
- **Domain**: Economic indicators and index prices
- **Purpose**: Predict index prices based on economic indicators
- **Features**: Multiple economic indicators
- **Target**: Index price

### Dataset Structure
The dataset contains various economic indicators and an index price:
- **Features (Independent Variables)**: Multiple economic indicators such as:
  - Interest rates
  - Unemployment rates
  - Stock market indicators
  - Other economic metrics
- **Target (Dependent Variable)**: Index price
- **Additional Columns**: 
  - `Unnamed: 0`: Index column (removed)
  - `year`: Year column (removed)
  - `month`: Month column (removed)

### Data Preprocessing
- **Column Removal**: Dropped unnecessary columns:
  - `Unnamed: 0`: Index column
  - `year`: Year information
  - `month`: Month information
- **Purpose**: Focus on relevant economic indicators

## Implementation Details

### 1. Data Loading and Exploration

#### Data Loading
- Loaded dataset from CSV file
- Examined dataset structure and basic information
- Checked for missing values

#### Data Exploration
- **Dataset Head**: Examined first few rows
- **Missing Values**: Checked for null values
- **Data Types**: Analyzed column types
- **Dataset Shape**: Understood data dimensions

### 2. Data Visualization

#### Pair Plot
- Created pair plot using seaborn
- **Purpose**: Visualize relationships between all variables
- **Benefits**: 
  - Shows pairwise relationships
  - Displays distributions
  - Identifies correlations visually
  - Helps understand data structure

#### Correlation Analysis
- Calculated correlation matrix
- **Purpose**: Quantify linear relationships between variables
- **Interpretation**: 
  - High positive correlation: Variables move together
  - High negative correlation: Variables move in opposite directions
  - Low correlation: Weak relationship
- **Usage**: Identifies which features are most related to the target

### 3. Data Preprocessing

#### Feature and Target Selection
- **Features (X)**: All columns except the last one (all economic indicators)
  - Method: `df.iloc[:,:-1]` (all rows, all columns except last)
- **Target (y)**: Last column (index price)
  - Method: `df.iloc[:,-1]` (all rows, last column)

#### Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Random State**: 0 (for reproducibility)
- **Purpose**: Evaluate model on unseen data

#### Feature Scaling
- Applied **StandardScaler** for feature standardization
- **Formula**: (x - mean) / std
- **Purpose**: 
  - Ensures all features are on the same scale
  - Prevents features with larger scales from dominating
  - Improves numerical stability
  - Required for proper coefficient interpretation
- **Process**:
  - Fit scaler on training data only
  - Transform both training and test data
  - Prevents data leakage

### 4. Model Training

#### Model Initialization
- **Algorithm**: `LinearRegression()` from scikit-learn
- **Method**: Ordinary Least Squares (OLS)
- **Optimization**: Closed-form solution (normal equation) or gradient descent

#### Model Fitting
- Trained model on scaled training data
- Model learns:
  - **Intercept (β₀)**: Base value when all features are zero
  - **Coefficients (β₁, β₂, ..., βₙ)**: Effect of each feature on target

#### Cross-Validation
- Performed **5-fold cross-validation**
- **Metric**: Negative Mean Squared Error
- **Purpose**: 
  - Assess model stability
  - Evaluate generalization ability
  - More robust than single train-test split
- **Result**: Cross-validation scores for model evaluation

### 5. Model Evaluation

#### Predictions
- Made predictions on test data
- Generated predicted index prices for test economic indicators

#### Evaluation Metrics

##### Mean Squared Error (MSE)
- **Formula**: (1/n) × Σ(y_actual - y_predicted)²
- **Result**: 7720.06
- **Interpretation**: Average squared difference between actual and predicted index prices
- **Units**: Squared units of index price

##### Mean Absolute Error (MAE)
- **Formula**: (1/n) × Σ|y_actual - y_predicted|
- **Result**: 69.90
- **Interpretation**: Average absolute difference between actual and predicted index prices
- **Units**: Same units as index price
- **Advantage**: More interpretable, less sensitive to outliers

##### Root Mean Squared Error (RMSE)
- **Formula**: √MSE
- **Result**: 87.86
- **Interpretation**: Standard deviation of prediction errors
- **Units**: Same units as index price
- **Advantage**: Same scale as target, easier to interpret

### 6. Model Diagnostics

#### Residual Analysis
- **Residuals**: Differences between actual and predicted values
- **Formula**: residuals = y_test - y_pred
- **Purpose**: Check model assumptions and identify patterns

#### Residual Visualization

##### Scatter Plot: Actual vs. Predicted
- **X-axis**: Predicted values (y_pred)
- **Y-axis**: Actual values (y_test)
- **Purpose**: Visualize prediction accuracy
- **Ideal Pattern**: Points should lie close to diagonal line (y=x)

##### Residual Distribution
- **Plot Type**: Kernel Density Estimation (KDE) plot
- **Purpose**: Check if residuals are normally distributed
- **Ideal Pattern**: Bell-shaped (normal) distribution
- **Interpretation**: 
  - Normal distribution: Assumption met
  - Skewed distribution: May indicate issues

##### Residual vs. Predicted Plot
- **X-axis**: Predicted values (y_pred)
- **Y-axis**: Residuals (y_test - y_pred)
- **Purpose**: Check homoscedasticity (constant variance)
- **Ideal Pattern**: Random scatter with no pattern
- **Interpretation**: 
  - Random scatter: Homoscedasticity (good)
  - Funnel shape: Heteroscedasticity (variance changes)

## Key Features

### Data Preprocessing
- Column removal (unnecessary features)
- Feature and target separation
- Train-test splitting
- Feature scaling (StandardScaler)
- Comprehensive data preparation

### Model Development
- Multiple linear regression implementation
- Cross-validation for robust evaluation
- Training and prediction

### Evaluation and Diagnostics
- Multiple evaluation metrics (MSE, MAE, RMSE)
- Residual analysis
- Comprehensive visualization
- Assumption checking

## Files
- `multiple_linear_regression.ipynb` - Jupyter notebook containing:
  - Data loading and exploration
  - Data visualization (pair plot, correlation)
  - Data preprocessing
  - Model training with cross-validation
  - Model evaluation
  - Residual analysis and diagnostics
- `dataset/economic_index.csv` - The economic index dataset

## Dependencies
```python
pandas      # Data manipulation
numpy       # Numerical computing
scikit-learn # Machine learning library
matplotlib  # Data visualization
seaborn     # Statistical visualization
```

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Code Example

Here's the complete code implementation:

```python
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv('dataset/economic_index.csv')

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0", "year", "month"], axis=1, inplace=True)

# Check for missing values
print(df.isnull().sum())

# Visualize relationships
sns.pairplot(df)
plt.show()

# Check correlations
correlation_matrix = df.corr()
print(correlation_matrix)

# Prepare features and target
x = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column (target)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Standardize features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize and train model
repr = LinearRegression()
repr.fit(x_train, y_train)

# Cross-validation
cv_scores = cross_val_score(repr, x_train, y_train, 
                           scoring="neg_mean_squared_error", cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

# Make predictions
y_pred = repr.predict(x_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Residual analysis
residuals = y_test - y_pred

# Visualize actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()

# Residual distribution
sns.displot(residuals, kind="kde")
plt.title('Residual Distribution')
plt.show()

# Residual vs predicted plot
sns.scatterplot(x=y_pred, y=residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

### Code Explanation

**Step 1: Data Loading and Cleaning**
```python
df = pd.read_csv('dataset/economic_index.csv')
df.drop(columns=["Unnamed: 0", "year", "month"], axis=1, inplace=True)
```
- Loads economic index dataset
- Removes unnecessary columns (index, year, month)

**Step 2: Data Exploration**
```python
sns.pairplot(df)
df.corr()
```
- Creates pair plot to visualize all variable relationships
- Calculates correlation matrix

**Step 3: Feature and Target Separation**
```python
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
```
- Separates all features (except last column) and target (last column)

**Step 4: Train-Test Split and Scaling**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
- Splits data: 80% training, 20% testing
- Standardizes features for better model performance

**Step 5: Model Training and Cross-Validation**
```python
repr = LinearRegression()
repr.fit(x_train, y_train)
cv_scores = cross_val_score(repr, x_train, y_train, scoring="neg_mean_squared_error", cv=5)
```
- Trains multiple linear regression model
- Performs 5-fold cross-validation for robust evaluation

**Step 6: Prediction and Evaluation**
```python
y_pred = repr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
```
- Makes predictions and calculates multiple metrics

**Step 7: Residual Analysis**
```python
residuals = y_test - y_pred
sns.displot(residuals, kind="kde")
sns.scatterplot(x=y_pred, y=residuals)
```
- Analyzes residuals to check model assumptions
- Visualizes residual distribution and patterns

## Usage
1. Ensure the dataset file `economic_index.csv` is in the `dataset/` folder
2. Open the Jupyter notebook: `multiple_linear_regression.ipynb`
3. Run all cells sequentially
4. The notebook will:
   - Load and explore the dataset
   - Visualize relationships (pair plot, correlation)
   - Preprocess the data
   - Train the multiple linear regression model
   - Perform cross-validation
   - Make predictions
   - Evaluate model performance
   - Analyze residuals

## Advantages of Multiple Linear Regression
- **Handles Multiple Features**: Can model complex relationships
- **Interpretable**: Coefficients show feature importance
- **Fast Training**: Efficient optimization algorithms
- **Well-Established**: Extensive statistical theory
- **Baseline Model**: Good starting point for regression
- **Feature Relationships**: Can capture interactions between features

## Disadvantages of Multiple Linear Regression
- **Linear Assumption**: Assumes linear relationships
- **Multicollinearity**: Correlated features can cause issues
- **Sensitive to Outliers**: Extreme values affect the model
- **Assumes Independence**: Features should be independent
- **May Underfit**: Too simple for complex non-linear relationships
- **Curse of Dimensionality**: Performance may degrade with many features

## Assumptions of Multiple Linear Regression
1. **Linearity**: Linear relationship between features and target
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed (for inference)
5. **No Multicollinearity**: Features are not highly correlated
6. **No Autocorrelation**: Errors are not correlated (for time series)

## When to Use Multiple Linear Regression
- Predicting continuous numerical values
- When you have multiple relevant features
- When relationships appear linear
- As a baseline model
- When interpretability is important
- For economic, scientific, or engineering applications
- When you need to understand feature contributions

## Evaluation Metrics Explained

### Mean Squared Error (MSE)
- **Use**: Standard regression metric
- **Pros**: Penalizes large errors more
- **Cons**: Sensitive to outliers, squared units
- **Best For**: When large errors are costly

### Mean Absolute Error (MAE)
- **Use**: Alternative to MSE
- **Pros**: Less sensitive to outliers, same units
- **Cons**: Doesn't penalize large errors as much
- **Best For**: When all errors are equally important

### Root Mean Squared Error (RMSE)
- **Use**: Most interpretable version of MSE
- **Pros**: Same units as target, penalizes large errors
- **Cons**: Still sensitive to outliers
- **Best For**: General-purpose evaluation (recommended)

### R² Score (Coefficient of Determination)
- **Use**: Proportion of variance explained
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: 
  - 1.0: Perfect predictions
  - 0.0: Model performs as well as predicting the mean
  - Negative: Model performs worse than mean prediction

## Residual Analysis Importance

### Why Analyze Residuals?
1. **Check Assumptions**: Verify model assumptions are met
2. **Identify Patterns**: Detect non-linear relationships
3. **Detect Outliers**: Find influential data points
4. **Improve Model**: Guide model improvements

### Residual Patterns to Look For
- **Random Scatter**: Good (assumptions met)
- **Curved Pattern**: Non-linearity (may need polynomial features)
- **Funnel Shape**: Heteroscedasticity (variance changes)
- **Trend**: Missing important features

## Future Enhancements
- R² score calculation
- Feature importance analysis
- Multicollinearity detection (VIF)
- Regularization (Ridge, Lasso, Elastic Net)
- Feature selection techniques
- Polynomial features for non-linearity
- Interaction terms
- Outlier detection and handling
- Model interpretation (coefficient analysis)

## References
- Scikit-learn Documentation: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- "An Introduction to Statistical Learning" by James et al.
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Applied Linear Statistical Models" by Kutner et al.





