# Polynomial Regression

## Overview
This folder contains a complete implementation of Polynomial Regression, an extension of linear regression that can model non-linear relationships by adding polynomial features. The implementation demonstrates how polynomial regression can capture curved relationships that simple linear regression cannot.

## Algorithm Description
**Polynomial Regression** is a form of regression analysis in which the relationship between the independent variable (x) and the dependent variable (y) is modeled as an nth-degree polynomial. It's essentially linear regression with polynomial features.

### Mathematical Foundation
Polynomial Regression models the relationship as:

**y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε**

Where:
- **y**: Dependent variable (target)
- **x**: Independent variable (feature)
- **β₀, β₁, β₂, ..., βₙ**: Coefficients
- **n**: Degree of the polynomial
- **ε**: Error term

### Key Characteristics
- **Non-linear Relationships**: Can capture curved relationships
- **Flexible**: Can model various curve shapes
- **Extension of Linear Regression**: Uses same optimization techniques
- **Feature Engineering**: Creates polynomial features from original features
- **Degree Control**: Can adjust complexity through polynomial degree

## Implementation Details

### 1. Data Generation

#### Synthetic Dataset
Created a synthetic dataset with a known polynomial relationship:
- **Samples**: 100 data points
- **Features**: 1 feature (x)
- **Target**: y with quadratic relationship
- **Relationship**: y = 0.5x² + 1.5x + 2 + noise
- **Noise**: Added random noise for realism
- **Range**: x values from -3 to 3

#### Dataset Characteristics
- **Non-linear Relationship**: Quadratic (degree 2) relationship
- **Controlled Complexity**: Known underlying pattern
- **Noise Added**: Realistic random variation
- **Visualizable**: 2D data for easy visualization

### 2. Data Visualization

#### Initial Scatter Plot
- Created scatter plot of the generated data
- **X-axis**: Feature values (x)
- **Y-axis**: Target values (y)
- **Color**: Green points
- **Purpose**: Visualize the non-linear relationship
- **Observation**: Clear curved (quadratic) pattern visible

### 3. Data Preprocessing

#### Train-Test Split
- **Split Ratio**: 80% training (80 samples), 20% testing (20 samples)
- **Random State**: 42 (for reproducibility)
- **Purpose**: Evaluate model on unseen data

### 4. Model Training

#### Step 1: Linear Regression Baseline
- **Model**: `LinearRegression()` from scikit-learn
- **Purpose**: Establish baseline with simple linear regression
- **Training**: Trained on original features (no polynomial transformation)
- **Visualization**: 
  - Plotted training data points
  - Plotted linear regression line (red)
  - **Observation**: Linear model cannot capture the curved relationship

#### Step 2: Polynomial Feature Transformation
- **Method**: `PolynomialFeatures` from scikit-learn
- **Degree**: 2 (quadratic)
- **Include Bias**: True
- **Process**:
  1. Transform training features: `X_train_poly = poly.fit_transform(X_train)`
  2. Transform test features: `X_test_poly = poly.transform(X_test)`
- **Result**: Creates polynomial features (x, x², ...)

#### Polynomial Features Created
For degree=2, the transformation creates:
- **Original**: x
- **Polynomial**: [1, x, x²]
  - 1: Bias term (if include_bias=True)
  - x: Original feature
  - x²: Squared feature

#### Step 3: Polynomial Regression
- **Model**: `LinearRegression()` on polynomial features
- **Training**: Trained on transformed polynomial features
- **Result**: Model can now capture the quadratic relationship

### 5. Model Visualization

#### Training Data Visualization
- **Scatter Plot**: Training data points
- **Polynomial Fit**: Predicted values from polynomial regression
- **Purpose**: Visualize how well polynomial regression fits the training data
- **Observation**: Polynomial regression captures the curved relationship

## Key Features

### Polynomial Feature Engineering
- **PolynomialFeatures**: Transforms features into polynomial features
- **Degree Control**: Can adjust polynomial degree
- **Bias Term**: Option to include or exclude bias term
- **Feature Transformation**: Applied consistently to train and test sets

### Model Comparison
- **Linear Regression**: Baseline model (cannot capture curves)
- **Polynomial Regression**: Extended model (can capture curves)
- **Visual Comparison**: Side-by-side visualization

### Implementation Approach
1. Generate synthetic data with known polynomial relationship
2. Try linear regression (baseline)
3. Transform features to polynomial features
4. Train polynomial regression
5. Visualize and compare results

## Files
- `polynomial_regression.py` - Python script containing:
  - Synthetic data generation
  - Data visualization
  - Linear regression baseline
  - Polynomial feature transformation
  - Polynomial regression training
  - Model visualization

## Dependencies
```python
numpy       # Numerical computing
pandas      # Data manipulation (if needed)
scikit-learn # Machine learning library
matplotlib  # Data visualization
```

## Installation
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

## Code Example

Here's the complete code implementation:

```python
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data with quadratic relationship
x = 6 * np.random.rand(100, 1) - 3  # x values from -3 to 3
y = 0.5 * x**2 + 1.5*x + 2 + np.random.randn(100, 1)  # y = 0.5x² + 1.5x + 2 + noise

# Visualize original data
plt.scatter(x, y, color="g", label="Data Points")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data (Quadratic Relationship)')
plt.legend()
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 1: Try Linear Regression (Baseline)
regression_1 = LinearRegression()
regression_1.fit(X_train, y_train)

# Visualize linear regression fit
plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_train, regression_1.predict(X_train), color='r', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression (Baseline)')
plt.legend()
plt.show()

# Step 2: Create Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 3: Train Polynomial Regression
regression = LinearRegression()
regression.fit(X_train_poly, y_train)

# Make predictions
y_pred = regression.predict(X_test_poly)

# Visualize polynomial regression fit
plt.scatter(X_train, y_train, label="Training Data", alpha=0.6)
plt.scatter(X_train, regression.predict(X_train_poly), 
           color='blue', label='Polynomial Regression', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression (Degree=2)')
plt.legend()
plt.show()

# Display model coefficients
print("Polynomial Regression Coefficients:")
print(f"Intercept (β₀): {regression.intercept_[0]:.4f}")
print(f"Coefficients: {regression.coef_[0]}")
```

### Code Explanation

**Step 1: Generate Synthetic Data**
```python
x = 6 * np.random.rand(100, 1) - 3
y = 0.5 * x**2 + 1.5*x + 2 + np.random.randn(100, 1)
```
- Creates 100 data points with x values from -3 to 3
- Generates y using quadratic relationship: y = 0.5x² + 1.5x + 2 + noise
- Noise makes the problem more realistic

**Step 2: Baseline - Linear Regression**
```python
regression_1 = LinearRegression()
regression_1.fit(X_train, y_train)
```
- Trains simple linear regression as baseline
- Cannot capture the quadratic relationship
- Shows underfitting on non-linear data

**Step 3: Polynomial Feature Transformation**
```python
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```
- Transforms features: [x] → [1, x, x²]
- Creates polynomial features up to degree 2
- Must transform both training and test sets

**Step 4: Polynomial Regression**
```python
regression = LinearRegression()
regression.fit(X_train_poly, y_train)
```
- Trains linear regression on polynomial features
- Now can capture quadratic relationship
- Model learns: y = β₀ + β₁x + β₂x²

**Step 5: Visualization and Evaluation**
- Compares linear vs polynomial regression visually
- Shows how polynomial regression captures the curve
- Displays learned coefficients

## Usage
1. Open the Python file: `polynomial_regression.py`
2. Run the script
3. The script will:
   - Generate synthetic data with quadratic relationship
   - Visualize the data
   - Train linear regression (baseline)
   - Transform features to polynomial features
   - Train polynomial regression
   - Visualize both models

## Advantages of Polynomial Regression
- **Handles Non-linearity**: Can capture curved relationships
- **Flexible**: Can model various curve shapes
- **Simple Extension**: Builds on linear regression
- **Interpretable**: Still relatively interpretable
- **Widely Applicable**: Useful for many real-world problems

## Disadvantages of Polynomial Regression
- **Overfitting Risk**: High-degree polynomials can overfit
- **Sensitive to Outliers**: Extreme values can significantly affect the model
- **Feature Explosion**: Number of features grows rapidly with degree
- **Extrapolation Issues**: Poor performance outside training range
- **Computational Cost**: Higher degree = more computation

## When to Use Polynomial Regression
- Non-linear relationships between features and target
- When linear regression underperforms
- When you suspect polynomial relationships
- For smooth, continuous curves
- When you need interpretable non-linear models
- As an alternative to more complex models

## Polynomial Degree Selection

### Low Degree (1-2)
- **Use**: Simple curves, slight non-linearity
- **Pros**: Less overfitting, more interpretable
- **Cons**: May underfit complex relationships

### Medium Degree (3-5)
- **Use**: Moderate non-linearity
- **Pros**: Good balance
- **Cons**: Risk of overfitting

### High Degree (6+)
- **Use**: Complex curves
- **Pros**: Can capture very complex patterns
- **Cons**: High overfitting risk, computational cost

### Selection Strategy
1. Start with degree 2 (quadratic)
2. Increase if underfitting
3. Decrease if overfitting
4. Use cross-validation to select optimal degree

## Overfitting Prevention

### Techniques
1. **Cross-Validation**: Use CV to select optimal degree
2. **Regularization**: Add L1/L2 regularization
3. **Early Stopping**: Stop before overfitting
4. **Feature Selection**: Remove unnecessary polynomial terms
5. **More Data**: Collect more training samples

### Signs of Overfitting
- Training accuracy much higher than test accuracy
- Model follows training data too closely
- Poor generalization to new data
- High variance in predictions

## Comparison with Other Methods

### vs. Linear Regression
- **Polynomial**: Can capture curves
- **Linear**: Only straight lines
- **Trade-off**: More complexity vs. simplicity

### vs. Other Non-linear Methods
- **Polynomial**: Interpretable, simple
- **SVR with RBF**: More flexible, less interpretable
- **Neural Networks**: Very flexible, black box
- **Decision Trees**: Non-parametric, different approach

## Future Enhancements
- Cross-validation for degree selection
- Regularization (Ridge, Lasso)
- Multiple features (multivariate polynomial)
- Model evaluation metrics (MSE, MAE, RMSE, R²)
- Residual analysis
- Comparison with other non-linear methods
- Real-world dataset application
- Hyperparameter tuning for degree
- Visualization improvements
- Jupyter notebook version

## Mathematical Details

### Polynomial Features
For a single feature x and degree d:
- **Features**: [1, x, x², x³, ..., xᵈ]
- **Number of Features**: d + 1 (if include_bias=True)

For multiple features, the number grows combinatorially:
- **2 features, degree 2**: [1, x₁, x₂, x₁², x₁x₂, x₂²]
- **Formula**: (n + d)! / (n! × d!) where n = number of features

### Optimization
- **Method**: Same as linear regression (OLS)
- **Solution**: Closed-form or gradient descent
- **Complexity**: O(n × m²) where n = samples, m = polynomial features

## References
- Scikit-learn Documentation: 
  - [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
  - [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- "An Introduction to Statistical Learning" by James et al.
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman





