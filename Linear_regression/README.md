# Linear Regression

## Overview
This folder contains a complete implementation of Simple Linear Regression, one of the fundamental regression algorithms in machine learning. The implementation uses a height-weight dataset to predict height based on weight, demonstrating the core concepts of linear regression.

## Algorithm Description
**Linear Regression** is a fundamental supervised learning algorithm used for predicting continuous numerical values. It models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear equation.

### Mathematical Foundation
Simple Linear Regression models the relationship as:

**y = β₀ + β₁x + ε**

Where:
- **y**: Dependent variable (target)
- **x**: Independent variable (feature)
- **β₀**: Y-intercept (bias term)
- **β₁**: Slope (coefficient)
- **ε**: Error term

### Key Characteristics
- **Linear Relationship**: Assumes a linear relationship between features and target
- **Interpretable**: Coefficients have clear meaning
- **Fast Training**: Efficient optimization algorithms
- **Baseline Model**: Often used as a baseline for regression problems
- **Probabilistic Foundation**: Based on maximum likelihood estimation

## Dataset: Height-Weight

### Dataset Overview
- **File**: `height-weight.csv`
- **Samples**: 23 data points
- **Features**: 1 feature (Weight)
- **Target**: Height
- **Domain**: Physical measurements

### Dataset Structure
- **Weight**: Independent variable (feature) in some unit
- **Height**: Dependent variable (target) in some unit
- **No Missing Values**: Complete dataset
- **Small Dataset**: Suitable for learning and demonstration

### Dataset Characteristics
- **Real-world Data**: Actual height and weight measurements
- **Positive Correlation**: Height and weight typically have a positive relationship
- **Linear Relationship**: Suitable for linear regression modeling

## Implementation Details

### 1. Data Loading and Exploration

#### Data Loading
- Loaded dataset from CSV file
- Examined basic dataset information
- Checked for missing values

#### Data Exploration
- **Dataset Info**: 23 entries, 2 columns (Weight, Height)
- **Data Types**: Both columns are integers
- **Missing Values**: No missing values detected
- **Basic Statistics**: Explored data distributions

### 2. Data Visualization

#### Scatter Plot
- Created scatter plot of Height vs. Weight
- **X-axis**: Height
- **Y-axis**: Weight
- **Purpose**: Visualize the relationship between variables
- **Observation**: Positive correlation visible

#### Correlation Analysis
- Calculated correlation coefficient between Height and Weight
- **Purpose**: Quantify the linear relationship strength
- **Range**: -1 to +1
- **Interpretation**: 
  - Close to +1: Strong positive correlation
  - Close to 0: Weak correlation
  - Close to -1: Strong negative correlation

#### Pair Plot
- Created pair plot using seaborn
- **Purpose**: Comprehensive visualization of variable relationships
- **Benefits**: Shows distributions and relationships simultaneously

### 3. Data Preprocessing

#### Feature and Target Selection
- **Feature (X)**: Weight (independent variable)
- **Target (y)**: Height (dependent variable)
- Reshaped data for model compatibility

#### Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Training Samples**: ~18 samples
- **Testing Samples**: ~5 samples
- **Random State**: 0 (for reproducibility)

#### Feature Scaling
- Applied **StandardScaler** for feature standardization
- **Formula**: (x - mean) / std
- **Purpose**: 
  - Ensures features are on the same scale
  - Improves numerical stability
  - Required for some optimization algorithms
- **Process**:
  - Fit scaler on training data
  - Transform both training and test data
  - Prevents data leakage

### 4. Model Training

#### Model Initialization
- **Algorithm**: `LinearRegression()` from scikit-learn
- **Method**: Ordinary Least Squares (OLS)
- **Optimization**: Closed-form solution or gradient descent

#### Model Fitting
- Trained model on scaled training data
- Model learns:
  - **Intercept (β₀)**: Y-intercept of the regression line
  - **Coefficient (β₁)**: Slope of the regression line

#### Training Visualization
- Plotted training data points
- Plotted best-fit line (regression line)
- **Color Coding**: 
  - Red: Training data points
  - Blue: Regression line
- **Purpose**: Visualize how well the model fits the training data

### 5. Model Evaluation

#### Predictions
- Made predictions on test data
- Generated predicted height values for test weights

#### Evaluation Metrics

##### Mean Squared Error (MSE)
- **Formula**: (1/n) × Σ(y_actual - y_predicted)²
- **Result**: 26.78
- **Interpretation**: Average squared difference between actual and predicted values
- **Units**: Squared units of the target variable

##### Mean Absolute Error (MAE)
- **Formula**: (1/n) × Σ|y_actual - y_predicted|
- **Result**: 4.32
- **Interpretation**: Average absolute difference between actual and predicted values
- **Units**: Same units as the target variable
- **Advantage**: More interpretable than MSE

##### Root Mean Squared Error (RMSE)
- **Formula**: √MSE
- **Result**: 5.18
- **Interpretation**: Standard deviation of prediction errors
- **Units**: Same units as the target variable
- **Advantage**: Same scale as target, easier to interpret than MSE

### Performance Summary
- **MSE**: 26.78 (lower is better)
- **MAE**: 4.32 (lower is better)
- **RMSE**: 5.18 (lower is better)
- **Interpretation**: On average, predictions are off by about 5.18 units

## Key Features

### Data Preprocessing
- Missing value checking
- Train-test splitting
- Feature scaling (StandardScaler)
- Proper data preparation pipeline

### Model Development
- Simple linear regression implementation
- Training and prediction
- Model visualization

### Evaluation
- Multiple evaluation metrics (MSE, MAE, RMSE)
- Comprehensive performance assessment

## Files
- `linear_regression.ipynb` - Jupyter notebook containing:
  - Data loading and exploration
  - Data visualization
  - Data preprocessing
  - Model training
  - Model evaluation
- `dataset/height-weight.csv` - The height-weight dataset

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

## Usage
1. Ensure the dataset file `height-weight.csv` is in the `dataset/` folder
2. Open the Jupyter notebook: `linear_regression.ipynb`
3. Run all cells sequentially
4. The notebook will:
   - Load and explore the dataset
   - Visualize the data
   - Preprocess the data (scaling, splitting)
   - Train the linear regression model
   - Make predictions
   - Evaluate model performance

## Advantages of Linear Regression
- **Simple and Interpretable**: Easy to understand and explain
- **Fast Training**: Efficient optimization algorithms
- **No Hyperparameters**: Fewer parameters to tune
- **Probabilistic Foundation**: Well-established statistical theory
- **Baseline Model**: Good starting point for regression problems
- **Works Well with Linear Relationships**: Excellent when assumptions are met

## Disadvantages of Linear Regression
- **Linear Assumption**: Assumes linear relationship (may not capture non-linear patterns)
- **Sensitive to Outliers**: Extreme values can significantly affect the model
- **Assumes Independence**: Features should be independent
- **Assumes Homoscedasticity**: Constant variance of errors
- **May Underfit**: Too simple for complex relationships

## Assumptions of Linear Regression
1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed (for inference)
5. **No Multicollinearity**: Features are not highly correlated (for multiple regression)

## When to Use Linear Regression
- Predicting continuous numerical values
- When relationship appears linear
- As a baseline model
- When interpretability is important
- For simple, well-understood problems
- When you have limited data

## Evaluation Metrics Explained

### Mean Squared Error (MSE)
- **Use**: Most common metric for regression
- **Pros**: Penalizes large errors more
- **Cons**: Sensitive to outliers, units are squared
- **Best For**: When large errors are particularly costly

### Mean Absolute Error (MAE)
- **Use**: Alternative to MSE
- **Pros**: Less sensitive to outliers, same units as target
- **Cons**: Doesn't penalize large errors as much
- **Best For**: When all errors are equally important

### Root Mean Squared Error (RMSE)
- **Use**: Most interpretable version of MSE
- **Pros**: Same units as target, penalizes large errors
- **Cons**: Still sensitive to outliers
- **Best For**: General-purpose evaluation (recommended)

### R² Score (Coefficient of Determination)
- **Use**: Measures proportion of variance explained
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: 
  - 1.0: Perfect predictions
  - 0.0: Model performs as well as predicting the mean
  - Negative: Model performs worse than predicting the mean

## Future Enhancements
- R² score calculation
- Residual analysis and plots
- Assumption checking (normality, homoscedasticity)
- Confidence intervals for predictions
- Feature importance analysis
- Comparison with polynomial regression
- Cross-validation for more robust evaluation
- Handling of outliers

## References
- Scikit-learn Documentation: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- "An Introduction to Statistical Learning" by James et al.
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

