# Support Vector Machine Regression (SVR)

## Overview
This folder contains an implementation of Support Vector Regression (SVR) using the tips dataset to predict total bill amounts based on various features.

## Technique Used
**Support Vector Regression (SVR)** - A regression algorithm that uses the same principles as Support Vector Machines (SVM) but for regression tasks. SVR finds a function that approximates the mapping from inputs to real-valued outputs while maintaining all the main features of SVM.

## What We Did

### 1. Data Exploration
- Loaded the tips dataset from seaborn (244 samples)
- Explored dataset structure with 7 features:
  - `total_bill` (target variable)
  - `tip`
  - `sex` (categorical)
  - `smoker` (categorical)
  - `day` (categorical)
  - `time` (categorical)
  - `size`

### 2. Feature Engineering
- Selected features: `tip`, `sex`, `smoker`, `day`, `time`, `size`
- **Label Encoding**: Applied to categorical features (`sex`, `smoker`, `time`)
- **One-Hot Encoding**: Applied to `day` feature using `OneHotEncoder` with `drop='first'`
- Used `ColumnTransformer` for efficient preprocessing

### 3. Model Training
- Split data into training (80%) and testing (20%) sets
- Trained Support Vector Regression model with default parameters
- Evaluated using R² score (coefficient of determination)

### 4. Hyperparameter Tuning
- Performed **GridSearchCV** with 5-fold cross-validation
- Tested different parameter combinations:
  - **Kernels**: linear, poly, rbf, sigmoid
  - **C values**: 1, 5, 10 (regularization parameter)
  - **Gamma**: 'scale', 'auto' (kernel coefficient)
- Found optimal hyperparameters for best R² score

## Key Features
- Categorical feature encoding (Label Encoding and One-Hot Encoding)
- Multiple kernel support (linear, polynomial, RBF, sigmoid)
- Hyperparameter optimization using GridSearchCV
- R² score evaluation for regression performance

## Files
- `Support_vector_machine_Regression.ipynb` - Jupyter notebook containing the complete implementation

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

