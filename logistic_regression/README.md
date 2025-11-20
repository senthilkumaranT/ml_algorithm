# Logistic Regression

## Overview
This folder contains a comprehensive implementation of Logistic Regression covering binary classification, multiclass classification, handling imbalanced datasets, and advanced evaluation techniques.

## Technique Used
**Logistic Regression** - A statistical model that uses a logistic function to model a binary dependent variable. It's a linear classification algorithm that estimates probabilities using a logistic function.

## What We Did

### 1. Binary Classification
- Created a synthetic binary classification dataset with 1000 samples and 10 features
- Split data into training (80%) and testing (20%) sets
- Trained Logistic Regression model
- Achieved **91% accuracy** on test set
- Evaluated using:
  - Accuracy score
  - Confusion matrix
  - Classification report (precision, recall, F1-score)

### 2. Hyperparameter Tuning
- **Grid Search CV**: 
  - Tested different combinations of `solver` (lbfgs, liblinear), `penalty` (l2), and `C` values (0.001 to 100)
  - Used StratifiedKFold cross-validation (2 folds)
  - Achieved 90% accuracy with optimized parameters
- **Randomized Search CV**:
  - Randomly sampled parameter combinations
  - Achieved 91% accuracy
  - More efficient than exhaustive grid search

### 3. Multiclass Classification
- Implemented One-vs-Rest (OvR) strategy for multiclass problems
- Created 3-class classification dataset
- Used `multi_class="ovr"` parameter
- Achieved 59% accuracy on 3-class problem

### 4. Handling Imbalanced Datasets
- Created highly imbalanced dataset (99% class 0, 1% class 1)
- Implemented class weight balancing in hyperparameter tuning
- Tested various class weight combinations
- Used GridSearchCV with comprehensive parameter grid including:
  - Different penalties (l1, l2, elasticnet)
  - Multiple solvers
  - Various C values
  - Class weight configurations
- Achieved 98.85% accuracy with improved recall for minority class

### 5. ROC Curve and ROC AUC Score
- Implemented ROC (Receiver Operating Characteristic) curve visualization
- Calculated ROC AUC (Area Under Curve) score
- Compared model performance against a dummy baseline
- Achieved ROC AUC score of 0.9645, demonstrating excellent model performance

## Key Features
- Binary and multiclass classification
- Hyperparameter tuning (GridSearchCV and RandomizedSearchCV)
- Handling imbalanced datasets with class weights
- ROC curve analysis and AUC score calculation
- Comprehensive evaluation metrics

## Results Summary
- **Binary Classification**: 91% accuracy
- **Multiclass Classification**: 59% accuracy
- **Imbalanced Dataset**: 98.85% accuracy with improved minority class recall
- **ROC AUC Score**: 0.9645 (excellent discrimination ability)

## Files
- `logistic_regression_.ipynb` - Jupyter notebook containing all implementations

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

