# Support Vector Machine Classification (SVC)

## Overview
This folder contains an implementation of Support Vector Classification (SVC) with multiple kernel functions and hyperparameter tuning.

## Technique Used
**Support Vector Classification (SVC)** - A supervised learning algorithm used for classification tasks. SVC finds the optimal hyperplane that separates classes in the feature space. It can use different kernel functions to handle non-linear classification problems.

## What We Did

### 1. Data Generation
- Created a synthetic binary classification dataset with:
  - 1000 samples
  - 2 features
  - 2 classes
  - 2 clusters per class
  - No redundant features
- Visualized the dataset using scatter plots

### 2. Model Training with Different Kernels
Tested four different kernel functions:
- **Linear Kernel**: Achieved 86.5% accuracy
  - Best for linearly separable data
- **RBF (Radial Basis Function) Kernel**: Achieved 90% accuracy
  - Best overall performance
  - Handles non-linear boundaries effectively
- **Polynomial Kernel**: Achieved 87% accuracy
  - Good for polynomial decision boundaries
- **Sigmoid Kernel**: Tested but used polynomial predictions (87% accuracy)
  - Less commonly used, similar to neural network activation

### 3. Hyperparameter Tuning
- Performed **GridSearchCV** with 5-fold cross-validation
- Tested parameter combinations:
  - **C values**: [0.1, 1, 10, 100] (regularization parameter)
  - **Gamma values**: [1, 0.1, 0.01, 0.001] (kernel coefficient)
  - **Kernel**: RBF (focus on best performing kernel)
- Found optimal hyperparameters for maximum accuracy
- Achieved **90% accuracy** with tuned parameters

### 4. Model Evaluation
- Calculated accuracy score
- Generated classification report with:
  - Precision
  - Recall
  - F1-score
  - Support for each class
- Created confusion matrix for performance visualization

## Results
- **Best Kernel**: RBF (Radial Basis Function)
- **Best Accuracy**: 90%
- **Performance Metrics**:
  - Precision: 0.87 (macro avg)
  - Recall: 0.87 (macro avg)
  - F1-score: 0.86 (macro avg)

## Key Features
- Multiple kernel function comparison (linear, RBF, polynomial, sigmoid)
- Hyperparameter optimization using GridSearchCV
- Comprehensive evaluation metrics
- Data visualization

## Files
- `Support_vector_machine_classfication.ipynb` - Jupyter notebook containing the complete implementation

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

