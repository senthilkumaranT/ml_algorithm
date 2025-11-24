# Logistic Regression

## Overview
This folder contains a comprehensive implementation of Logistic Regression covering multiple scenarios: binary classification, multiclass classification, handling imbalanced datasets, and advanced evaluation techniques including ROC curves and hyperparameter tuning.

## Algorithm Description
**Logistic Regression** is a statistical model that uses a logistic function (sigmoid function) to model a binary dependent variable. Despite its name, it's actually a classification algorithm, not a regression algorithm.

### Mathematical Foundation
The logistic function (sigmoid) transforms any real-valued number into a value between 0 and 1:

**σ(z) = 1 / (1 + e^(-z))**

where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

### Key Characteristics
- **Probabilistic Output**: Provides probability estimates for class membership
- **Linear Decision Boundary**: Creates a linear separation between classes
- **Regularization Support**: Can use L1, L2, or Elastic Net regularization
- **Multiclass Support**: Can handle multiple classes using One-vs-Rest (OvR) or multinomial strategies

## Implementation Scenarios

### 1. Binary Classification

#### Dataset
- **Type**: Synthetic classification dataset
- **Samples**: 1,000 data points
- **Features**: 10 features per sample
- **Classes**: 2 classes (binary classification)
- **Generation**: Created using `make_classification` from scikit-learn

#### Implementation Steps
1. Generate synthetic binary classification dataset
2. Split data: 80% training, 20% testing
3. Train Logistic Regression model with default parameters
4. Make predictions on test set
5. Evaluate using multiple metrics

#### Results
- **Accuracy**: 91%
- **Evaluation Metrics**:
  - Accuracy score
  - Confusion matrix (True Positives, False Positives, True Negatives, False Negatives)
  - Classification report (Precision, Recall, F1-score for each class)

### 2. Hyperparameter Tuning

#### Grid Search Cross-Validation (GridSearchCV)
- **Method**: Exhaustive search over specified parameter values
- **Cross-Validation**: StratifiedKFold with 2 folds
- **Parameters Tested**:
  - `solver`: ['lbfgs', 'liblinear']
  - `penalty`: ['l2']
  - `C`: [0.001, 0.01, 0.1, 1, 10, 100] (inverse of regularization strength)
- **Result**: 90% accuracy with optimized parameters
- **Best Parameters**: Found through exhaustive search

#### Randomized Search Cross-Validation (RandomizedSearchCV)
- **Method**: Random sampling of parameter combinations
- **Advantage**: More efficient than exhaustive grid search
- **Parameters Tested**: Similar to GridSearchCV but with random sampling
- **Result**: 91% accuracy
- **Efficiency**: Faster than GridSearchCV for large parameter spaces

### 3. Multiclass Classification

#### Strategy: One-vs-Rest (OvR)
- **Approach**: Trains one binary classifier per class
- **For 3 classes**: Creates 3 binary classifiers
  - Classifier 1: Class 0 vs. (Class 1 and Class 2)
  - Classifier 2: Class 1 vs. (Class 0 and Class 2)
  - Classifier 3: Class 2 vs. (Class 0 and Class 1)
- **Prediction**: Selects the class with the highest probability

#### Dataset
- **Type**: Synthetic 3-class classification dataset
- **Samples**: Multiple samples across 3 classes
- **Configuration**: `multi_class="ovr"` parameter

#### Results
- **Accuracy**: 59%
- **Note**: Lower accuracy expected for multiclass problems compared to binary

### 4. Handling Imbalanced Datasets

#### Problem
- **Dataset Distribution**: Highly imbalanced
  - Class 0: 99% of samples
  - Class 1: 1% of samples
- **Challenge**: Standard models tend to predict the majority class, ignoring the minority class

#### Solution: Class Weight Balancing
- **Approach**: Adjust class weights during training
- **GridSearchCV Configuration**:
  - **Penalties**: ['l1', 'l2', 'elasticnet']
  - **Solvers**: Multiple solver options
  - **C Values**: Various regularization strengths
  - **Class Weights**: 
    - 'balanced': Automatically adjusts weights inversely proportional to class frequencies
    - Custom weight dictionaries
    - None: Equal weights

#### Results
- **Accuracy**: 98.85%
- **Improved Recall**: Better performance on minority class
- **Balanced Performance**: Model now considers both classes appropriately

### 5. ROC Curve and AUC Score

#### ROC (Receiver Operating Characteristic) Curve
- **Purpose**: Visualizes the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR)
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity/Recall)
- **Diagonal Line**: Represents random guessing (AUC = 0.5)

#### AUC (Area Under Curve) Score
- **Range**: 0 to 1
- **Interpretation**:
  - 0.5: Random classifier (no discriminative ability)
  - 0.7-0.8: Acceptable
  - 0.8-0.9: Excellent
  - 0.9-1.0: Outstanding
- **Result**: 0.9645 (excellent discrimination ability)

#### Comparison
- **Model Performance**: Compared against a dummy baseline classifier
- **Visualization**: ROC curve plotted for both model and baseline

## Key Features

### Algorithms Implemented
- Binary classification
- Multiclass classification (One-vs-Rest)
- Imbalanced dataset handling
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- ROC curve analysis
- Comprehensive evaluation metrics

### Evaluation Metrics Used
- **Accuracy**: Overall correctness
- **Precision**: Proportion of positive predictions that are correct
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC AUC Score**: Area under the ROC curve

### Regularization Techniques
- **L1 Regularization (Lasso)**: Encourages sparsity, feature selection
- **L2 Regularization (Ridge)**: Prevents overfitting, smooths coefficients
- **Elastic Net**: Combination of L1 and L2 regularization

### Solvers Available
- **lbfgs**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno (good for small datasets)
- **liblinear**: Library for large linear classification (good for large datasets)
- **sag**: Stochastic Average Gradient descent
- **saga**: Extension of sag that supports L1 regularization

## Results Summary

| Scenario | Accuracy | Key Metric | Notes |
|----------|----------|------------|-------|
| Binary Classification | 91% | Accuracy | Standard implementation |
| GridSearchCV | 90% | Accuracy | Optimized hyperparameters |
| RandomizedSearchCV | 91% | Accuracy | Efficient optimization |
| Multiclass (OvR) | 59% | Accuracy | 3-class problem |
| Imbalanced Dataset | 98.85% | Accuracy + Recall | Class weight balancing |
| ROC Analysis | - | AUC: 0.9645 | Excellent discrimination |

## Files
- `logistic_regression_.ipynb` - Jupyter notebook containing all implementations:
  - Binary classification
  - Hyperparameter tuning
  - Multiclass classification
  - Imbalanced dataset handling
  - ROC curve analysis

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
1. Open the Jupyter notebook: `logistic_regression_.ipynb`
2. Navigate through different sections:
   - Binary classification
   - Hyperparameter tuning
   - Multiclass classification
   - Imbalanced dataset handling
   - ROC curve analysis
3. Run cells sequentially within each section

## Advantages of Logistic Regression
- **Interpretable**: Coefficients can be interpreted as log-odds
- **Probabilistic Output**: Provides probability estimates
- **Fast Training**: Efficient optimization algorithms
- **No Feature Scaling Required**: For most solvers
- **Regularization Support**: Prevents overfitting
- **Multiclass Support**: Can handle multiple classes

## Disadvantages of Logistic Regression
- **Linear Decision Boundary**: Cannot capture non-linear relationships
- **Assumes Linear Relationship**: Between features and log-odds
- **Sensitive to Outliers**: Extreme values can affect the model
- **Requires Large Sample Size**: For stable estimates

## When to Use Logistic Regression
- Binary or multiclass classification
- When you need probability estimates
- When interpretability is important
- As a baseline model
- When features have a linear relationship with the target
- For real-time prediction (fast inference)

## Future Enhancements
- Multinomial logistic regression
- Feature importance analysis
- Cross-validation for all scenarios
- Visualization of decision boundaries
- Handling of categorical features
- Feature selection techniques
- Model interpretation (coefficient analysis)

## References
- Scikit-learn Documentation: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- "An Introduction to Statistical Learning" by James et al.
- "Pattern Recognition and Machine Learning" by Christopher Bishop
