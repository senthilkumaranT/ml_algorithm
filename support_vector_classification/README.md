# Support Vector Machine Classification (SVC)

## Overview
This folder contains a comprehensive implementation of Support Vector Classification (SVC) with multiple kernel functions, hyperparameter tuning, and thorough model evaluation. The implementation demonstrates how different kernel functions affect classification performance.

## Algorithm Description
**Support Vector Classification (SVC)** is a supervised learning algorithm used for classification tasks. SVC finds the optimal hyperplane that separates classes in the feature space with the maximum margin. It can use different kernel functions to handle non-linear classification problems.

### How SVC Works
1. **Optimal Hyperplane**: Finds the decision boundary that maximizes the margin between classes
2. **Support Vectors**: Uses only the data points closest to the decision boundary
3. **Kernel Trick**: Maps data to higher-dimensional space for non-linear separation
4. **Margin Maximization**: Maximizes the distance between the hyperplane and nearest data points

### Key Characteristics
- **Maximum Margin**: Finds the hyperplane with the largest margin
- **Sparse Solution**: Uses only support vectors
- **Kernel Support**: Can handle non-linear boundaries through kernels
- **Regularization**: C parameter controls the trade-off between margin and classification error

## Implementation Details

### 1. Data Generation

#### Synthetic Dataset
Created a synthetic binary classification dataset with controlled characteristics:
- **Samples**: 1,000 data points
- **Features**: 2 features (for easy visualization)
- **Classes**: 2 classes (binary classification)
- **Clusters per Class**: 2 clusters per class (creates non-linear separation)
- **Redundant Features**: 0 (no redundant features)
- **Random State**: 12 (for reproducibility)

#### Dataset Characteristics
- **Non-linearly Separable**: Requires non-linear decision boundaries
- **Visualizable**: 2D features allow for easy visualization
- **Controlled Complexity**: Designed to test different kernel functions

#### Data Visualization
- Created scatter plots to visualize the dataset
- Used color coding to distinguish classes
- Helped understand the data distribution and separation complexity

### 2. Model Training with Different Kernels

#### Data Splitting
- **Train-Test Split**: 80% training (800 samples) and 20% testing (200 samples)
- **Random State**: 12 (for reproducibility)

#### Kernel Functions Tested

##### 1. Linear Kernel
- **Kernel**: `kernel="linear"`
- **Accuracy**: 86.5%
- **Best For**: Linearly separable data
- **Characteristics**:
  - Fastest training and prediction
  - Most interpretable
  - Cannot handle non-linear boundaries
  - Lower accuracy on this non-linear dataset

##### 2. RBF (Radial Basis Function) Kernel
- **Kernel**: `kernel="rbf"` (default)
- **Accuracy**: 90% (Best Performance)
- **Best For**: Non-linear classification problems
- **Characteristics**:
  - Most versatile kernel
  - Handles complex non-linear boundaries
  - Most commonly used
  - Best overall performance on this dataset

##### 3. Polynomial Kernel
- **Kernel**: `kernel="poly"`
- **Accuracy**: 87%
- **Best For**: Polynomial decision boundaries
- **Characteristics**:
  - Can capture polynomial relationships
  - More hyperparameters (degree)
  - Moderate performance

##### 4. Sigmoid Kernel
- **Kernel**: `kernel="sigmoid"`
- **Accuracy**: 87% (using polynomial predictions)
- **Best For**: Similar to neural network activation
- **Characteristics**:
  - Less commonly used
  - Similar to tanh activation function
  - May not always converge

### 3. Hyperparameter Tuning

#### GridSearchCV
Performed exhaustive grid search with 5-fold cross-validation on the best performing kernel (RBF):

**Parameters Tested**:

1. **C Values**: [0.1, 1, 10, 100]
   - Regularization parameter
   - Controls the trade-off between margin size and classification error
   - **Low C (0.1)**: More regularization, wider margin, more misclassifications allowed
   - **High C (100)**: Less regularization, narrower margin, fewer misclassifications allowed
   - **Default (1.0)**: Balanced approach

2. **Gamma Values**: [1, 0.1, 0.01, 0.001]
   - Kernel coefficient (for RBF, poly, sigmoid kernels)
   - Controls the influence of individual training examples
   - **High gamma (1)**: Narrow influence, more complex boundaries, may overfit
   - **Low gamma (0.001)**: Wide influence, smoother boundaries, may underfit
   - **Default ('scale')**: 1 / (n_features × X.var())

3. **Kernel**: RBF (focus on best performing kernel)

**Cross-Validation**: 5-fold CV ensures robust parameter selection

**Result**: Achieved **90% accuracy** with tuned parameters

### 4. Model Evaluation

#### Evaluation Metrics

##### Accuracy Score
- **Definition**: Proportion of correct predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Result**: 90%

##### Classification Report
Provides detailed metrics for each class:
- **Precision**: Proportion of positive identifications that were correct
  - Formula: TP / (TP + FP)
  - Macro Average: 0.87
- **Recall (Sensitivity)**: Proportion of actual positives that were identified correctly
  - Formula: TP / (TP + FN)
  - Macro Average: 0.87
- **F1-Score**: Harmonic mean of precision and recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Macro Average: 0.86
- **Support**: Number of actual occurrences of each class in the test set

##### Confusion Matrix
A 2×2 matrix showing:
- **True Positives (TP)**: Correctly predicted positive class
- **False Positives (FP)**: Incorrectly predicted as positive
- **False Negatives (FN)**: Missed positive instances
- **True Negatives (TN)**: Correctly predicted negative class

## Results Summary

### Best Configuration
- **Best Kernel**: RBF (Radial Basis Function)
- **Best Accuracy**: 90%
- **Performance Metrics**:
  - Precision: 0.87 (macro avg)
  - Recall: 0.87 (macro avg)
  - F1-score: 0.86 (macro avg)

### Kernel Comparison

| Kernel | Accuracy | Best For |
|--------|----------|----------|
| Linear | 86.5% | Linearly separable data |
| RBF | 90% | Non-linear problems (BEST) |
| Polynomial | 87% | Polynomial boundaries |
| Sigmoid | 87% | Alternative non-linear |

### Performance Analysis
- **RBF Kernel**: Achieved the best performance (90% accuracy)
- **Non-linear Separation**: Required due to dataset characteristics
- **Hyperparameter Tuning**: Improved model performance
- **Balanced Metrics**: Good precision, recall, and F1-score

## Key Features

### Kernel Function Comparison
- Multiple kernel functions tested
- Performance comparison across kernels
- Best kernel identification

### Hyperparameter Optimization
- GridSearchCV for exhaustive search
- Cross-validation for robust evaluation
- Optimal parameter selection

### Comprehensive Evaluation
- Multiple evaluation metrics
- Classification report
- Confusion matrix
- Data visualization

## Files
- `Support_vector_machine_classfication.ipynb` - Jupyter notebook containing:
  - Synthetic data generation
  - Data visualization
  - Multiple kernel function testing
  - Hyperparameter tuning with GridSearchCV
  - Comprehensive model evaluation

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
1. Open the Jupyter notebook: `Support_vector_machine_classfication.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Generate synthetic classification data
   - Visualize the dataset
   - Test different kernel functions
   - Tune hyperparameters with GridSearchCV
   - Evaluate model performance with multiple metrics

## Advantages of SVC
- **Effective for Non-linear Problems**: Through kernel functions
- **Memory Efficient**: Uses only support vectors
- **Versatile**: Multiple kernel options
- **Regularization**: Built-in through C parameter
- **Works Well with High-dimensional Data**: Effective in high-dimensional spaces
- **Robust**: Less sensitive to overfitting than some algorithms

## Disadvantages of SVC
- **Slow Training**: Can be slow for large datasets
- **Sensitive to Feature Scaling**: Requires scaled features
- **Hyperparameter Sensitive**: Performance depends heavily on C, gamma, kernel
- **Not Probabilistic**: Doesn't provide probability estimates (without calibration)
- **Difficult to Interpret**: Less interpretable than linear models
- **Memory Usage**: Can be memory-intensive for large datasets

## When to Use SVC
- Binary and multiclass classification
- Non-linearly separable data
- When you need good performance with proper tuning
- High-dimensional data
- When you have a moderate-sized dataset
- As an alternative to neural networks for simpler problems

## Kernel Selection Guide

### Linear Kernel
- **Use When**: Data is linearly separable
- **Pros**: Fast, interpretable, less hyperparameters
- **Cons**: Cannot handle non-linear boundaries
- **Best For**: Simple, linearly separable problems

### RBF Kernel
- **Use When**: Non-linear relationships exist
- **Pros**: Most versatile, handles most cases, good default
- **Cons**: More computationally expensive, more hyperparameters
- **Best For**: General-purpose classification (recommended default)

### Polynomial Kernel
- **Use When**: Polynomial relationships are expected
- **Pros**: Can capture polynomial patterns
- **Cons**: More hyperparameters (degree), may be slower
- **Best For**: Known polynomial relationships

### Sigmoid Kernel
- **Use When**: Similar to neural network activation needed
- **Pros**: Alternative non-linear option
- **Cons**: Less commonly used, may not always converge
- **Best For**: Specific use cases similar to neural networks

## Hyperparameter Tuning Guide

### C Parameter (Regularization)
- **Low C (0.1-1)**: 
  - More regularization
  - Wider margin
  - More misclassifications allowed
  - Simpler model
- **High C (10-100)**: 
  - Less regularization
  - Narrower margin
  - Fewer misclassifications allowed
  - More complex model
- **Default (1.0)**: Good starting point

### Gamma Parameter (Kernel Coefficient)
- **Low gamma (0.001-0.01)**: 
  - Wide influence of support vectors
  - Smoother decision boundaries
  - May underfit
- **High gamma (0.1-1)**: 
  - Narrow influence
  - More complex boundaries
  - May overfit
- **Default ('scale')**: Usually good choice

### Tuning Strategy
1. Start with RBF kernel and default parameters
2. Use GridSearchCV for initial exploration
3. Fine-tune C and gamma based on results
4. Try other kernels if RBF doesn't perform well
5. Use cross-validation for robust evaluation

## Future Enhancements
- Decision boundary visualization
- Support vector visualization
- Probability calibration (Platt scaling)
- Multiclass classification examples
- Comparison with other classification algorithms
- Feature importance analysis
- Handling of imbalanced datasets
- Cross-validation for all kernels

## References
- Scikit-learn Documentation: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Support Vector Machines" by Nello Cristianini and John Shawe-Taylor
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
