# Support Vector Machine Regression (SVR)

## Overview
This folder contains a comprehensive implementation of Support Vector Regression (SVR) using the tips dataset to predict total bill amounts based on various features. The implementation includes categorical feature encoding, multiple kernel functions, and hyperparameter tuning.

## Algorithm Description
**Support Vector Regression (SVR)** is a regression algorithm that uses the same principles as Support Vector Machines (SVM) but for regression tasks. SVR finds a function that approximates the mapping from inputs to real-valued outputs while maintaining all the main features of SVM.

### How SVR Works
1. **Epsilon-Insensitive Loss**: Defines a margin (epsilon tube) where errors are not penalized
2. **Support Vectors**: Uses only a subset of training points (support vectors) near the decision boundary
3. **Kernel Trick**: Can handle non-linear relationships using kernel functions
4. **Optimization**: Minimizes a cost function that balances model complexity and prediction error

### Key Characteristics
- **Sparse Solution**: Uses only support vectors, making it memory efficient
- **Kernel Support**: Can handle non-linear relationships
- **Robust to Outliers**: Epsilon-insensitive loss provides some robustness
- **Regularization**: C parameter controls the trade-off between margin size and training error

## Dataset: Tips

### Dataset Overview
- **Source**: Loaded from seaborn's built-in datasets
- **Samples**: 244 restaurant bills
- **Domain**: Restaurant tipping data
- **Target Variable**: `total_bill` (total bill amount in dollars)

### Dataset Features
The dataset contains 7 features:
1. **total_bill** (target): Total bill amount in dollars
2. **tip**: Tip amount in dollars
3. **sex**: Gender of the bill payer (categorical: Male, Female)
4. **smoker**: Whether the party included smokers (categorical: Yes, No)
5. **day**: Day of the week (categorical: Thur, Fri, Sat, Sun)
6. **time**: Time of day (categorical: Lunch, Dinner)
7. **size**: Party size (numerical: 1-6)

### Dataset Characteristics
- **No Missing Values**: Complete dataset
- **Mixed Data Types**: Contains both numerical and categorical features
- **Real-world Data**: Actual restaurant billing data
- **Small Dataset**: 244 samples (suitable for SVR)

## Implementation Details

### 1. Data Exploration

#### Data Loading
- Loaded tips dataset from seaborn
- Explored dataset structure and statistics
- Analyzed feature distributions

#### Feature Analysis
- Examined value counts for categorical features
- Analyzed distributions of numerical features
- Identified feature types (numerical vs. categorical)

### 2. Feature Engineering

#### Feature Selection
Selected the following features for prediction:
- `tip`: Tip amount (numerical)
- `sex`: Gender (categorical)
- `smoker`: Smoking status (categorical)
- `day`: Day of week (categorical)
- `time`: Time of day (categorical)
- `size`: Party size (numerical)

**Target Variable**: `total_bill`

#### Categorical Feature Encoding

**Label Encoding**:
Applied to ordinal or binary categorical features:
- `sex`: Male/Female → 0/1
- `smoker`: Yes/No → 0/1
- `time`: Lunch/Dinner → 0/1
- *Method*: `LabelEncoder` from scikit-learn
- *Purpose*: Converts categorical labels to numerical values

**One-Hot Encoding**:
Applied to nominal categorical features:
- `day`: Thur, Fri, Sat, Sun → Binary vectors
- *Method*: `OneHotEncoder` with `drop='first'`
- *Purpose*: Creates binary columns for each category
- *drop='first'*: Removes one column to avoid multicollinearity

#### Preprocessing Pipeline
- Used `ColumnTransformer` for efficient preprocessing
- Separates encoding strategies for different feature types
- Applies transformations consistently to training and test sets

### 3. Model Training

#### Data Splitting
- **Train-Test Split**: 80% training (195 samples) and 20% testing (49 samples)
- **Random State**: Set for reproducible results

#### Model Configuration
- **Algorithm**: `SVR` (Support Vector Regression) from scikit-learn
- **Initial Parameters**: Default parameters
- **Evaluation Metric**: R² score (coefficient of determination)

#### R² Score Interpretation
- **Range**: -∞ to 1
- **1.0**: Perfect predictions
- **0.0**: Model performs as well as predicting the mean
- **Negative**: Model performs worse than predicting the mean

### 4. Hyperparameter Tuning

#### GridSearchCV
Performed exhaustive grid search with 5-fold cross-validation:

**Parameters Tested**:

1. **Kernels**:
   - `linear`: Linear kernel (for linear relationships)
   - `poly`: Polynomial kernel (for polynomial relationships)
   - `rbf`: Radial Basis Function kernel (for non-linear relationships) - most commonly used
   - `sigmoid`: Sigmoid kernel (less commonly used)

2. **C Values**: [1, 5, 10]
   - Regularization parameter
   - Controls the trade-off between margin size and training error
   - Higher C = less regularization, more complex model
   - Lower C = more regularization, simpler model

3. **Gamma**: ['scale', 'auto']
   - Kernel coefficient (for RBF, poly, sigmoid kernels)
   - 'scale': 1 / (n_features × X.var())
   - 'auto': 1 / n_features
   - Controls the influence of individual training examples

**Cross-Validation**: 5-fold CV ensures robust parameter selection

**Best Parameters**: Found through exhaustive search over all combinations

## Key Features

### Encoding Techniques
- **Label Encoding**: For ordinal/binary categorical features
- **One-Hot Encoding**: For nominal categorical features
- **Efficient Pipeline**: ColumnTransformer for organized preprocessing

### Kernel Functions
- **Linear Kernel**: For linear relationships
- **Polynomial Kernel**: For polynomial relationships
- **RBF Kernel**: For non-linear relationships (most versatile)
- **Sigmoid Kernel**: Alternative non-linear option

### Model Evaluation
- **R² Score**: Primary evaluation metric
- **Cross-Validation**: For robust hyperparameter selection
- **Train-Test Split**: For final model evaluation

## Results

### Model Performance
- **Evaluation Metric**: R² score
- **Performance**: Optimized through hyperparameter tuning
- **Best Kernel**: Determined through grid search
- **Best C and Gamma**: Found through exhaustive search

### Hyperparameter Tuning Results
The GridSearchCV identified optimal hyperparameters that maximize the R² score on the validation set.

## Files
- `Support_vector_machine_Regression.ipynb` - Jupyter notebook containing:
  - Data exploration
  - Feature engineering and encoding
  - Model training
  - Hyperparameter tuning with GridSearchCV
  - Model evaluation

## Dependencies
```python
pandas      # Data manipulation
numpy       # Numerical computing
scikit-learn # Machine learning library
matplotlib  # Data visualization
seaborn     # Statistical visualization and dataset
```

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage
1. Open the Jupyter notebook: `Support_vector_machine_Regression.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load the tips dataset
   - Explore and analyze the data
   - Encode categorical features
   - Train the SVR model
   - Tune hyperparameters with GridSearchCV
   - Evaluate model performance

## Advantages of SVR
- **Handles Non-linearity**: Through kernel functions
- **Sparse Solution**: Uses only support vectors
- **Memory Efficient**: Stores only support vectors
- **Robust to Outliers**: Epsilon-insensitive loss
- **Flexible**: Multiple kernel options
- **Regularization**: Built-in through C parameter

## Disadvantages of SVR
- **Slow Training**: Can be slow for large datasets
- **Sensitive to Feature Scaling**: Requires scaled features
- **Hyperparameter Sensitive**: Performance depends on C, gamma, kernel choice
- **Not Probabilistic**: Doesn't provide probability estimates
- **Difficult to Interpret**: Less interpretable than linear models

## When to Use SVR
- Regression tasks with non-linear relationships
- When you have a moderate-sized dataset
- When you need a sparse solution
- When dealing with high-dimensional data
- When linear regression doesn't perform well

## Kernel Selection Guide

### Linear Kernel
- Use when: Data is linearly separable/related
- Pros: Fast, interpretable
- Cons: Cannot handle non-linear relationships

### Polynomial Kernel
- Use when: Data has polynomial relationships
- Pros: Can capture polynomial patterns
- Cons: More hyperparameters to tune

### RBF Kernel
- Use when: Data has complex non-linear relationships
- Pros: Most versatile, handles most cases
- Cons: More computationally expensive

### Sigmoid Kernel
- Use when: Similar to neural network activation
- Pros: Alternative non-linear option
- Cons: Less commonly used, may not converge

## Hyperparameter Tuning Tips

### C Parameter
- **Low C (0.1-1)**: More regularization, simpler model
- **High C (10-100)**: Less regularization, more complex model
- **Default (1.0)**: Good starting point

### Gamma Parameter
- **Low gamma**: Wider influence of support vectors
- **High gamma**: Narrower influence, more complex boundaries
- **'scale' or 'auto'**: Good defaults

### Kernel Selection
- Start with RBF (most versatile)
- Try linear if data appears linear
- Use polynomial for known polynomial relationships

## Future Enhancements
- Additional evaluation metrics (MSE, MAE, RMSE)
- Feature importance analysis
- Visualization of predictions vs. actual values
- Residual analysis
- Comparison with other regression algorithms
- Handling of missing values
- Feature scaling visualization

## References
- Scikit-learn Documentation: [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Support Vector Machines" by Nello Cristianini and John Shawe-Taylor
