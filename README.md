# Machine Learning Algorithms Repository

## Overview
This repository contains comprehensive implementations of various machine learning algorithms, covering both classification and regression tasks. Each algorithm is implemented with proper data preprocessing, model training, evaluation, and hyperparameter tuning where applicable.

## Repository Structure

```
ml_algorithm/
├── KNN/                          # K-Nearest Neighbors Classification
├── Navie_bayes/                  # Naive Bayes Classification
├── Random_forest_classification/ # Random Forest Classification
├── logistic_regression/          # Logistic Regression (Binary & Multiclass)
├── support_vecotr_regression/    # Support Vector Regression
├── support_vector_classification/# Support Vector Classification
├── Linear_regression/            # Linear Regression
├── multiple_linear_regression/   # Multiple Linear Regression
└── Polynomial_regression/        # Polynomial Regression
```

## Algorithms Implemented

### Classification Algorithms

#### 1. **K-Nearest Neighbors (KNN)**
- **Location**: `KNN/`
- **Technique**: Instance-based learning algorithm
- **Features**: Binary classification with k=5 neighbors
- **Accuracy**: Evaluated using accuracy score

#### 2. **Naive Bayes**
- **Location**: `Navie_bayes/`
- **Technique**: Gaussian Naive Bayes
- **Features**: Multi-class classification on Iris dataset
- **Accuracy**: ~96.67%
- **Evaluation**: Accuracy, classification report, confusion matrix

#### 3. **Logistic Regression**
- **Location**: `logistic_regression/`
- **Technique**: Logistic Regression with various configurations
- **Features**:
  - Binary classification (91% accuracy)
  - Multiclass classification (One-vs-Rest strategy)
  - Handling imbalanced datasets with class weights
  - ROC curve and AUC score analysis
  - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- **Accuracy**: 91% (binary), 59% (multiclass), 98.85% (imbalanced)

#### 4. **Random Forest Classification**
- **Location**: `Random_forest_classification/`
- **Technique**: Ensemble learning with multiple decision trees
- **Features**:
  - Comprehensive data cleaning and preprocessing
  - Feature engineering
  - Multiple model comparison (Logistic Regression, Decision Tree, Random Forest)
  - Hyperparameter tuning with RandomizedSearchCV
- **Accuracy**: 90.90%
- **Best Model**: Random Forest with optimized hyperparameters

#### 5. **Support Vector Classification (SVC)**
- **Location**: `support_vector_classification/`
- **Technique**: Support Vector Machine for classification
- **Features**:
  - Multiple kernel functions (linear, RBF, polynomial, sigmoid)
  - Hyperparameter tuning with GridSearchCV
  - Best kernel: RBF
- **Accuracy**: 90%

### Regression Algorithms

#### 6. **Support Vector Regression (SVR)**
- **Location**: `support_vecotr_regression/`
- **Technique**: Support Vector Machine for regression
- **Features**:
  - Categorical feature encoding (Label Encoding, One-Hot Encoding)
  - Multiple kernel support
  - Hyperparameter tuning
- **Evaluation**: R² score

#### 7. **Linear Regression**
- **Location**: `Linear_regression/`
- **Technique**: Simple linear regression
- **Dataset**: Height-weight dataset

#### 8. **Multiple Linear Regression**
- **Location**: `multiple_linear_regression/`
- **Technique**: Multiple linear regression with multiple features
- **Dataset**: Economic index dataset

#### 9. **Polynomial Regression**
- **Location**: `Polynomial_regression/`
- **Technique**: Polynomial regression for non-linear relationships

## Common Techniques Used Across Projects

### Data Preprocessing
- Train-test splitting (typically 80-20 split)
- Handling missing values (median/mode imputation)
- Feature encoding (Label Encoding, One-Hot Encoding)
- Feature scaling (StandardScaler)
- Feature engineering

### Model Evaluation
- **Classification Metrics**:
  - Accuracy score
  - Precision, Recall, F1-score
  - Confusion matrix
  - Classification report
  - ROC curve and AUC score
- **Regression Metrics**:
  - R² score (coefficient of determination)
  - Mean squared error (where applicable)

### Hyperparameter Tuning
- **GridSearchCV**: Exhaustive search over parameter grid
- **RandomizedSearchCV**: Random search over parameter distributions
- Cross-validation (typically 3-5 folds)

### Model Comparison
- Multiple algorithm comparison
- Performance metrics comparison
- Best model selection based on evaluation metrics

## Technologies Used

- **Python**: Primary programming language
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Jupyter Notebooks**: Interactive development environment

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebooks
1. Navigate to the desired algorithm folder
2. Open the Jupyter notebook file (`.ipynb`)
3. Run all cells to execute the complete workflow

### Example
```bash
cd KNN
jupyter notebook KNN_algorithm.ipynb
```

## Key Highlights

1. **Comprehensive Coverage**: Both classification and regression algorithms
2. **Real-world Applications**: Practical datasets and use cases
3. **Best Practices**: Proper data preprocessing, evaluation, and hyperparameter tuning
4. **Documentation**: Each folder contains detailed README explaining the implementation
5. **Code Quality**: Clean, well-structured code with comments

## Performance Summary

| Algorithm | Task Type | Best Accuracy/R² | Dataset |
|-----------|-----------|------------------|---------|
| KNN | Classification | Evaluated | Synthetic |
| Naive Bayes | Classification | 96.67% | Iris |
| Logistic Regression | Classification | 91% (binary) | Synthetic |
| Random Forest | Classification | 90.90% | Travel |
| SVC | Classification | 90% | Synthetic |
| SVR | Regression | Evaluated | Tips |

## Notes

- All implementations use scikit-learn library for consistency
- Random states are set for reproducibility
- Evaluation metrics are comprehensive and appropriate for each task type
- Hyperparameter tuning is performed where applicable to optimize performance

## Contributing

Feel free to explore each algorithm folder for detailed implementation and documentation. Each folder contains:
- A README.md file explaining the technique and implementation
- Jupyter notebook(s) with complete code
- Dataset files (where applicable)

## License

This repository is for educational purposes, demonstrating various machine learning algorithms and their implementations.

