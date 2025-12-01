# Naive Bayes Classification

## Overview
This folder contains a comprehensive implementation of the Naive Bayes classification algorithm using the Gaussian Naive Bayes variant. The implementation is applied to the famous Iris dataset, demonstrating multi-class classification capabilities.

## Algorithm Description
**Naive Bayes** is a probabilistic classification algorithm based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this simplifying assumption, Naive Bayes often performs remarkably well in practice.

### Bayes' Theorem
The algorithm uses Bayes' theorem to calculate the probability of a class given the features:

**P(Class | Features) = P(Features | Class) × P(Class) / P(Features)**

### Gaussian Naive Bayes
The **Gaussian Naive Bayes** variant assumes that continuous features follow a normal (Gaussian) distribution. For each feature, it estimates:
- **Mean (μ)**: Average value of the feature for each class
- **Variance (σ²)**: Spread of the feature values for each class

The probability density function is:
**P(x | Class) = (1 / √(2πσ²)) × exp(-(x-μ)² / (2σ²))**

### Key Characteristics
- **Probabilistic**: Provides probability estimates for each class
- **Fast Training**: Simple parameter estimation (mean and variance)
- **Fast Prediction**: Quick classification once trained
- **Feature Independence Assumption**: Assumes features are conditionally independent given the class
- **Works Well with Small Data**: Can perform well even with limited training data

## Dataset: Iris

### Dataset Overview
The **Iris dataset** is one of the most famous datasets in machine learning, introduced by Ronald Fisher in 1936.

- **Total Samples**: 150 iris flowers
- **Features**: 4 features per sample
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Classes**: 3 classes (multi-class classification)
  - Setosa (class 0)
  - Versicolor (class 1)
  - Virginica (class 2)
- **Samples per Class**: 50 samples for each class (balanced dataset)
- **Source**: Loaded from scikit-learn's built-in datasets

### Dataset Characteristics
- **No Missing Values**: Complete dataset
- **Balanced Classes**: Equal representation of all three classes
- **Linearly Separable**: Setosa is linearly separable from the other two classes
- **Real-world Data**: Actual measurements from iris flowers

## Implementation Details

### Data Preprocessing
- **Train-Test Split**: 80% training (120 samples) and 20% testing (30 samples)
- **Random State**: Set for reproducible results
- **No Feature Scaling**: Gaussian Naive Bayes handles different scales naturally
- **No Missing Value Handling**: Dataset is complete

### Model Configuration
- **Algorithm**: `GaussianNB()` from scikit-learn
- **Smoothing**: Uses default smoothing parameters
- **Prior Probabilities**: Automatically estimated from training data

### Model Training
The model learns:
1. **Prior Probabilities**: P(Class) for each of the 3 classes
2. **Mean Values**: Average feature values for each class
3. **Variance Values**: Spread of feature values for each class

### Model Evaluation

#### 1. Accuracy Score
- **Result**: ~96.67% accuracy
- **Interpretation**: The model correctly classifies approximately 29 out of 30 test samples

#### 2. Classification Report
Provides detailed metrics for each class:
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual occurrences of each class in the test set

#### 3. Confusion Matrix
A 3×3 matrix showing:
- **True Positives (TP)**: Correctly predicted class instances
- **False Positives (FP)**: Incorrectly predicted as a class
- **False Negatives (FN)**: Missed class instances
- **True Negatives (TN)**: Correctly rejected instances

### Performance Analysis
- **Class 0 (Setosa)**: Perfect classification (100% accuracy)
  - Easily separable from other classes
- **Class 1 (Versicolor)**: High accuracy
  - Some confusion with Class 2 (Virginica)
- **Class 2 (Virginica)**: High accuracy
  - Some confusion with Class 1 (Versicolor)

## Code Structure

### Step-by-Step Implementation
1. **Import Libraries**: scikit-learn, numpy
2. **Load Dataset**: Load Iris dataset from scikit-learn
3. **Split Data**: Divide into training and testing sets
4. **Initialize Model**: Create Gaussian Naive Bayes classifier
5. **Train Model**: Fit the model on training data
6. **Make Predictions**: Predict classes for test data
7. **Evaluate Model**: 
   - Calculate accuracy score
   - Generate classification report
   - Create confusion matrix

## Advantages of Naive Bayes
- **Fast Training**: Simple parameter estimation
- **Fast Prediction**: Quick classification
- **Probabilistic Output**: Provides probability estimates
- **Works Well with Small Data**: Can perform well with limited samples
- **Handles Multiple Classes**: Naturally supports multi-class classification
- **Not Sensitive to Irrelevant Features**: Due to independence assumption
- **Memory Efficient**: Only stores mean and variance for each feature-class combination

## Disadvantages of Naive Bayes
- **Feature Independence Assumption**: Rarely true in real-world data
- **Sensitive to Feature Scaling**: Though less critical than some algorithms
- **Zero Probability Problem**: Can occur if a feature value never appears with a class (handled by smoothing)
- **May Underperform**: Compared to more sophisticated algorithms on complex datasets

## When to Use Naive Bayes
- Text classification (spam detection, sentiment analysis)
- Multi-class classification problems
- When you need fast training and prediction
- When features are approximately independent
- As a baseline model for comparison
- Real-time prediction systems

## Hyperparameter Tuning (Not Implemented)
Potential hyperparameters to tune:
- **Smoothing Parameter (var_smoothing)**: Controls the amount of smoothing applied to variance estimates
- **Prior Probabilities**: Can be set manually if class distribution is known

## Files
- `NAVIE_BAYES_THEOREM.ipynb` - Jupyter notebook containing the complete implementation with evaluation metrics

## Dependencies
```python
scikit-learn  # Machine learning library (for GaussianNB and datasets)
numpy         # Numerical computing
```

## Installation
```bash
pip install scikit-learn numpy jupyter
```

## Code Example

Here's the complete code implementation:

```python
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
x, y = load_iris(return_X_y=True)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the model
gnb.fit(x_train, y_train)

# Make predictions
y_pred = gnb.predict(x_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Code Explanation

**Step 1: Load Dataset**
```python
from sklearn.datasets import load_iris
x, y = load_iris(return_X_y=True)
```
- Loads Iris dataset (150 samples, 4 features, 3 classes)
- Returns features (x) and target labels (y)

**Step 2: Train-Test Split**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
- Splits data: 80% training (120 samples), 20% testing (30 samples)
- Random split ensures different samples each run

**Step 3: Initialize and Train Model**
```python
gnb = GaussianNB()
gnb.fit(x_train, y_train)
```
- Creates Gaussian Naive Bayes classifier
- Trains model by estimating:
  - Prior probabilities for each class
  - Mean and variance for each feature-class combination

**Step 4: Make Predictions**
```python
y_pred = gnb.predict(x_test)
```
- Predicts class labels for test samples
- Uses Bayes' theorem with Gaussian probability distributions

**Step 5: Evaluate Model**
```python
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
- Calculates accuracy (proportion of correct predictions)
- Generates detailed classification report (precision, recall, F1-score)
- Creates confusion matrix showing prediction breakdown

## Usage
1. Open the Jupyter notebook: `NAVIE_BAYES_THEOREM.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load the Iris dataset
   - Split into train/test sets
   - Train the Gaussian Naive Bayes model
   - Make predictions
   - Evaluate with accuracy, classification report, and confusion matrix

## Results Summary
- **Accuracy**: 96.67%
- **Best Performing Class**: Setosa (100% accuracy)
- **Model Performance**: Excellent for this dataset
- **Evaluation Metrics**: Comprehensive (accuracy, precision, recall, F1-score, confusion matrix)

## Future Enhancements
- Hyperparameter tuning (var_smoothing)
- Comparison with other Naive Bayes variants (Multinomial, Bernoulli)
- Feature importance analysis
- Visualization of decision boundaries
- Cross-validation for more robust evaluation
- Handling of missing values

## References
- Scikit-learn Documentation: [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- Iris Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- "Pattern Recognition and Machine Learning" by Christopher Bishop
