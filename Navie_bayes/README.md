# Naive Bayes Classification

## Overview
This folder contains an implementation of the Naive Bayes classification algorithm using the Gaussian Naive Bayes variant on the famous Iris dataset.

## Technique Used
**Gaussian Naive Bayes** - A probabilistic classification algorithm based on Bayes' theorem with the "naive" assumption of feature independence. The Gaussian variant assumes that continuous features follow a normal (Gaussian) distribution.

## What We Did
1. **Dataset Loading**: Loaded the Iris dataset from scikit-learn, which contains 150 samples of iris flowers with 4 features (sepal length, sepal width, petal length, petal width) and 3 classes
2. **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets
3. **Model Training**: Implemented Gaussian Naive Bayes classifier using `GaussianNB()`
4. **Model Evaluation**: 
   - Calculated accuracy score (achieved ~96.67% accuracy)
   - Generated classification report showing precision, recall, and F1-score for each class
   - Created confusion matrix to visualize classification performance

## Results
- **Accuracy**: 96.67%
- **Performance**: Excellent classification performance across all three iris species classes
- The model shows perfect classification for class 0, and high accuracy for classes 1 and 2

## Key Features
- Uses `GaussianNB` from scikit-learn
- Comprehensive evaluation with accuracy, classification report, and confusion matrix
- Applied to the classic Iris multi-class classification problem

## Files
- `NAVIE_BAYES_THEOREM.ipynb` - Jupyter notebook containing the complete implementation

## Dependencies
- scikit-learn
- numpy


