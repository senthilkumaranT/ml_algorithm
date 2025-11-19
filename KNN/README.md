# K-Nearest Neighbors (KNN) Algorithm

## Overview
This folder contains an implementation of the K-Nearest Neighbors (KNN) classification algorithm using scikit-learn.

## Technique Used
**K-Nearest Neighbors (KNN)** - A non-parametric, instance-based learning algorithm used for classification tasks. KNN classifies data points based on the majority class of their k nearest neighbors in the feature space.

## What We Did
1. **Data Generation**: Created a synthetic classification dataset with 1000 samples, 5 features, and 2 classes using `make_classification` from scikit-learn
2. **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets
3. **Model Training**: Implemented KNN classifier with `n_neighbors=5` (k=5)
4. **Model Evaluation**: Evaluated the model performance using accuracy score

## Key Features
- Uses `KNeighborsClassifier` from scikit-learn
- Default k value of 5 neighbors
- Train-test split with 20% test size
- Accuracy-based evaluation

## Files
- `KNN_algorithm.ipynb` - Jupyter notebook containing the complete implementation

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


