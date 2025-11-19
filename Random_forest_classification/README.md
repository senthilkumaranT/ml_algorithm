# Random Forest Classification

## Overview
This folder contains a comprehensive implementation of Random Forest classification on a Travel dataset, including extensive data preprocessing, feature engineering, and model comparison.

## Technique Used
**Random Forest Classifier** - An ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees. It uses bagging (bootstrap aggregating) to reduce overfitting.

## What We Did

### 1. Data Cleaning
- **Missing Value Analysis**: Identified and analyzed missing values across multiple features:
  - Age (4.62% missing)
  - TypeofContact (0.51% missing)
  - DurationOfPitch (5.14% missing)
  - NumberOfFollowups (0.92% missing)
  - PreferredPropertyStar (0.53% missing)
  - NumberOfTrips (2.86% missing)
  - NumberOfChildrenVisiting (1.35% missing)
  - MonthlyIncome (4.77% missing)
- **Data Imputation**: 
  - Used median imputation for numerical features (Age, DurationOfPitch, PreferredPropertyStar, NumberOfTrips, MonthlyIncome)
  - Used mode imputation for categorical features (TypeofContact, NumberOfFollowups, NumberOfChildrenVisiting)
- **Data Standardization**: Fixed inconsistent values (e.g., "Fe Male" → "female")
- **Feature Removal**: Dropped CustomerID column

### 2. Feature Engineering
- Created a new feature `Total_visit` by combining `NumberOfChildrenVisiting` and `NumberOfPersonVisiting`
- Removed redundant features after feature combination
- Identified discrete and continuous features

### 3. Data Preprocessing
- Applied `StandardScaler` for numerical features
- Applied `OneHotEncoder` for categorical features
- Used `ColumnTransformer` for efficient preprocessing pipeline

### 4. Model Training and Comparison
- Compared three models:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
- Evaluated models using multiple metrics:
  - Accuracy
  - F1 Score
  - Recall
  - ROC AUC Score
  - Precision

### 5. Hyperparameter Tuning
- Performed `RandomizedSearchCV` to find optimal hyperparameters for Random Forest:
  - `max_depth`: [5, 8, 15, None, 10]
  - `max_features`: [5, 7, "auto", 10]
  - `min_samples_split`: [8, 10, 20]
  - `n_estimators`: [10, 100, 200, 500]
- Best parameters found: `{'n_estimators': 500, 'min_samples_split': 8, 'max_features': 10, 'max_depth': 15}`

## Results
Random Forest achieved the best performance:
- **Test Accuracy**: 90.90%
- **Test F1 Score**: 0.7327
- **Test Precision**: 0.9385
- **Test Recall**: 0.6010
- **Test ROC AUC**: 0.7953

## Key Features
- Comprehensive data cleaning pipeline
- Feature engineering and transformation
- Multiple model comparison
- Hyperparameter optimization using RandomizedSearchCV
- Extensive evaluation metrics

## Files
- `RANDOM_FOREST.ipynb` - Jupyter notebook containing the complete implementation
- `Travel.csv` - Dataset used for training
- `dataset/` - Additional dataset folder

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


