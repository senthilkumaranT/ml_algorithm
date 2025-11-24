# Random Forest Classification

## Overview
This folder contains a comprehensive implementation of Random Forest classification on a real-world Travel dataset. The implementation includes extensive data cleaning, feature engineering, model comparison, and hyperparameter tuning to achieve optimal performance.

## Algorithm Description
**Random Forest Classifier** is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (for classification) of the individual trees. It uses bagging (bootstrap aggregating) to reduce overfitting and improve generalization.

### How Random Forest Works
1. **Bootstrap Sampling**: Creates multiple training sets by sampling with replacement from the original dataset
2. **Tree Construction**: Builds a decision tree for each bootstrap sample
3. **Feature Randomness**: At each split, considers only a random subset of features (reduces correlation between trees)
4. **Voting**: For classification, each tree votes for a class, and the majority class is selected
5. **Averaging**: For regression, averages the predictions from all trees

### Key Characteristics
- **Ensemble Method**: Combines multiple weak learners (trees) into a strong learner
- **Reduces Overfitting**: More robust than single decision trees
- **Feature Importance**: Can identify which features are most important
- **Handles Non-linearity**: Can capture complex relationships
- **Handles Missing Values**: Can work with incomplete data
- **No Feature Scaling Required**: Works with raw feature values

## Dataset: Travel

### Dataset Overview
- **Source**: Travel.csv
- **Domain**: Travel and tourism industry
- **Purpose**: Predict customer behavior or travel preferences
- **Features**: Multiple features including demographic, behavioral, and preference data

### Dataset Structure
The dataset contains various features related to customers and their travel preferences, including:
- Customer demographics (Age, Gender, MaritalStatus)
- Contact information (TypeofContact)
- Behavioral data (DurationOfPitch, NumberOfFollowups)
- Preference data (PreferredPropertyStar)
- Travel history (NumberOfTrips, NumberOfChildrenVisiting, NumberOfPersonVisiting)
- Financial information (MonthlyIncome)

## Implementation Details

### 1. Data Cleaning

#### Missing Value Analysis
Identified and analyzed missing values across multiple features:
- **Age**: 4.62% missing values
- **TypeofContact**: 0.51% missing values
- **DurationOfPitch**: 5.14% missing values
- **NumberOfFollowups**: 0.92% missing values
- **PreferredPropertyStar**: 0.53% missing values
- **NumberOfTrips**: 2.86% missing values
- **NumberOfChildrenVisiting**: 1.35% missing values
- **MonthlyIncome**: 4.77% missing values

#### Data Imputation Strategy
- **Numerical Features**: Used **median imputation** for:
  - Age
  - DurationOfPitch
  - PreferredPropertyStar
  - NumberOfTrips
  - MonthlyIncome
  - *Rationale*: Median is robust to outliers

- **Categorical Features**: Used **mode imputation** for:
  - TypeofContact
  - NumberOfFollowups
  - NumberOfChildrenVisiting
  - *Rationale*: Mode represents the most common value

#### Data Standardization
- Fixed inconsistent values:
  - "Fe Male" → "female" (standardized gender values)
- Ensured consistent formatting across categorical features

#### Feature Removal
- Dropped `CustomerID` column (identifier, not predictive)

### 2. Feature Engineering

#### Feature Creation
- **Total_visit**: Created by combining:
  - `NumberOfChildrenVisiting`
  - `NumberOfPersonVisiting`
  - *Purpose*: Captures total visit count in a single feature

#### Feature Analysis
- Identified discrete and continuous features
- Removed redundant features after feature combination
- Analyzed feature distributions and relationships

### 3. Data Preprocessing

#### Feature Scaling
- **Numerical Features**: Applied `StandardScaler`
  - Standardizes features to have mean=0 and std=1
  - Formula: (x - mean) / std
  - *Purpose*: Ensures all features are on the same scale

#### Feature Encoding
- **Categorical Features**: Applied `OneHotEncoder`
  - Converts categorical variables into binary vectors
  - Uses `drop='first'` to avoid multicollinearity
  - *Purpose*: Converts non-numeric data into numeric format

#### Preprocessing Pipeline
- Used `ColumnTransformer` for efficient preprocessing
- Separates numerical and categorical transformations
- Applies appropriate transformations to each feature type

### 4. Model Training and Comparison

#### Models Compared
1. **Logistic Regression**
   - Linear classification model
   - Baseline for comparison
   - Fast training and prediction

2. **Decision Tree**
   - Single tree model
   - Interpretable but prone to overfitting
   - Baseline for Random Forest

3. **Random Forest**
   - Ensemble of decision trees
   - Reduces overfitting
   - Better generalization

#### Evaluation Metrics
Comprehensive evaluation using multiple metrics:
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Recall**: Proportion of actual positives correctly identified
- **ROC AUC Score**: Area under the ROC curve
- **Precision**: Proportion of positive predictions that are correct

### 5. Hyperparameter Tuning

#### RandomizedSearchCV
Performed randomized search to find optimal hyperparameters:

**Parameters Tested**:
- **max_depth**: [5, 8, 15, None, 10]
  - Maximum depth of trees
  - None = no limit
  - Controls model complexity

- **max_features**: [5, 7, "auto", 10]
  - Number of features to consider for best split
  - "auto" = sqrt(n_features)
  - Controls feature randomness

- **min_samples_split**: [8, 10, 20]
  - Minimum samples required to split a node
  - Prevents overfitting
  - Higher values = simpler trees

- **n_estimators**: [10, 100, 200, 500]
  - Number of trees in the forest
  - More trees = better performance but slower
  - Diminishing returns after certain point

**Best Parameters Found**:
```python
{
    'n_estimators': 500,
    'min_samples_split': 8,
    'max_features': 10,
    'max_depth': 15
}
```

**Cross-Validation**: Used to ensure robust parameter selection

## Results

### Random Forest Performance
Random Forest achieved the best performance among all models:

- **Test Accuracy**: 90.90%
- **Test F1 Score**: 0.7327
- **Test Precision**: 0.9385
- **Test Recall**: 0.6010
- **Test ROC AUC**: 0.7953

### Performance Interpretation
- **High Accuracy (90.90%)**: Model correctly classifies most instances
- **High Precision (0.9385)**: When model predicts positive, it's usually correct
- **Moderate Recall (0.6010)**: Model identifies 60% of actual positives
- **Good F1 Score (0.7327)**: Balanced precision and recall
- **Good ROC AUC (0.7953)**: Good discrimination ability

### Model Comparison Results
Random Forest outperformed both Logistic Regression and single Decision Tree, demonstrating the power of ensemble methods.

## Key Features

### Data Preprocessing Pipeline
- Comprehensive missing value handling
- Feature engineering and transformation
- Efficient preprocessing with ColumnTransformer
- Data quality improvements

### Model Development
- Multiple model comparison
- Hyperparameter optimization
- Cross-validation for robust evaluation
- Comprehensive performance metrics

### Best Practices
- Proper train-test splitting
- Feature scaling and encoding
- Hyperparameter tuning
- Multiple evaluation metrics

## Files
- `RANDOM_FOREST.ipynb` - Jupyter notebook containing the complete implementation:
  - Data cleaning and preprocessing
  - Feature engineering
  - Model training and comparison
  - Hyperparameter tuning
  - Model evaluation
- `dataset/Travel.csv` - The travel dataset used for training

## Dependencies
```python
pandas      # Data manipulation and analysis
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
1. Ensure the dataset file `Travel.csv` is in the `dataset/` folder
2. Open the Jupyter notebook: `RANDOM_FOREST.ipynb`
3. Run all cells sequentially
4. The notebook will:
   - Load and explore the dataset
   - Clean and preprocess the data
   - Engineer new features
   - Train and compare multiple models
   - Tune hyperparameters
   - Evaluate final model performance

## Advantages of Random Forest
- **Reduces Overfitting**: More robust than single decision trees
- **Handles Non-linearity**: Can capture complex relationships
- **Feature Importance**: Identifies important features
- **Handles Missing Values**: Can work with incomplete data
- **No Feature Scaling Required**: Works with raw values
- **Parallelizable**: Trees can be built in parallel
- **Works with Mixed Data Types**: Handles both numerical and categorical features

## Disadvantages of Random Forest
- **Less Interpretable**: Harder to interpret than single trees
- **Memory Intensive**: Stores multiple trees
- **Slower Prediction**: Takes longer than simpler models
- **Can Overfit**: With too many trees or insufficient regularization
- **Black Box**: Less transparent than linear models

## When to Use Random Forest
- Classification and regression tasks
- When you need good performance without extensive tuning
- When dealing with non-linear relationships
- When you want feature importance
- As a baseline for complex problems
- When you have mixed data types

## Hyperparameter Tuning Guide

### Key Hyperparameters
1. **n_estimators**: Number of trees (more = better but slower)
2. **max_depth**: Tree depth (controls complexity)
3. **min_samples_split**: Minimum samples to split (prevents overfitting)
4. **min_samples_leaf**: Minimum samples in leaf (prevents overfitting)
5. **max_features**: Features per split (controls randomness)
6. **bootstrap**: Whether to use bootstrap sampling

### Tuning Strategy
- Start with default parameters
- Use RandomizedSearchCV for initial exploration
- Use GridSearchCV for fine-tuning
- Use cross-validation for robust evaluation

## Future Enhancements
- Feature importance visualization
- Partial dependence plots
- Model interpretation techniques (SHAP values)
- Handling of class imbalance
- Cross-validation for all models
- Additional ensemble methods (Gradient Boosting, XGBoost)
- Model deployment considerations

## References
- Scikit-learn Documentation: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Hands-On Machine Learning" by Aurélien Géron
