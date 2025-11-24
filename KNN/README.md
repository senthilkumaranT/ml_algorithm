# K-Nearest Neighbors (KNN) Algorithm

## Overview
This folder contains a complete implementation of the K-Nearest Neighbors (KNN) classification algorithm using scikit-learn. KNN is a simple yet powerful non-parametric algorithm that makes predictions based on the similarity of data points.

## Algorithm Description
**K-Nearest Neighbors (KNN)** is a non-parametric, instance-based learning algorithm used for both classification and regression tasks. For classification, KNN classifies a data point by finding the k nearest neighbors in the feature space and assigning the majority class among those neighbors.

### How KNN Works
1. **Distance Calculation**: Computes the distance (typically Euclidean) between the query point and all training points
2. **Neighbor Selection**: Selects the k nearest neighbors based on the calculated distances
3. **Voting**: For classification, assigns the class that appears most frequently among the k neighbors
4. **Prediction**: Returns the predicted class for the query point

### Key Characteristics
- **Lazy Learning**: No explicit training phase; all computation happens during prediction
- **Non-parametric**: Makes no assumptions about the underlying data distribution
- **Instance-based**: Uses the entire training dataset for predictions
- **Sensitive to k value**: The choice of k significantly affects model performance

## Implementation Details

### Dataset
- **Type**: Synthetic classification dataset
- **Samples**: 1,000 data points
- **Features**: 5 features per sample
- **Classes**: 2 classes (binary classification)
- **Generation**: Created using `make_classification` from scikit-learn with `random_state=123` for reproducibility

### Data Preprocessing
- **Train-Test Split**: 80% training (800 samples) and 20% testing (200 samples)
- **Random State**: Set to 123 for reproducible results
- **No scaling required**: KNN uses distance metrics, but in this implementation, the synthetic data is already well-scaled

### Model Configuration
- **Algorithm**: `KNeighborsClassifier` from scikit-learn
- **k value (n_neighbors)**: 5
- **Distance Metric**: Euclidean (default)
- **Weight Function**: Uniform (default) - all neighbors have equal weight

### Model Training
The KNN model is trained by storing the training data. Since KNN is a lazy learner, the `fit()` method simply stores the training data for later use during prediction.

### Model Evaluation
- **Metric**: Accuracy Score
- **Formula**: (Number of Correct Predictions) / (Total Number of Predictions)
- The accuracy score provides a simple measure of how well the model performs on the test set

## Code Structure

### Step-by-Step Implementation
1. **Import Libraries**: pandas, seaborn, matplotlib, numpy, scikit-learn
2. **Generate Dataset**: Create synthetic classification dataset
3. **Split Data**: Divide into training and testing sets
4. **Initialize Model**: Create KNN classifier with k=5
5. **Train Model**: Fit the model on training data
6. **Make Predictions**: Predict classes for test data
7. **Evaluate Model**: Calculate accuracy score

## Advantages of KNN
- Simple to understand and implement
- No assumptions about data distribution
- Works well for non-linear decision boundaries
- Can be used for both classification and regression
- Effective for small to medium-sized datasets

## Disadvantages of KNN
- Computationally expensive for large datasets (requires distance calculation to all points)
- Sensitive to irrelevant features and noise
- Requires feature scaling for optimal performance
- Performance depends heavily on the choice of k
- Memory-intensive (stores entire training dataset)

## Hyperparameter Considerations

### Choosing k Value
- **Small k (e.g., k=1)**: 
  - More sensitive to noise
  - Higher variance, lower bias
  - May overfit
- **Large k (e.g., k=20)**:
  - Smoother decision boundaries
  - Lower variance, higher bias
  - May underfit
- **Optimal k**: Typically chosen through cross-validation (not implemented in this basic example)

### Distance Metrics
- **Euclidean Distance**: Default, works well for continuous features
- **Manhattan Distance**: Better for high-dimensional data
- **Minkowski Distance**: Generalization of Euclidean and Manhattan

## Files
- `KNN_algorithm.ipynb` - Jupyter notebook containing the complete implementation with step-by-step code

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
1. Open the Jupyter notebook: `KNN_algorithm.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Generate synthetic data
   - Split into train/test sets
   - Train the KNN model
   - Make predictions
   - Calculate accuracy

## Future Enhancements
- Cross-validation for optimal k selection
- Feature scaling implementation
- Different distance metrics comparison
- Visualization of decision boundaries
- Performance comparison with different k values
- Handling of categorical features

## References
- Scikit-learn Documentation: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- Machine Learning: A Probabilistic Perspective by Kevin P. Murphy
