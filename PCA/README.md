# Principal Component Analysis (PCA)

## Overview
This folder contains a complete implementation of Principal Component Analysis (PCA), a dimensionality reduction technique used to reduce the number of features while preserving as much variance as possible. The implementation uses the Breast Cancer Wisconsin dataset to demonstrate how PCA can reduce 30 features to 2 principal components for visualization and analysis.

## Algorithm Description
**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while retaining most of the information. It finds the directions (principal components) of maximum variance in the data.

### Mathematical Foundation
PCA works by:
1. **Standardization**: Centers and scales the data
2. **Covariance Matrix**: Computes the covariance matrix of the standardized data
3. **Eigenvalue Decomposition**: Finds eigenvalues and eigenvectors of the covariance matrix
4. **Principal Components**: Selects the eigenvectors (principal components) with the largest eigenvalues
5. **Transformation**: Projects the data onto the principal components

### Key Characteristics
- **Unsupervised**: Doesn't require target labels
- **Linear Transformation**: Projects data onto lower-dimensional space
- **Variance Preservation**: Retains maximum variance in fewer dimensions
- **Orthogonal Components**: Principal components are orthogonal (uncorrelated)
- **Dimensionality Reduction**: Reduces number of features while preserving information

## Dataset: Breast Cancer Wisconsin

### Dataset Overview
- **Source**: Loaded from scikit-learn's built-in datasets
- **Samples**: 569 breast cancer cases
- **Features**: 30 numeric features
- **Classes**: 2 classes (Malignant, Benign)
- **Domain**: Medical diagnosis

### Dataset Features
The dataset contains 30 features derived from digitized images of fine needle aspirates (FNA) of breast masses. Features include:

**Mean Values** (10 features):
- radius, texture, perimeter, area, smoothness
- compactness, concavity, concave points, symmetry, fractal dimension

**Standard Error** (10 features):
- Standard error of the mean values above

**Worst Values** (10 features):
- Largest (worst) values of the mean features above

### Dataset Characteristics
- **No Missing Values**: Complete dataset
- **High Dimensionality**: 30 features (benefits from dimensionality reduction)
- **Real-world Data**: Actual medical diagnosis data
- **Binary Classification**: Two classes (though PCA is unsupervised)

## Implementation Details

### 1. Data Loading

#### Dataset Loading
- Loaded Breast Cancer dataset from scikit-learn
- **Method**: `load_breast_cancer()`
- **Data Structure**: Dictionary containing data, target, feature names, etc.

#### Data Preparation
- Created pandas DataFrame from the data
- **Features**: All 30 features from the dataset
- **Columns**: Used feature names as column names
- **Shape**: 569 rows × 30 columns

### 2. Data Preprocessing

#### Feature Scaling
- Applied **StandardScaler** for standardization
- **Formula**: (x - mean) / std
- **Purpose**: 
  - PCA is sensitive to feature scales
  - Ensures all features contribute equally
  - Centers data around zero
  - Normalizes variance
- **Process**:
  1. Fit scaler on the data
  2. Transform the data to scaled version
- **Result**: Scaled data with mean=0 and std=1 for each feature

### 3. PCA Implementation

#### PCA Initialization
- **Algorithm**: `PCA` from scikit-learn
- **n_components**: 2 (reduces from 30 to 2 dimensions)
- **Purpose**: 
  - Reduce dimensionality for visualization
  - Demonstrate PCA transformation
  - Enable 2D plotting

#### PCA Fitting and Transformation
- **Fit**: `pca.fit(scaled_data)` - Computes principal components
- **Transform**: `pca.fit_transform(scaled_data)` - Projects data onto principal components
- **Result**: 569 samples × 2 components (reduced from 30 features)

### 4. Visualization

#### PCA Scatter Plot
- **X-axis**: First Principal Component (PC1)
- **Y-axis**: Second Principal Component (PC2)
- **Color Coding**: Based on target labels (Malignant/Benign)
- **Colormap**: "plasma" colormap
- **Title**: "Breast Cancer Dataset PCA"
- **Purpose**: Visualize how PCA separates the data in 2D space

#### Visualization Insights
- Shows how well the two classes are separated in the reduced 2D space
- Demonstrates that PCA can preserve class separation
- Enables visual inspection of data structure

## Key Features

### Dimensionality Reduction
- **Original Dimensions**: 30 features
- **Reduced Dimensions**: 2 principal components
- **Reduction Ratio**: 93.3% reduction (30 → 2)
- **Information Retention**: First 2 components capture significant variance

### Data Preprocessing
- Feature scaling (StandardScaler)
- Proper data preparation
- No missing value handling needed

### Visualization
- 2D scatter plot of principal components
- Color-coded by class labels
- Clear visualization of data structure

## Files
- `Principal_Component_Analysis.ipynb` - Jupyter notebook containing:
  - Data loading from scikit-learn
  - Data preprocessing (scaling)
  - PCA implementation
  - Visualization of principal components

## Dependencies
```python
pandas      # Data manipulation
numpy       # Numerical computing
scikit-learn # Machine learning library (PCA, datasets)
matplotlib  # Data visualization
seaborn     # Statistical visualization
```

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage
1. Open the Jupyter notebook: `Principal_Component_Analysis.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load the Breast Cancer dataset
   - Scale the features
   - Apply PCA with 2 components
   - Visualize the transformed data

## Code Example

Here's the complete code implementation:

```python
# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Breast Cancer dataset
Cancer_data = load_breast_cancer()

# Create DataFrame from the data
data = pd.DataFrame(Cancer_data["data"], columns=Cancer_data["feature_names"])

# Standardize the features
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)

# Apply PCA with 2 components
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(x=data_pca[:, 0], y=data_pca[:, 1], c=Cancer_data["target"], cmap="plasma")
plt.title("Breast Cancer Dataset PCA")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(label="Target (0=Malignant, 1=Benign)")
plt.show()
```

### Code Explanation

**Step 1: Import Libraries**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
- `pandas` and `numpy` for data manipulation
- `matplotlib` and `seaborn` for visualization
- `sklearn` for dataset, preprocessing, and PCA implementation

**Step 2: Load Dataset**
```python
Cancer_data = load_breast_cancer()
data = pd.DataFrame(Cancer_data["data"], columns=Cancer_data["feature_names"])
```
- Loads the Breast Cancer Wisconsin dataset
- Creates a pandas DataFrame with 569 samples and 30 features

**Step 3: Standardize Features**
```python
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
```
- Standardizes features to have mean=0 and std=1
- Essential for PCA as it's sensitive to feature scales

**Step 4: Apply PCA**
```python
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)
```
- Reduces 30 features to 2 principal components
- `fit_transform()` computes components and transforms data in one step

**Step 5: Visualize Results**
```python
plt.figure(figsize=(8, 6))
plt.scatter(x=data_pca[:, 0], y=data_pca[:, 1], c=Cancer_data["target"], cmap="plasma")
plt.title("Breast Cancer Dataset PCA")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(label="Target (0=Malignant, 1=Benign)")
plt.show()
```
- Creates a 2D scatter plot of the transformed data
- Colors points by target labels (Malignant/Benign)
- Shows how well PCA preserves class separation

## Advantages of PCA
- **Dimensionality Reduction**: Reduces number of features
- **Noise Reduction**: Removes noise by focusing on principal components
- **Visualization**: Enables visualization of high-dimensional data
- **Feature Decorrelation**: Creates uncorrelated features
- **Variance Preservation**: Retains maximum variance
- **Unsupervised**: Doesn't require labels
- **Fast**: Efficient computation

## Disadvantages of PCA
- **Linear Assumption**: Assumes linear relationships
- **Information Loss**: Some information is lost in reduction
- **Interpretability**: Principal components may be hard to interpret
- **Scale Sensitive**: Requires feature scaling
- **Not Always Beneficial**: May not help if features are already uncorrelated

## When to Use PCA
- High-dimensional datasets (many features)
- Visualization of high-dimensional data
- Reducing computational cost
- Removing multicollinearity
- Feature extraction
- Data compression
- Noise reduction
- Before applying other ML algorithms

## Understanding Principal Components

### Principal Component 1 (PC1)
- **Direction**: Direction of maximum variance
- **Variance Explained**: Captures the most variance in the data
- **Interpretation**: Most important dimension of variation

### Principal Component 2 (PC2)
- **Direction**: Direction of second maximum variance (orthogonal to PC1)
- **Variance Explained**: Captures second most variance
- **Interpretation**: Second most important dimension

### Explained Variance
- **Total Variance**: Sum of all eigenvalues
- **Variance Ratio**: Proportion of variance explained by each component
- **Cumulative Variance**: Total variance explained by first n components

## Choosing Number of Components

### Methods
1. **Fixed Number**: Specify exact number (e.g., n_components=2)
2. **Variance Threshold**: Keep components that explain X% variance (e.g., 0.95)
3. **Elbow Method**: Plot explained variance vs. number of components

### Guidelines
- **Visualization**: Use 2 or 3 components
- **Feature Reduction**: Keep components explaining 80-95% variance
- **Computational Efficiency**: Reduce to manageable number
- **Information Retention**: Balance between reduction and information loss

## Applications of PCA

### Machine Learning
- **Preprocessing**: Reduce features before training models
- **Visualization**: Visualize high-dimensional data
- **Feature Engineering**: Create new features from principal components

### Data Analysis
- **Exploratory Data Analysis**: Understand data structure
- **Noise Reduction**: Remove noise from data
- **Data Compression**: Compress data while preserving information

### Domain-Specific
- **Image Processing**: Reduce image dimensions
- **Genomics**: Analyze gene expression data
- **Finance**: Analyze stock market data
- **Signal Processing**: Reduce signal dimensions

## Future Enhancements
- Explained variance ratio calculation and visualization
- Scree plot (eigenvalues vs. components)
- Cumulative variance plot
- Component analysis (which original features contribute most)
- Comparison with different numbers of components
- Application to other datasets
- Integration with classification/regression models
- Kernel PCA for non-linear dimensionality reduction
- Incremental PCA for large datasets

## Mathematical Details

### Covariance Matrix
- **Formula**: C = (1/n) × X^T × X (for standardized data)
- **Size**: p × p where p = number of features
- **Properties**: Symmetric, positive semi-definite

### Eigenvalue Decomposition
- **Equation**: C × v = λ × v
- **Eigenvalues (λ)**: Variance along each principal component
- **Eigenvectors (v)**: Directions of principal components

### Transformation
- **Formula**: Y = X × W
- **Y**: Transformed data (n × k)
- **X**: Original data (n × p)
- **W**: Principal components matrix (p × k)

## References
- Scikit-learn Documentation: 
  - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  - [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- "An Introduction to Statistical Learning" by James et al.
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Pattern Recognition and Machine Learning" by Christopher Bishop

