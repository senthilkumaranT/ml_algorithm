# Clustering Algorithms

## Overview
This folder contains comprehensive implementations of three fundamental clustering algorithms: K-Means Clustering, Hierarchical Clustering, and DBSCAN. Clustering is an unsupervised machine learning technique used to group similar data points together without labeled data.

## Algorithms Implemented

### 1. K-Means Clustering
### 2. Hierarchical Clustering (Agglomerative)
### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

---

## 1. K-Means Clustering

### Algorithm Description
**K-Means** is a centroid-based clustering algorithm that partitions data into k clusters by minimizing the within-cluster sum of squares (WCSS). It iteratively assigns data points to the nearest centroid and updates centroids based on assigned points.

### Mathematical Foundation
- **Objective Function**: Minimize WCSS = Σᵢ Σⱼ ||xᵢⱼ - cⱼ||²
- **Centroid Update**: cⱼ = (1/|Sⱼ|) × Σᵢ∈Sⱼ xᵢ
- **Distance Metric**: Euclidean distance

### Key Characteristics
- **Centroid-based**: Each cluster has a centroid (center point)
- **Requires k**: Must specify number of clusters beforehand
- **Fast**: Efficient for large datasets
- **Sensitive to initialization**: Different initializations can yield different results
- **Assumes spherical clusters**: Works best with circular/spherical cluster shapes

### Implementation Details

#### Dataset
- **Type**: Synthetic blob dataset
- **Samples**: 150 data points
- **Features**: 2 features (for 2D visualization)
- **Clusters**: 3 clusters
- **Generation**: Created using `make_blobs` from scikit-learn
- **Cluster Standard Deviation**: 1.5
- **Random State**: 1 (for reproducibility)

#### Code Implementation

```python
# Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Generate synthetic data
x, y = make_blobs(n_samples=150, n_features=2, centers=3, 
                  cluster_std=1.5, random_state=1)

# Visualize original data
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# Split data (for demonstration purposes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Elbow Method to find optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Automatic elbow detection using KneeLocator
kl = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
print(f"The optimal number of clusters is: {kl.elbow}")

# Silhouette Score Analysis
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(x_train)
    score = silhouette_score(x_train, kmeans.labels_)
    silhouette_coefficients.append(score)

# Train K-Means with optimal k
kmeans = KMeans(n_clusters=kl.elbow, init='k-means++')
kmeans.fit(x_train)

# Make predictions
y_pred = kmeans.predict(x_test)

# Visualize results
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='black', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

#### Key Steps Explained

**Step 1: Data Generation**
- Creates synthetic 2D data with 3 distinct clusters
- Uses `make_blobs` for controlled cluster generation

**Step 2: Data Preprocessing**
- Splits data into train/test sets (70/30 split)
- Standardizes features using StandardScaler
- Ensures all features are on the same scale

**Step 3: Optimal k Selection - Elbow Method**
- Tests k values from 1 to 10
- Calculates WCSS (Within-Cluster Sum of Squares) for each k
- Plots WCSS vs. number of clusters
- Uses KneeLocator to automatically detect the "elbow" point
- Elbow point indicates optimal number of clusters

**Step 4: Silhouette Analysis**
- Calculates silhouette score for k values from 2 to 10
- Silhouette score measures how similar a point is to its own cluster vs. other clusters
- Range: -1 to 1 (higher is better)
- Helps validate the optimal k choice

**Step 5: Model Training and Prediction**
- Trains K-Means with optimal k value
- Uses 'k-means++' initialization (smart centroid initialization)
- Predicts cluster assignments for test data

**Step 6: Visualization**
- Plots clustered data points colored by cluster assignment
- Displays cluster centroids as black X markers

### Advantages of K-Means
- Simple and easy to implement
- Fast and efficient for large datasets
- Works well with spherical clusters
- Guaranteed convergence
- Scales well to large number of samples

### Disadvantages of K-Means
- Requires specifying k beforehand
- Sensitive to initialization
- Assumes clusters are spherical
- Sensitive to outliers
- May converge to local minima

---

## 2. Hierarchical Clustering (Agglomerative)

### Algorithm Description
**Hierarchical Clustering** builds a hierarchy of clusters using a bottom-up (agglomerative) or top-down (divisive) approach. Agglomerative clustering starts with each point as its own cluster and merges the closest clusters iteratively.

### Mathematical Foundation
- **Linkage Criteria**: 
  - **Ward**: Minimizes variance within clusters
  - **Complete**: Maximum distance between clusters
  - **Average**: Average distance between clusters
  - **Single**: Minimum distance between clusters
- **Distance Metric**: Euclidean distance

### Key Characteristics
- **Hierarchical Structure**: Creates a dendrogram showing cluster relationships
- **No k Required**: Can determine clusters from dendrogram
- **Deterministic**: Same data always produces same dendrogram
- **Visualization**: Dendrogram provides intuitive cluster visualization
- **Flexible**: Can cut dendrogram at different levels for different k values

### Implementation Details

#### Dataset
- **Type**: Iris dataset (reduced to 2D using PCA)
- **Samples**: 150 samples
- **Features**: 4 original features → 2 principal components
- **Classes**: 3 classes (for comparison with clustering results)

#### Code Implementation

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering

# Load Iris dataset
iris = datasets.load_iris()

# Create DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(data)

# Apply PCA for dimensionality reduction (for visualization)
pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(x_scaled)

# Visualize data in 2D
plt.scatter(pca_scaled[:, 0], pca_scaled[:, 1], c=iris.target)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset (PCA Reduced)')
plt.show()

# Create Dendrogram
plt.figure(figsize=(10, 7))
sc.dendrogram(sc.linkage(pca_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
cluster.fit(pca_scaled)

# Get cluster labels
labels = cluster.labels_

# Visualize clustering results
plt.scatter(pca_scaled[:, 0], pca_scaled[:, 1], c=cluster.labels_)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Hierarchical Clustering Results')
plt.show()
```

#### Key Steps Explained

**Step 1: Data Loading and Preprocessing**
- Loads Iris dataset (150 samples, 4 features)
- Standardizes features using StandardScaler
- Applies PCA to reduce to 2D for visualization

**Step 2: Dendrogram Creation**
- Uses `scipy.cluster.hierarchy.linkage()` with Ward linkage
- Ward linkage minimizes within-cluster variance
- Creates hierarchical tree structure
- Visualizes cluster merging process

**Step 3: Cluster Extraction**
- Applies AgglomerativeClustering with n_clusters=2
- Uses Euclidean distance metric
- Uses Ward linkage method
- Assigns cluster labels to each data point

**Step 4: Visualization**
- Plots clustered data points colored by cluster assignment
- Shows how hierarchical clustering groups similar points

### Advantages of Hierarchical Clustering
- No need to specify k beforehand
- Dendrogram provides intuitive visualization
- Can handle non-spherical clusters
- Deterministic results
- Flexible cluster extraction

### Disadvantages of Hierarchical Clustering
- Computationally expensive (O(n³) for most linkage methods)
- Sensitive to noise and outliers
- Difficult to handle large datasets
- Once clusters are merged, cannot be undone
- Memory intensive

---

## 3. DBSCAN (Density-Based Spatial Clustering)

### Algorithm Description
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed together, marking points in low-density regions as outliers.

### Mathematical Foundation
- **Core Point**: Point with at least `min_samples` neighbors within `eps` distance
- **Border Point**: Point within `eps` of a core point but not a core point itself
- **Noise Point**: Point that is neither core nor border point
- **Density-Reachable**: Points connected through a chain of core points

### Key Characteristics
- **No k Required**: Automatically determines number of clusters
- **Noise Handling**: Identifies outliers as noise points
- **Arbitrary Shapes**: Can find clusters of arbitrary shapes
- **Density-Based**: Groups points based on density, not distance
- **Robust to Outliers**: Handles outliers effectively

### Implementation Details

#### Dataset
- **Type**: Synthetic moon-shaped dataset
- **Samples**: 1,000 data points
- **Features**: 2 features (for 2D visualization)
- **Shape**: Two interleaving half-circles (non-linear, non-spherical)
- **Noise**: 0.05 (5% noise level)
- **Generation**: Created using `make_moons` from scikit-learn

#### Code Implementation

```python
# Import libraries
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate synthetic moon-shaped data
X, y = make_moons(n_samples=1000, noise=0.05)

# Visualize original data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Original Data (Moon-shaped)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3)
dbscan.fit(x_scaled)

# Get cluster labels (-1 indicates noise/outliers)
labels = dbscan.labels_

# Visualize clustering results
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label (-1 = Noise)')
plt.show()

# Count clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')
```

#### Key Steps Explained

**Step 1: Data Generation**
- Creates synthetic moon-shaped dataset
- Two interleaving half-circles (non-linear pattern)
- Includes noise to make clustering more challenging

**Step 2: Data Preprocessing**
- Standardizes features using StandardScaler
- Important for DBSCAN as it uses distance-based density

**Step 3: DBSCAN Application**
- **eps (epsilon)**: 0.3 - Maximum distance between two samples to be considered neighbors
- **min_samples**: Default 5 - Minimum number of samples in a neighborhood for a point to be a core point
- Fits the model to identify clusters

**Step 4: Cluster Label Interpretation**
- **Positive integers**: Cluster IDs (0, 1, 2, ...)
- **-1**: Noise/outlier points
- Each point is assigned to a cluster or marked as noise

**Step 5: Visualization**
- Plots data points colored by cluster assignment
- Noise points (label -1) are clearly visible
- Shows how DBSCAN handles non-spherical clusters

### DBSCAN Parameters

#### eps (epsilon)
- **Definition**: Maximum distance between two samples to be considered neighbors
- **Too Small**: Many points marked as noise, many small clusters
- **Too Large**: Fewer clusters, may merge separate clusters
- **Tuning**: Use k-distance graph to find optimal eps

#### min_samples
- **Definition**: Minimum number of samples in a neighborhood for a point to be a core point
- **Too Small**: More noise points, more clusters
- **Too Large**: Fewer clusters, more noise points
- **Default**: 5 (good starting point)

### Advantages of DBSCAN
- No need to specify number of clusters
- Can find clusters of arbitrary shapes
- Handles outliers effectively
- Robust to noise
- Works well with non-linear cluster boundaries

### Disadvantages of DBSCAN
- Sensitive to parameter selection (eps, min_samples)
- Struggles with varying density clusters
- Performance depends on distance metric
- May not work well with high-dimensional data
- Border points may be assigned to different clusters in different runs

---

## Comparison of Clustering Algorithms

| Feature | K-Means | Hierarchical | DBSCAN |
|---------|---------|--------------|--------|
| **k Required** | Yes | No | No |
| **Cluster Shape** | Spherical | Arbitrary | Arbitrary |
| **Outlier Handling** | Poor | Poor | Excellent |
| **Speed** | Fast | Slow | Medium |
| **Scalability** | Excellent | Poor | Good |
| **Deterministic** | No (initialization) | Yes | Yes |
| **Memory Usage** | Low | High | Medium |
| **Best For** | Spherical clusters, known k | Small datasets, dendrogram needed | Arbitrary shapes, noise present |

## Common Techniques Used

### Data Preprocessing
- **StandardScaler**: Standardizes features to have mean=0 and std=1
- **PCA**: Dimensionality reduction for visualization
- **Train-Test Split**: For evaluation purposes (though clustering is unsupervised)

### Evaluation Methods
- **Elbow Method**: Finding optimal k for K-Means
- **Silhouette Score**: Measuring cluster quality
- **Dendrogram**: Visualizing hierarchical structure
- **Visual Inspection**: 2D scatter plots for cluster visualization

## Files
- `k Means Clustering Algorithm (1).ipynb` - K-Means implementation with elbow method and silhouette analysis
- `hierarichal Clustering Implementatio .ipynb` - Hierarchical clustering with dendrogram
- `DBSCAN.ipynb` - DBSCAN implementation for non-linear clusters

## Dependencies
```python
pandas          # Data manipulation
numpy           # Numerical computing
scikit-learn    # Machine learning library (clustering algorithms)
matplotlib      # Data visualization
seaborn         # Statistical visualization
scipy           # Scientific computing (hierarchical clustering)
kneed           # Elbow detection algorithm
```

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy kneed jupyter
```

## Usage

### K-Means Clustering
1. Open `k Means Clustering Algorithm (1).ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Generate synthetic blob data
   - Find optimal k using elbow method
   - Calculate silhouette scores
   - Train K-Means model
   - Visualize clustering results

### Hierarchical Clustering
1. Open `hierarichal Clustering Implementatio .ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load Iris dataset
   - Apply PCA for visualization
   - Create dendrogram
   - Perform agglomerative clustering
   - Visualize cluster assignments

### DBSCAN
1. Open `DBSCAN.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Generate moon-shaped synthetic data
   - Standardize features
   - Apply DBSCAN clustering
   - Visualize clusters and noise points

## When to Use Each Algorithm

### Use K-Means When:
- You know the number of clusters
- Clusters are spherical/globular
- You need fast clustering for large datasets
- You want simple, interpretable results

### Use Hierarchical Clustering When:
- You don't know the number of clusters
- You want to visualize cluster hierarchy
- You have a small to medium dataset
- You need to understand cluster relationships

### Use DBSCAN When:
- You don't know the number of clusters
- Clusters have arbitrary shapes
- You have outliers/noise in your data
- Clusters have varying densities (to some extent)

## Future Enhancements
- **K-Means++**: Improved initialization
- **Mini-Batch K-Means**: For very large datasets
- **Divisive Hierarchical Clustering**: Top-down approach
- **HDBSCAN**: Hierarchical version of DBSCAN
- **Parameter Tuning**: Automated parameter selection
- **Cluster Validation**: More evaluation metrics
- **3D Visualization**: For multi-dimensional data
- **Real-world Datasets**: Apply to actual clustering problems

## References
- Scikit-learn Documentation:
  - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  - [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
  - [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- "Introduction to Data Mining" by Tan, Steinbach, and Kumar
- "Pattern Recognition and Machine Learning" by Christopher Bishop

