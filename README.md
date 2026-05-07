# Clustering Module

## What I Learned
- Clustering is unsupervised ML: groups similar data points without labels.
- **K-Means**: Choose k, assign points to nearest centroids, update until stable. Evaluated with **WCSS** (Within-Cluster Sum of Squares).
- **Hierarchical (Agglomerative)**: Bottom-up merging using linkage (Ward, Average, Complete) and distances (Euclidean, Manhattan, Cosine).
- Evaluated clusters with **Silhouette score** (-1 to +1). Closer to **+1** = better separated clusters.
- Used PCA for visualization.

## Key Concepts
- K-Means is scalable but needs predefined k.
- Hierarchical builds tree of clusters, no need to set k upfront.

## Personal Note
Silhouette score closer to **1** is better. It is a numerical metric, not case-sensitive, and applies to any dataset.


## Requirements
```bash
pip install -r requirements.txt
```


