# Clustering & Dimensionality Reduction on Wholesale Customer Data
# Assignment: Week 02 - Unsupervised Learning (Digital Empowerment Network)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Step 1: Load and Explore Dataset
df = pd.read_csv("D:/Internship-DEN/Task2/Wholesale customers data.csv")
print("Dataset Loaded Successfully!")
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 2: Preprocessing
print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Drop categorical columns for clustering (Region and Channel if present)
data = df.select_dtypes(include=[np.number])

# Normalize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Dimensionality Reduction using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
print("\nExplained variance ratio:", pca.explained_variance_ratio_)

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6)
plt.title("PCA - 2D Projection of Wholesale Customer Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# Step 4: K-Means Clustering with Elbow Method
inertia = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Choose optimal k (e.g., 3 based on Elbow Method)
kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(data_scaled)

# Visualize K-Means clusters
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=k_labels, cmap='Set2')
plt.title("K-Means Clustering (k=3) on PCA-Reduced Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# Step 5: DBSCAN Clustering
dbscan = DBSCAN(eps=2, min_samples=5)
db_labels = dbscan.fit_predict(data_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=db_labels, cmap='Accent')
plt.title("DBSCAN Clustering on PCA-Reduced Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# Step 6: Clustering Evaluation
print("\nEvaluation Scores:")
print("K-Means Silhouette Score:", silhouette_score(data_scaled, k_labels))
print("K-Means DB Index:", davies_bouldin_score(data_scaled, k_labels))

if len(set(db_labels)) > 1:
    print("DBSCAN Silhouette Score:", silhouette_score(data_scaled, db_labels))
    print("DBSCAN DB Index:", davies_bouldin_score(data_scaled, db_labels))
else:
    print("DBSCAN formed only 1 cluster or all noise.")

print("\n--- Script Execution Completed ---")
