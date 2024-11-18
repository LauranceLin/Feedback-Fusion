import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Documents
documents = [
    "Apple is a technology company that designs iPhones.",
    "Google is known for its search engine and cloud computing services.",
    "Microsoft develops Windows and Office software.",
    "Tesla produces electric cars and invests in renewable energy.",
    "SpaceX focuses on space exploration and rocket development.",
    "Amazon is an e-commerce giant with a global reach.",
    "Netflix provides streaming services for movies and TV shows."
]

# Hyperparameters
max_clusters = 6  
silhouette_scores_weight = 0.7
calinski_scores_weight = 0.3

# TF-IDF vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Distance matrix & linkage clustering
distance_matrix = pdist(X.toarray(), metric='euclidean')
linkage_matrix = linkage(distance_matrix, method='ward')

# Go through possible clusters
silhouette_scores = []
calinski_scores = []

for n_clusters in range(2, max_clusters + 1):
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    
    # Silhouette Score
    silhouette_avg = silhouette_score(X.toarray(), clusters)
    silhouette_scores.append(silhouette_avg)
    
    # Calinski-Harabasz Index
    calinski_avg = calinski_harabasz_score(X.toarray(), clusters)
    calinski_scores.append(calinski_avg)

# Normalize the indices
silhouette_scores_norm = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
calinski_scores_norm = (calinski_scores - np.min(calinski_scores)) / (np.max(calinski_scores) - np.min(calinski_scores))

# Weighted average
combined_scores = silhouette_scores_weight * silhouette_scores_norm + calinski_scores_weight * calinski_scores_norm
optimal_clusters = np.argmax(combined_scores) + 2 
print(f"Optimal number of clusters: {optimal_clusters}")

# Convert the result to actual clustering
final_clusters = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
print("\nCluster labels for each document:")
for i, cluster in enumerate(final_clusters):
    print(f"Document {i + 1}: Cluster {cluster}")

# Visualization
plt.figure(figsize=(15, 5))
plt.plot(range(2, max_clusters + 1), silhouette_scores_norm, label='Normalized Silhouette Score', marker='o')
plt.plot(range(2, max_clusters + 1), calinski_scores_norm, label='Normalized CH Index', marker='o')
plt.plot(range(2, max_clusters + 1), combined_scores, label='Combined Score', marker='o', linestyle='--')
plt.title('Cluster Evaluation Metrics')
plt.xlabel('Number of Clusters')
plt.ylabel('Normalized Score')
plt.legend()
plt.show()
