import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import re

# 超參數
max_clusters = 6  
silhouette_scores_weight = 0.7
calinski_scores_weight = 0.3
keyword_number = 5

# 載入 Excel 檔案並讀取 text 欄位
file_path = 'NTU Reviews Dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
documents = df['text'].dropna()

# Load Chinese stop words
stopwords_file = 'chinese_stopwords.txt'  # Replace with the path to your stopwords file
with open(stopwords_file, encoding='utf-8') as f:
    chinese_stopwords = set(f.read().splitlines())

# Tokenize text using jieba and remove stopwords
def tokenize_and_remove_stopwords(text):
    words = jieba.cut(text)
    return ' '.join(word for word in words if word not in chinese_stopwords)

tokenized_documents = documents.apply(tokenize_and_remove_stopwords)

# TF-IDF 向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_documents)

# 計算每個點到其他點的距離
knn = NearestNeighbors(n_neighbors=10)  # 設定最近鄰為 10
knn.fit(X.toarray())
distances, _ = knn.kneighbors(X.toarray())

# 計算每個點的平均距離
avg_distances = distances.mean(axis=1)

# 設定一個距離閾值（可以根據需要調整）
distance_threshold = np.percentile(avg_distances, 90)  # 取距離的 90 百分位作為閾值

# 過濾掉那些超過閾值的異常值
outlier_indices = np.where(avg_distances > distance_threshold)[0]

# 使用 reset_index 來保證索引的一致性
documents_reset = documents.reset_index(drop=True)

# 刪除異常值
filtered_documents = documents_reset.drop(outlier_indices, errors='ignore')

# 重新進行 TF-IDF 向量化
tokenized_documents_filtered = filtered_documents.apply(tokenize_and_remove_stopwords)
X_filtered = vectorizer.fit_transform(tokenized_documents_filtered)

# 重新計算距離矩陣和層次聚類
distance_matrix_filtered = pdist(X_filtered.toarray(), metric='euclidean')
linkage_matrix_filtered = linkage(distance_matrix_filtered, method='ward')

# 遍歷可能的群數
silhouette_scores_filtered = []
calinski_scores_filtered = []

for n_clusters in range(2, max_clusters + 1):
    clusters = fcluster(linkage_matrix_filtered, t=n_clusters, criterion='maxclust')
    
    # 計算 Silhouette 分數
    silhouette_avg = silhouette_score(X_filtered.toarray(), clusters)
    silhouette_scores_filtered.append(silhouette_avg)
    
    # 計算 Calinski-Harabasz 指數
    calinski_avg = calinski_harabasz_score(X_filtered.toarray(), clusters)
    calinski_scores_filtered.append(calinski_avg)

# 正規化分數
silhouette_scores_norm_filtered = (silhouette_scores_filtered - np.min(silhouette_scores_filtered)) / (np.max(silhouette_scores_filtered) - np.min(silhouette_scores_filtered))
calinski_scores_norm_filtered = (calinski_scores_filtered - np.min(calinski_scores_filtered)) / (np.max(calinski_scores_filtered) - np.min(calinski_scores_filtered))

# 加權平均計算最佳群數
combined_scores_filtered = silhouette_scores_weight * silhouette_scores_norm_filtered + calinski_scores_weight * calinski_scores_norm_filtered
optimal_clusters_filtered = np.argmax(combined_scores_filtered) + 2  # 群數範圍從 2 開始

# 產生最終的群組標籤
final_clusters_filtered = fcluster(linkage_matrix_filtered, t=optimal_clusters_filtered, criterion='maxclust')

# 顯示最佳群數和每篇文件的群組標籤
print(f"最佳群數 (去除異常值後): {optimal_clusters_filtered}")
print("\n每篇文件的群組標籤 (去除異常值後):")
for i, cluster in enumerate(final_clusters_filtered):
    print(f"文件 {i + 1}: 群組 {cluster}")

# 提取每個群組的關鍵詞
def get_top_keywords(tfidf_matrix, labels, vectorizer, n_keywords=keyword_number):
    keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for cluster in set(labels):
        # 找出屬於當前群組的文檔索引
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
        if not cluster_indices:
            continue

        # 計算該群組的平均 TF-IDF 向量
        cluster_matrix = tfidf_matrix[cluster_indices].mean(axis=0).A1
        # 找出最高權重的特徵索引
        top_indices = cluster_matrix.argsort()[-n_keywords:][::-1]
        # 映射到特徵名稱
        top_keywords = [feature_names[i] for i in top_indices]
        # 確保有固定數量的關鍵詞
        keywords[cluster] = top_keywords[:n_keywords] + [None] * (n_keywords - len(top_keywords))
    return keywords

# 提取關鍵詞
top_keywords_filtered = get_top_keywords(X_filtered, final_clusters_filtered, vectorizer)

# 儲存關鍵詞到 Excel
keywords_df_filtered = pd.DataFrame.from_dict(top_keywords_filtered, orient='index', columns=[f'Keyword {i+1}' for i in range(keyword_number)])
output_file_keywords_filtered = 'filtered_cluster_keywords.xlsx'
keywords_df_filtered.to_excel(output_file_keywords_filtered, index=True)

print(f"Cluster keywords (filtered) saved to {output_file_keywords_filtered}")

# 繪製群數評估結果
# plt.figure(figsize=(15, 5))
# plt.plot(range(2, max_clusters + 1), silhouette_scores_norm_filtered, label='Normalized Silhouette Score', marker='o')
# plt.plot(range(2, max_clusters + 1), calinski_scores_norm_filtered, label='Normalized CH Index', marker='o')
# plt.plot(range(2, max_clusters + 1), combined_scores_filtered, label='Combined Score', marker='o', linestyle='--')
# plt.title('Cluster Evaluation Metrics (Filtered)')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Normalized Score')
# plt.legend()
# plt.show()
