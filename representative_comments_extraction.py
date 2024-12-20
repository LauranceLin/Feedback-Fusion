import jieba
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# 超參數
n_clusters = 5  # 固定分群數
n_representative_comments = 5  # 每個群組提取的代表性評論數量

# 載入資料
file_path = 'dataset_1.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')
df = df[df['stars'] <= 3]
documents = df['text'].dropna()

# 過濾非中文資料
def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

documents = documents[documents.apply(contains_chinese)]

# 載入中文停用詞
stopwords_file = 'chinese_stopwords.txt'
with open(stopwords_file, encoding='utf-8') as f:
    chinese_stopwords = set(f.read().splitlines())

# 文本處理：分詞與去停用詞
def tokenize_and_remove_stopwords(text):
    words = jieba.cut(text)
    return [
        word for word in words
        if word not in chinese_stopwords and len(word) > 1
    ]

tokenized_documents = Parallel(n_jobs=-1)(
    delayed(tokenize_and_remove_stopwords)(text) for text in documents
)

# 將分詞後的文本轉為單一字符串形式
joined_documents = [" ".join(doc) for doc in tokenized_documents]

# 使用 TF-IDF 提取文本向量
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
word_count_matrix = vectorizer.fit_transform(joined_documents)
tfidf_matrix = tfidf_transformer.fit_transform(word_count_matrix)

# 分群
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# 提取群組代表性評論
def extract_representative_comments(tfidf_matrix, labels, original_documents, n_representative):
    representative_comments = {}
    for cluster_id in set(labels):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_vectors = tfidf_matrix[cluster_indices]
        
        # 計算群組中心
        cluster_center = np.mean(cluster_vectors.toarray(), axis=0)
        similarities = cosine_similarity([cluster_center], cluster_vectors).flatten()
        
        # 找到與群組中心最相似的評論索引
        top_indices = np.argsort(similarities)[::-1][:n_representative]
        representative_comments[cluster_id] = original_documents.iloc[
            [cluster_indices[idx] for idx in top_indices]
        ].tolist()
    return representative_comments

representative_comments = extract_representative_comments(tfidf_matrix, clusters, documents, n_representative_comments)

# 結果視覺化 (t-SNE)
def visualize_clusters(tfidf_matrix, clusters):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_vectors = tsne.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_vectors[:, 0], 
        y=reduced_vectors[:, 1],
        hue=clusters,
        palette="tab10",
        legend="full"
    )
    plt.title("Cluster Visualization with t-SNE")
    plt.show()

visualize_clusters(tfidf_matrix, clusters)

# 儲存結果到 Excel
output_data = {
    cluster_id: comments for cluster_id, comments in representative_comments.items()
}
representative_df = pd.DataFrame.from_dict(output_data, orient='index')
representative_df.columns = [f"Comment {i+1}" for i in range(n_representative_comments)]
output_file = 'representative_comments.xlsx'
representative_df.to_excel(output_file, index=True)

print(f"代表性評論已儲存到 {output_file}")
