import jieba
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# 超參數
n_clusters = 6  # 固定分群數
n_representative_comments = 5  # 每個群組提取的代表性評論數量

# 載入資料
file_path = 'hotpot_reviews_dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
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

# 結合 TF-IDF 過濾低權重詞
def filter_by_tfidf(tokenized_docs):
    joined_docs = [" ".join(doc) for doc in tokenized_docs]
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    word_count = vectorizer.fit_transform(joined_docs)
    tfidf_matrix = tfidf_transformer.fit_transform(word_count)
    
    # 過濾 TF-IDF 權重最低的詞
    tfidf_scores = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    filtered_docs = []
    for i, doc in enumerate(tokenized_docs):
        filtered_docs.append([
            word for word in doc if word in tfidf_scores.columns and tfidf_scores.loc[i, word] > 0.1
        ])
    return filtered_docs

filtered_documents = filter_by_tfidf(tokenized_documents)

# 使用 BERT 模型提取文檔嵌入向量
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese").to(device)

def compute_bert_embeddings(documents):
    batch_size = 16
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch_texts = [" ".join(doc) for doc in documents[i:i+batch_size]]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(cls_embeddings)
    return np.array(embeddings)

doc_vectors = compute_bert_embeddings(filtered_documents)

# 分群
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(doc_vectors)

# 提取群組代表性評論
def extract_representative_comments(doc_vectors, labels, original_documents, n_representative):
    representative_comments = {}
    for cluster_id in set(labels):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_vectors = doc_vectors[cluster_indices]
        
        # 計算群組中心
        cluster_center = np.mean(cluster_vectors, axis=0)
        similarities = cosine_similarity([cluster_center], cluster_vectors).flatten()
        
        # 找到與群組中心最相似的評論索引
        top_indices = np.argsort(similarities)[::-1][:n_representative]
        representative_comments[cluster_id] = original_documents.iloc[
            [cluster_indices[idx] for idx in top_indices]
        ].tolist()
    return representative_comments

representative_comments = extract_representative_comments(doc_vectors, clusters, documents, n_representative_comments)

# 結果視覺化 (t-SNE)
def visualize_clusters(doc_vectors, clusters):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_vectors = tsne.fit_transform(doc_vectors)
    
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

visualize_clusters(doc_vectors, clusters)

# 儲存結果到 Excel
output_data = {
    cluster_id: comments for cluster_id, comments in representative_comments.items()
}
representative_df = pd.DataFrame.from_dict(output_data, orient='index')
representative_df.columns = [f"Comment {i+1}" for i in range(n_representative_comments)]
output_file = 'representative_comments_fixed_clusters.xlsx'
representative_df.to_excel(output_file, index=True)

print(f"代表性評論已儲存到 {output_file}")
