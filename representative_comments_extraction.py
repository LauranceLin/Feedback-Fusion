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
from collections import Counter

# 超參數
n_clusters = 5  # 固定分群數
n_representative_comments = 3  # 每個群組提取的代表性評論數量
pmi_threshold = 2  # PMI 閾值
min_freq = 3  # 最小詞頻

# 載入資料
file_path = 'dataset_3.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
df = df[df['stars'] < 3]
documents = df['text'].dropna()

# 停用詞載入
stopwords_path = 'chinese_stopwords.txt'
with open(stopwords_path, "r", encoding="utf-8") as f:
    chinese_stopwords = set(f.read().splitlines())

# 過濾非中文資料
def contains_chinese(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_count = len(chinese_characters)
    total_count = len(text)
    return chinese_count > total_count / 2

documents = documents[documents.apply(contains_chinese)]

# 清理詞內停用字
def clean_word(word, stop_words):
    return ''.join(char for char in word if char not in stop_words)

# 自定義詞典生成
def generate_custom_dict(comments, pmi_threshold, min_freq): 
    segmented_words = [
        word for comment in comments for word in jieba.lcut(comment) if len(word) > 1
    ]
    filtered_words = [
        clean_word(word, chinese_stopwords) for word in segmented_words if clean_word(word, chinese_stopwords) not in chinese_stopwords
    ]
    word_freq = Counter(filtered_words)

    pair_freq = Counter()
    for comment in comments:
        tokens = [
            clean_word(word, chinese_stopwords) for word in jieba.lcut(comment) if clean_word(word, chinese_stopwords) not in chinese_stopwords
        ]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1

    total_words = sum(word_freq.values())
    total_pairs = sum(pair_freq.values())

    def calculate_pmi(pair, freq_dict, total_count):
        prob_pair = freq_dict[pair] / total_count if freq_dict[pair] > 0 else 1e-10
        prob_w1 = word_freq[pair[0]] / total_words if word_freq[pair[0]] > 0 else 1e-10
        prob_w2 = word_freq[pair[1]] / total_words if word_freq[pair[1]] > 0 else 1e-10
        return prob_pair / (prob_w1 * prob_w2)

    custom_terms = [
        "".join(pair) for pair in pair_freq
        if calculate_pmi(pair, pair_freq, total_pairs) > pmi_threshold and pair_freq[pair] >= min_freq
    ]

    dict_file = "custom_dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for term in custom_terms:
            f.write(f"{term} 10\n")
    print(f"自定義詞典已生成，包含 {len(custom_terms)} 個詞彙")
    return dict_file

custom_dict = generate_custom_dict(documents, pmi_threshold, min_freq)
jieba.load_userdict(custom_dict)

# 文本處理
def tokenize_and_remove_stopwords(text):
    words = jieba.cut(text)
    return [
        word for word in words
        if word not in chinese_stopwords and len(word) > 1
    ]

tokenized_documents = Parallel(n_jobs=-1)(
    delayed(tokenize_and_remove_stopwords)(text) for text in documents
)

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
        cluster_center = np.mean(cluster_vectors.toarray(), axis=0)
        similarities = cosine_similarity([cluster_center], cluster_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:n_representative]
        representative_comments[cluster_id] = original_documents.iloc[
            [cluster_indices[idx] for idx in top_indices]
        ].tolist()
    return representative_comments

representative_comments = extract_representative_comments(tfidf_matrix, clusters, documents, n_representative_comments)

# 視覺化
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

# 儲存結果
output_data = {
    cluster_id: comments for cluster_id, comments in representative_comments.items()
}
representative_df = pd.DataFrame.from_dict(output_data, orient='index')
representative_df.columns = [f"Comment {i+1}" for i in range(n_representative_comments)]
output_file = 'representative_comments.xlsx'
representative_df.to_excel(output_file, index=True)

print(f"代表性評論已儲存到 {output_file}")
