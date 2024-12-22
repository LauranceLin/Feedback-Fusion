import jieba
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from collections import Counter
from kneed import KneeLocator 
import matplotlib.pyplot as plt

# 超參數
n_gram = 10  # 最大連結數
mi_threshold = 0.5  # MI 閾值
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

# 清理詞內停用字並移除換行符
def clean_word(word, stop_words):
    return ''.join(char for char in word if char not in stop_words and char not in ['\n', '\r', '\t', ' '])

def generate_custom_dict(comments, mi_threshold, min_freq, n_gram): 
    segmented_words = [
        word for comment in comments for word in jieba.lcut(comment) if len(word) > 1
    ]
    filtered_words = [
        clean_word(word, chinese_stopwords) for word in segmented_words if clean_word(word, chinese_stopwords) not in chinese_stopwords
    ]
    word_freq = Counter(filtered_words)

    ngram_freq = {n: Counter() for n in range(2, n_gram + 1)}  # 初始化 2-gram 到 n-gram 的頻率統計

    for comment in comments:
        tokens = [
            clean_word(word, chinese_stopwords) for word in jieba.lcut(comment) if clean_word(word, chinese_stopwords) not in chinese_stopwords
        ]
        for n in range(2, n_gram + 1):  # 遍歷 2 到 n-gram
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_freq[n][ngram] += 1

    total_words = sum(word_freq.values())
    total_ngram_counts = {n: sum(ngram_freq[n].values()) for n in range(2, n_gram + 1)}

    def calculate_mi(group, freq_dict, total_count):
        joint_prob = freq_dict[group] / total_count if freq_dict[group] > 0 else 1e-10
        individual_probs = [word_freq[w] / total_words if word_freq[w] > 0 else 1e-10 for w in group]
        marginal_prob = sum(individual_probs) / len(individual_probs)
        return joint_prob / marginal_prob

    custom_ngrams = []
    for n in range(2, n_gram + 1):  # 遍歷每個 n-gram
        custom_ngrams += [
            "".join(ngram) for ngram in ngram_freq[n]
            if calculate_mi(ngram, ngram_freq[n], total_ngram_counts[n]) > mi_threshold
            and ngram_freq[n][ngram] >= min_freq
            and contains_chinese("".join(ngram))
        ]

    # 印出所有的自定義詞
    print("自定義詞列表:", end=' ')
    print(custom_ngrams)

    dict_file = "custom_dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for term in custom_ngrams:
            f.write(f"{term} 10\n")
    print(f"自定義詞典已生成，包含 {len(custom_ngrams)} 個詞彙")
    return dict_file

custom_dict = generate_custom_dict(documents, mi_threshold, min_freq, n_gram)
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

def determine_optimal_k(tfidf_matrix, max_k=15):
    distortions = []
    K = range(2, max_k + 1)  # 測試的 k 範圍

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_matrix)
        distortions.append(kmeans.inertia_)  # SSE (Sum of Squared Errors)

    # 找到肘部點
    kneedle = KneeLocator(K, distortions, curve="convex", direction="decreasing")
    optimal_k = kneedle.knee

    if optimal_k is None:
        print("Warning: Unable to find a clear elbow point. Using default k = 8.")
        optimal_k = 8  # 默認值，可根據需要設置

    # 視覺化肘部法
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, marker='o', label="SSE")
    if optimal_k is not None:
        plt.axvline(optimal_k, linestyle="--", color="r", label=f"Optimal k = {optimal_k}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE")
    plt.title("Elbow Method for Optimal k")
    plt.legend()
    plt.show()

    return optimal_k

# 計算最佳的 k 值
optimal_k = determine_optimal_k(tfidf_matrix)

# 分群
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# 提取每個 cluster 的評論
cluster_comments = {cluster_id: [] for cluster_id in range(optimal_k)}
for idx, cluster_id in enumerate(clusters):
    cluster_comments[cluster_id].append(documents.iloc[idx])

# 對每個 cluster 處理
top_keywords_by_cluster = {}

for cluster_id, comments in cluster_comments.items():
    # 為每個 cluster 生成自定義詞典
    custom_dict_path = generate_custom_dict(comments, mi_threshold, min_freq, n_gram)
    jieba.initialize()
    jieba.load_userdict(custom_dict_path)

    # 將原始評論進行 tokenize
    tokenized_comments = [" ".join(tokenize_and_remove_stopwords(c)) for c in comments]

    # 提取 Top 3 關鍵字
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(tokenized_comments)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(word_count_matrix)
    words = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [words[idx] for idx in sorted_indices[:3]]
    top_keywords_by_cluster[cluster_id] = top_keywords

# 輸出每個 cluster 的 Top 3 關鍵字
for cluster_id, keywords in top_keywords_by_cluster.items():
    print(f"Cluster {cluster_id} Top 3 Keywords: {', '.join(keywords)}")