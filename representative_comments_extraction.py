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
n_clusters = 10  # 固定分群數
mi_threshold = 0.3  # MI 閾值
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
def generate_custom_dict(comments, mi_threshold, min_freq): 
    segmented_words = [
        word for comment in comments for word in jieba.lcut(comment) if len(word) > 1
    ]
    filtered_words = [
        clean_word(word, chinese_stopwords) for word in segmented_words if clean_word(word, chinese_stopwords) not in chinese_stopwords
    ]
    word_freq = Counter(filtered_words)

    pair_freq = Counter()
    trigram_freq = Counter()
    fourgram_freq = Counter()
    for comment in comments:
        tokens = [
            clean_word(word, chinese_stopwords) for word in jieba.lcut(comment) if clean_word(word, chinese_stopwords) not in chinese_stopwords
        ]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
            trigram_freq[trigram] += 1
        for i in range(len(tokens) - 3):
            fourgram = (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
            fourgram_freq[fourgram] += 1

    total_words = sum(word_freq.values())
    total_pairs = sum(pair_freq.values())
    total_trigrams = sum(trigram_freq.values())
    total_fourgrams = sum(fourgram_freq.values())

    def calculate_mi(group, freq_dict, total_count):
        joint_prob = freq_dict[group] / total_count if freq_dict[group] > 0 else 1e-10
        individual_probs = [word_freq[w] / total_words if word_freq[w] > 0 else 1e-10 for w in group]
        marginal_prob = sum(individual_probs) / len(individual_probs)
        return joint_prob / marginal_prob

    custom_bigrams = [
        "".join(pair) for pair in pair_freq
        if calculate_mi(pair, pair_freq, total_pairs) > mi_threshold and pair_freq[pair] >= min_freq
        and contains_chinese("".join(pair))
    ]
    custom_trigrams = [
        "".join(trigram) for trigram in trigram_freq
        if calculate_mi(trigram, trigram_freq, total_trigrams) > mi_threshold and trigram_freq[trigram] >= min_freq
        and contains_chinese("".join(trigram))
    ]
    custom_fourgrams = [
        "".join(fourgram) for fourgram in fourgram_freq
        if calculate_mi(fourgram, fourgram_freq, total_fourgrams) > mi_threshold and fourgram_freq[fourgram] >= min_freq
        and contains_chinese("".join(fourgram))
    ]

    all_terms = custom_bigrams + custom_trigrams + custom_fourgrams

    dict_file = "custom_dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for term in all_terms:
            f.write(f"{term} 10\n")
    print(f"自定義詞典已生成，包含 {len(all_terms)} 個詞彙")
    return dict_file

custom_dict = generate_custom_dict(documents, mi_threshold, min_freq)
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

# 提取每個 cluster 的評論
cluster_comments = {cluster_id: [] for cluster_id in range(n_clusters)}
for idx, cluster_id in enumerate(clusters):
    cluster_comments[cluster_id].append(documents.iloc[idx])

for id in range(len(cluster_comments)):
    print(str(id) + ': ' + str(len(cluster_comments[id])))

# 對每個 cluster 處理
top_keywords_by_cluster = {}

for cluster_id, comments in cluster_comments.items():
    # 為每個 cluster 生成自定義詞典
    custom_dict_path = generate_custom_dict(comments, mi_threshold, min_freq)
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