import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
from collections import Counter
import os
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# 停用詞清單（可擴展）
# stop_words = {"我", "的", "是", "和", "在", "一", "有", "了", "也", "就", "不", "都", "很", "，", "。", "？", "、", "~", "～", "-", "...", "/", "!", "！", "（", "）", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}
stopwords_path = "C:\\Users\\barry\\My_Document\\資訊檢索與文字探勘\\term_project\\stop_word_list.txt"
with open(stopwords_path, "r", encoding="utf-8") as f:
    stop_words = set(line.strip() for line in f if line.strip())


# 自定義函數：清理詞內停用字
def clean_word(word, stop_words):
    """移除詞內的停用字"""
    return ''.join(char for char in word if char not in stop_words)

# 自定義函數：生成自定義詞典
def generate_custom_dict(comments, pmi_threshold=2, min_freq=3): 
    # 分詞
    segmented_words = [
        word for comment in comments for word in jieba.lcut(comment) if len(word) > 1
    ]
    
    # 過濾停用字
    filtered_words = [
        clean_word(word, stop_words) for word in segmented_words if clean_word(word, stop_words) not in stop_words
    ]
    
    # 統計詞頻
    word_freq = Counter(filtered_words)

    # 計算雙詞共現 (Bigram)
    pair_freq = Counter()
    # 計算三詞共現 (Trigram)
    trigram_freq = Counter()
    
    for comment in comments:
        tokens = [
            clean_word(word, stop_words) for word in jieba.lcut(comment) if clean_word(word, stop_words) not in stop_words
        ]
        # 計算雙詞共現
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1
        # 計算三詞共現
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
            trigram_freq[trigram] += 1

    total_words = sum(word_freq.values())
    total_pairs = sum(pair_freq.values())
    total_trigrams = sum(trigram_freq.values())

    # PMI 計算函數
    def calculate_pmi(pair, freq_dict, total_count):
        prob_pair = freq_dict[pair] / total_count if freq_dict[pair] > 0 else 1e-10
        prob_w1 = word_freq[pair[0]] / total_words if word_freq[pair[0]] > 0 else 1e-10
        prob_w2 = word_freq[pair[1]] / total_words if word_freq[pair[1]] > 0 else 1e-10
        if len(pair) == 2:
            return prob_pair / (prob_w1 * prob_w2)
        elif len(pair) == 3:
            prob_w3 = word_freq[pair[2]] / total_words if word_freq[pair[2]] > 0 else 1e-10
            return prob_pair / (prob_w1 * prob_w2 * prob_w3)

    # 篩選高 PMI 的雙詞組
    custom_terms = [
        "".join(pair) for pair in pair_freq
        if calculate_pmi(pair, pair_freq, total_pairs) > pmi_threshold and pair_freq[pair] >= min_freq
    ]
    
    # 篩選高 PMI 的三詞組
    custom_trigrams = [
        "".join(trigram) for trigram in trigram_freq
        if calculate_pmi(trigram, trigram_freq, total_trigrams) > pmi_threshold and trigram_freq[trigram] >= min_freq
    ]
    
    # 合併雙詞與三詞組
    all_terms = custom_terms + custom_trigrams
    # all_terms = custom_terms

    # 保存詞典
    dict_file = "custom_dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for term in all_terms:
            f.write(f"{term} 10\n")  # 預設詞頻為 10
    print(f"自定義詞典已生成，包含 {len(all_terms)} 個詞彙")
    return dict_file

def split_sentences(comment):
    """將評論分割為多句話，包括換行符號作為句號"""
    import re
    # 使用正則表達式分割句子，包含標點符號和換行符號
    sentences = re.split(r'(?<=[。！？])', comment)
    # 移除空白和多餘的換行符號
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Load the dataset
file_path = "C:\\Users\\barry\\My_Document\\資訊檢索與文字探勘\\hotpot_reviews_dataset.xlsx"
data = pd.read_excel(file_path)

# Filter for Chinese comments
chinese_comments = data['text'].dropna()

# 生成自定義詞典
custom_dict = generate_custom_dict(chinese_comments)

# 加載自定義詞典
jieba.load_userdict(custom_dict)

tokenized_sentences = chinese_comments.apply(
    lambda comment: [
        ''.join(  # 直接拼接，不加空格
            clean_word(word, stop_words) 
            for word in jieba.cut(sentence, cut_all=False) 
            if clean_word(word, stop_words).strip()  # 過濾空字符串和空白
        )
        for sentence in split_sentences(comment)
    ]
)

# 如果需要展平 (每句話一列)
tokenized_comments = tokenized_sentences.explode().dropna().reset_index(drop=True)

tokenized_comments = tokenized_comments.dropna().reset_index(drop=True)
tokenized_comments = tokenized_comments[tokenized_comments.str.strip() != ''].reset_index(drop=True)
print(tokenized_comments)

# Generate TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(tokenized_comments)

# Spectral Clustering
similarity_matrix = cosine_similarity(tfidf_matrix)
num_clusters = 5
spectral = SpectralClustering(
    n_clusters=num_clusters,
    affinity='precomputed',
    random_state=42
)
labels = spectral.fit_predict(similarity_matrix)

# Add the cluster labels to the dataset
final_data = pd.DataFrame({'text': tokenized_comments})
final_data['cluster'] = -1  # Default for missing or non-Chinese comments
final_data.loc[tokenized_comments.index, 'cluster'] = labels

# Extract keywords for each cluster
def extract_cluster_keywords(tfidf_matrix, labels, vectorizer, n_keywords=20):
    feature_names = vectorizer.get_feature_names_out()
    num_clusters = len(set(labels))
    cluster_keywords = {}
    for cluster in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
        if not cluster_indices:
            continue
        cluster_matrix = tfidf_matrix[cluster_indices].mean(axis=0).A1
        top_indices = cluster_matrix.argsort()[-n_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        cluster_keywords[cluster] = keywords
    return cluster_keywords

keywords = extract_cluster_keywords(tfidf_matrix, labels, vectorizer)

# Save the results
output_file = 'clustered_reviews.xlsx'
final_data.to_excel(output_file, index=False)

# Save keywords to another file
keywords_df = pd.DataFrame.from_dict(keywords, orient='index', columns=[f'Keyword {i+1}' for i in range(20)])
keywords_file = 'cluster_keywords.xlsx'
keywords_df.to_excel(keywords_file)

print(f"Clustering complete. Results saved to {output_file}")
print(f"Cluster keywords saved to {keywords_file}")
