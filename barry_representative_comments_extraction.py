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
from kneed import KneeLocator

# 超參數
mi_threshold = 1  # MI 閾值
min_freq = 3  # 最小詞頻
n_gram = 10
max_rand_seed = 100
max_K = 10
num_of_top_keywords = 5

# 載入資料
file_path = 'C:\\Users\\barry\\My_Document\\資訊檢索與文字探勘\\hotpot_reviews_dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

while(True):
    NorP = input('n or p: ')
    if(NorP == 'P' or NorP == 'p'):
        df = df[df['stars'] > 3]
        break
    elif(NorP == 'N' or NorP == 'n'):
        df = df[df['stars'] < 3]
        break
        
def split_sentences(comment):
    """將評論分割為多句話，包括換行符號作為句號"""
    import re
    # 使用正則表達式分割句子，包含標點符號和換行符號
    sentences = re.split(r'(?<=[。！？!?.])', comment)
    # 移除空白和多餘的換行符號
    # return [sentence.strip() for sentence in sentences if sentence.strip()]
    # 移除標點符號
    sentences = [re.sub(r'[。！？!?.\n]', '', sentence).strip() for sentence in sentences]
    # 移除空白和多餘的換行符號
    return [sentence for sentence in sentences if sentence]

documents = df['text'].dropna()


# 停用詞載入
stopwords_path = 'C:\\Users\\barry\\My_Document\\資訊檢索與文字探勘\\term_project\\stop_word_list.txt'
with open(stopwords_path, "r", encoding="utf-8") as f:
    chinese_stopwords = set(f.read().splitlines())

# 過濾非中文資料
def contains_chinese(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_count = len(chinese_characters)
    total_count = len(text)
    return chinese_count > total_count / 2

documents = documents[documents.apply(contains_chinese)]
# print(type(documents))
# documents = documents.apply(
#     lambda comment: [sentence for sentence in split_sentences(comment)]
# )
# documents =  documents[documents.apply(split_sentences)]
tokenized_sentences = documents.apply(
    lambda comment: [sentence for sentence in split_sentences(comment)]
)

# 如果需要展平 (每句話一列)
documents = tokenized_sentences.explode().dropna().reset_index(drop=True)
documents = documents[documents.str.strip() != ''].drop_duplicates().reset_index(drop=True)

# 清理詞內停用字
def clean_word(word, stop_words):
    return ''.join(char for char in word if char not in stop_words and char not in ['\n', '\r', '\t', ' '])

# 自定義詞典生成
def generate_custom_dict(comments, mi_threshold, min_freq, n_gram=10): 
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
            clean_word(word, chinese_stopwords) for word in jieba.lcut(comment) if clean_word(word, chinese_stopwords) not in chinese_stopwords and len(clean_word(word, chinese_stopwords)) > 1
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
    # print(f"自定義詞典已生成，包含 {len(custom_ngrams)} 個詞彙")
    return dict_file

custom_dict = generate_custom_dict(documents, mi_threshold, min_freq)
jieba.load_userdict(custom_dict)

# 文本處理
def tokenize_and_remove_stopwords(text):
    words = jieba.cut(text)
    return [ 
        word for word in words
        if word not in chinese_stopwords and len(word) > 1 and not word.isdigit()
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

def knee_method_with_sse(max_k=20, max_rand_seed=100):
    max_cluster_num = 0
    min_sse = float('inf')
    best_seed = 0
    for rand_seed in range(1, max_rand_seed + 1):
        # print(rand_seed)
        sse = []
        # Compute SSE (inertia) for different values of K
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=rand_seed)
            # kmeans.fit(tfidf_matrix)
            _ = kmeans.fit_predict(tfidf_matrix)
            sse.append(kmeans.inertia_)
        
        # Use KneeLocator to find the knee point
        kneedle = KneeLocator(range(1, max_k + 1), sse, curve="convex", direction="decreasing")
        optimal_k = kneedle.knee
        # print(rand_seed, optimal_k, max_cluster_num)
        if optimal_k is None:
            # print(f"seed number {rand_seed}: No knee point detected. Please check your data or adjust the K range.")
            optimal_k = 1
        elif sse[optimal_k] < min_sse:
            max_cluster_num = optimal_k
            min_sse = sse[optimal_k]
            best_seed = rand_seed
        # elif optimal_k > max_cluster_num:
        #     max_cluster_num = optimal_k
        #     best_seed = rand_seed

        # Plot the SSE graph with the optimal K highlighted
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, max_k + 1), sse, 'bx-', label="SSE (Inertia)")
        # plt.axvline(optimal_k, color='r', linestyle='--', label=f"Optimal K: {optimal_k}")
        # plt.xlabel('Number of Clusters (K)')
        # plt.ylabel('SSE (Sum of Squared Errors)')
        # plt.title('Elbow Method using SSE')
        # plt.legend()
        # plt.show()
        
    return max_cluster_num, best_seed

optimal_k, best_seed = knee_method_with_sse(max_K, max_rand_seed)

print(f"The optimal number of clusters is: {optimal_k}")
print(f"The best random seed for kmeans is: {best_seed}")

# 分群
kmeans = KMeans(n_clusters=optimal_k, random_state=best_seed)
clusters = kmeans.fit_predict(tfidf_matrix)

# 提取每個 cluster 的評論
cluster_comments = {cluster_id: [] for cluster_id in range(optimal_k)}
for idx, cluster_id in enumerate(clusters):
    cluster_comments[cluster_id].append(documents.iloc[idx])

# 對每個 cluster 處理
top_keywords_by_cluster = {}

for cluster_id, comments in cluster_comments.items():
    if not comments: 
        continue

    # 為每個 cluster 生成自定義詞典
    custom_dict_path = generate_custom_dict(comments, mi_threshold, min_freq)
    jieba.load_userdict(custom_dict_path)

    # 將原始評論進行 tokenize
    tokenized_comments = [" ".join(tokenize_and_remove_stopwords(c)) for c in comments]
    # print(cluster_id, comments)
    # 提取 Top 3 關鍵字
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(tokenized_comments)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(word_count_matrix)
    words = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [words[idx] for idx in sorted_indices[:num_of_top_keywords]]
    top_keywords_by_cluster[cluster_id] = top_keywords

    # tokenized_comments_score = [0] * len(tokenized_comments)
    # for i, tokenized_comment in enumerate(tokenized_comments):
    #     for idx in sorted_indices[:10]:
    #         if words[idx] in tokenized_comment:
    #             tokenized_comments_score[i] += tfidf_scores[idx]

    # sorted_tokenized_comments_score = np.argsort(tokenized_comments_score)[::-1]
    # print(f"Cluster {cluster_id}: ")
    # for idx in sorted_tokenized_comments_score:
    #     print(f"'{tokenized_comments[idx]}'")
    #     if idx < 3:
    #         break

print()

# 輸出每個 cluster 的 Top N 關鍵字
for cluster_id, keywords in top_keywords_by_cluster.items():
    print(f"Cluster {cluster_id} Top {num_of_top_keywords} Keywords: {', '.join(keywords)}")

# class PATNode:
#     def __init__(self):
#         self.children = {}
#         self.data = set()  # Use set to avoid duplicate indices

# class PATTree:
#     def __init__(self):
#         self.root = PATNode()

#     def insert(self, words, index):
#         """
#         Insert a segmented comment into the PAT tree.
#         Each word directly connects to the root, and the rest form a tree structure.
#         """
#         for i in range(len(words)):
#             node = self.root
#             for j in range(i, len(words)):
#                 word = words[j]
#                 if word not in node.children:
#                     node.children[word] = PATNode()
#                 node = node.children[word]
#                 node.data.add(index)  # Add index to the node's data

#     def build_tree(self, segmented_comments):
#         """Build the PAT tree from a list of segmented comments."""
#         for idx, words in enumerate(segmented_comments):
#             self.insert(words, idx)

#     def cluster(self, min_shared=5):
#         """Cluster comments based on longest shared word sequences, skipping duplicates."""
#         clusters = []
#         processed_indices = set()

#         def dfs(node, prefix):
#             # If node qualifies as a cluster
#             if len(node.data) >= min_shared:
#                 unique_data = [idx for idx in node.data if idx not in processed_indices]
#                 if unique_data:
#                     clusters.append((prefix, unique_data))
#                     processed_indices.update(unique_data)
#                 return  # Stop further traversal

#             # Continue to children
#             for word, child in node.children.items():
#                 dfs(child, prefix + [word])

#         dfs(self.root, [])
#         return clusters
    
# def cluster_traditional_chinese_comments(comments, min_shared=5, min_word_length=2): 
#     # Set Jieba dictionary for Traditional Chinese
#     # jieba.set_dictionary('dict.txt.big')  # Use appropriate Traditional Chinese dictionary file

#     # Preprocess comments: segment into words
#     segmented_comments = [list(jieba.cut(comment)) for comment in comments]

#     # Create and build the PAT tree
#     pat_tree = PATTree()
#     pat_tree.build_tree(segmented_comments)

#     # Cluster comments
#     clusters = pat_tree.cluster(min_shared=min_shared)
    
#     # Filter clusters to include only shared words with a minimum length
#     # filtered_clusters = [
#     #     (words, indices) 
#     #     for words, indices in clusters 
#     #     if len(indices) >= min_shared and all(len(word) >= min_word_length for word in words)
#     # ]
#     seen_indices = set()
#     filtered_clusters = []

#     for words, indices in clusters:
#         if (
#             len(indices) >= min_shared 
#             and all(len(word) >= min_word_length for word in words)
#             and not seen_indices.intersection(indices)
#         ):
#             filtered_clusters.append((words, indices))
#             seen_indices.update(indices)
    
#     # Map indices back to comments for better readability
#     return [
#         {
#             "shared_words": " ".join(words),
#             "comments": list(set(comments[i] for i in indices))  # Remove duplicates
#         }
#         for words, indices in filtered_clusters
#     ]

# for cluster_id, comments in cluster_comments.items():
#     # 為每個 cluster 生成自定義詞典
#     custom_dict_path = generate_custom_dict(comments, mi_threshold, min_freq)
#     jieba.initialize()
#     jieba.load_userdict(custom_dict_path)

#     # 將原始評論進行 tokenize
#     tokenized_comments = ["".join(tokenize_and_remove_stopwords(c)) for c in comments]
#     print(f"Cluster {cluster_id} comments: {len(comments)}")
#     # print(comments)

#     print(f"Cluster {cluster_id} Keywords: ")
#     result = cluster_traditional_chinese_comments(tokenized_comments, min_shared=5, min_word_length=2)
#     for cluster in result:
#         print(f"'{cluster['shared_words']}'")
#         # print(f"Cluster with shared words '{cluster['shared_words']}':")
#         # for comment in cluster["comments"]:
#         #     print(f"  - {comment}")

