# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:55:50 2024

@author: user
"""

# part 2
import jieba
import numpy as np
from collections import Counter
from itertools import combinations
import pandas as pd

# Tokenize the text and split into sentences
sentences = [
    jieba.lcut("天使紅蝦非常好吃，肉質鮮甜，讓人回味無窮。"),
    jieba.lcut("龍蝦湯稍微鹹了一些，但味道還不錯。"),
    jieba.lcut("服務生非常親切，服務很好，讓人感到滿意。"),
    jieba.lcut("環境非常舒適，是個適合聚會的好地方。"),
    jieba.lcut("這裡的天使紅蝦真是一絕，強烈推薦！")
]

# Flatten and count word frequencies
all_words = [word for sentence in sentences for word in sentence]
word_counts = Counter(all_words)

# Build co-occurrence matrix
unique_words = list(word_counts.keys())
word_index = {word: i for i, word in enumerate(unique_words)}

co_occurrence_matrix = np.zeros((len(unique_words), len(unique_words)))

for sentence in sentences:
    for word1, word2 in combinations(sentence, 2):
        i, j = word_index[word1], word_index[word2]
        co_occurrence_matrix[i][j] += 1
        co_occurrence_matrix[j][i] += 1

# Convert to DataFrame for readability
co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=unique_words, columns=unique_words)

print("Word Frequencies:")
print(word_counts.most_common(10))  # Top 10 frequent words
print("\nCo-occurrence Matrix:")
print(co_occurrence_df)

for i in word_counts:
    jieba.add_word(i)
# part 2