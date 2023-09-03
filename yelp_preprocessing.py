import pandas as pd
import scipy as sp
import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Adapted from https://github.com/dehaoterryzhang/Yelp_Sentiment_Analysis/blob/master/code/sentiment-analysis-with-lr.ipynb

# Read in the data
reviews = []
lines_read = 0
with open('./data/yelp_academic_dataset_review.json') as f:
    for line in f:
        reviews.append(json.loads(line))
        lines_read += 1
        if lines_read % 200000 == 0:
            print('Read {} lines'.format(lines_read))
df_reviews = pd.DataFrame(reviews)

df_reviews = df_reviews[df_reviews.stars != 3] # remove neutral reviews

# Convert to binary classification problem
pd.set_option('mode.chained_assignment', None)
df_reviews["labels"] = df_reviews["stars"].apply(lambda x: -1 if x < 3  else 1) # positive as 1 and negative as -1
df_reviews = df_reviews.drop("stars",axis=1)

# Split into train and test
train, test = train_test_split(df_reviews, test_size = 0.2, stratify = df_reviews['labels'], random_state = 42)

# Preprocess the text into unigrams and bigrams
start_time = time.time()
cv= CountVectorizer(binary=True, min_df = 10, max_df = 0.95, ngram_range=(1, 2))
cv.fit_transform(train['text'].values)
train_feature_set=cv.transform(train['text'].values)
test_feature_set=cv.transform(test['text'].values)
print("Time taken to convert text input into feature vectors: ", round((time.time() - start_time)/60, 2), " mins")

y_train = train['labels'].values
y_test = test['labels'].values

# Save the data
sp.sparse.save_npz('./data/yelp_train.npz', train_feature_set)
sp.sparse.save_npz('./data/yelp_test.npz', test_feature_set)
np.save('./data/yelp_train_labels.npy', y_train)
np.save('./data/yelp_test_labels.npy', y_test)