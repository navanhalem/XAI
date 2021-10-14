import os

import pandas as pd
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Get the IMDB movie review dataset
dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",
                                  origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                  untar=True,
                                  cache_dir='.',
                                  cache_subdir='')

# Organise the data
df = pd.DataFrame([], columns=['question_text', 'target'])
for file in os.listdir(os.path.join('aclImdb', 'test', 'neg')):
    with open(os.path.join('aclImdb', 'test', 'neg', file), encoding="utf-8") as f:
        df = df.append(pd.DataFrame([[f.readlines()[0], 0]], columns=['question_text', 'target']))

for file in os.listdir(os.path.join('aclImdb', 'test', 'pos')):
    with open(os.path.join('aclImdb', 'test', 'pos', file), encoding="utf-8") as f:
        df = df.append(pd.DataFrame([[f.readlines()[0], 1]], columns=['question_text', 'target']))

print("Dataframe shape : ", df.shape)

# Split to train and val
train_df, val_df = train_test_split(df, test_size=0.1, random_state=1)
val_df.reset_index(drop=True)

# Vectorize to tf-idf vectors
tfidf_vc = TfidfVectorizer(min_df=10, max_features=100000, analyzer="word", ngram_range=(1, 2), stop_words='english',
                           lowercase=True)
train_vc = tfidf_vc.fit_transform(train_df["question_text"])
val_vc = tfidf_vc.transform(val_df["question_text"])

# Train model and print F1 score on validation set
text_model = LogisticRegression(C=0.5, solver="sag")
text_model = text_model.fit(train_vc, train_df.target.astype('int'))
val_pred = text_model.predict(val_vc)

val_cv = f1_score(val_df.target.astype('int'), val_pred, average="binary")
print(val_cv)

# Select review (or use custom one)
idx = 1000
class_names = ["negative", "positive"]
# review_text = val_df["question_text"].iloc[idx]
review_text = "This movie was not bad at all. It was very enjoyable and featured some interesting characters."
# true_class = class_names[val_df["target"].iloc[idx]]
true_class = "positive"

# Make pipeline and explainer model
c = make_pipeline(tfidf_vc, text_model)
explainer = LimeTextExplainer(class_names=class_names)
exp_instance = explainer.explain_instance(review_text, c.predict_proba, num_features=5)

# Show review text, predictions, and label
print("Review: \n", review_text)
print("Probability (Negative) =", c.predict_proba([review_text])[0, 0])
print("Probability (Positive) =", c.predict_proba([review_text])[0, 1])
print("True Class is:", true_class)

exp_instance.as_pyplot_figure()
# exp_instance.show_in_notebook(text=review_text, labels=(1,))
