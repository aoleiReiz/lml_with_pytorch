from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from preprocess import clean_tweet


train = pd.read_csv("train_2kmZucJ.csv")

freq_dict = defaultdict(lambda: 0)

for _, row in train.iterrows():
    cleaned_tweet = clean_tweet(row["tweet"])
    for w in cleaned_tweet:
        freq_dict[(w,row["label"])] += 1

X = []
y = []
for _, row in train.iterrows():
    cleaned_tweet = clean_tweet(row["tweet"])
    pos_count = 0
    neg_count = 0
    for w in cleaned_tweet:
        pos_count += freq_dict[(w, 1)]
        neg_count += freq_dict[(w, 0)]
    X.append([pos_count, neg_count])
    y.append(row["label"])


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lrg = LogisticRegression(class_weight="balanced")
lrg.fit(X_train, y_train)
print(lrg.score(X_test, y_test))

# fit all
lrg.fit(X, y)

test = pd.read_csv("test_oJQbWVk.csv")
rows = []
for _, row in test.iterrows():
    cleaned_tweet = clean_tweet(row["tweet"])
    pos_count = 0
    neg_count = 0
    for w in cleaned_tweet:
        pos_count += freq_dict[(w, 1)]
        neg_count += freq_dict[(w, 0)]
    rows.append([pos_count, neg_count])

X_input = scaler.transform(rows)
predictions = lrg.predict(X_input)
df = pd.DataFrame()
df["id"] = test.id
df["label"] = predictions
df.to_csv("submission.csv", index_label=False, index=False)