import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from scipy import sparse
import numpy as np

from eventdetector.spam import get_features, StemmedTfidfVectorizer
from eventdetector.constant import stopwords
from eventdetector.util import get_config


def evaluate_lightgbm(datas):
    scores = {"acc": [], "prec": [], "rec": []}
    for tra_data, val_data, tra_labels, val_labels in datas:

        model = LGBMClassifier(
            random_state=0,
            learning_rate=0.2,
            n_estimators=5000,
            num_leaves=150,
            max_depth=200,
            min_child_samples=10,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            n_jobs=-1,
        )
        model.fit(
            tra_data,
            tra_labels,
            eval_set=[(val_data, val_labels)],
            early_stopping_rounds=50,
            verbose=50,
        )

        start_time = time.time()
        predicted = model.predict(val_data)
        print(time.time() - start_time, val_data.shape[0])

        acc = accuracy_score(val_labels, predicted)
        scores["acc"].append(acc)
        prec = precision_score(val_labels, predicted)
        scores["prec"].append(prec)
        rec = recall_score(val_labels, predicted)
        scores["rec"].append(rec)
        print("Acc:", round(acc, 3), "Prec:", round(prec, 3), "Rec:", round(rec, 3))

    print_scores(scores)


def evaluate_randomforest(datas):
    scores = {"acc": [], "prec": [], "rec": []}
    for tra_data, val_data, tra_labels, val_labels in datas:

        model = RandomForestClassifier(
            random_state=0,
            n_jobs=-1,
            n_estimators=200,
            max_depth=200,
            min_samples_leaf=1,
        )
        model.fit(tra_data, tra_labels)

        start_time = time.time()
        predicted = model.predict(val_data)
        print(time.time() - start_time, val_data.shape[0])

        acc = accuracy_score(val_labels, predicted)
        scores["acc"].append(acc)
        prec = precision_score(val_labels, predicted)
        scores["prec"].append(prec)
        rec = recall_score(val_labels, predicted)
        scores["rec"].append(rec)
        print("Acc:", round(acc, 3), "Prec:", round(prec, 3), "Rec:", round(rec, 3))

    print_scores(scores)


def evaluate_logisticregression(datas):
    scores = {"acc": [], "prec": [], "rec": []}
    for tra_data, val_data, tra_labels, val_labels in datas:

        model = LogisticRegression(
            solver="liblinear", random_state=0, max_iter=1000, C=1, penalty="l2"
        )
        model.fit(tra_data, tra_labels)

        start_time = time.time()
        predicted = model.predict(val_data)
        print(time.time() - start_time, val_data.shape[0])

        acc = accuracy_score(val_labels, predicted)
        scores["acc"].append(acc)
        prec = precision_score(val_labels, predicted)
        scores["prec"].append(prec)
        rec = recall_score(val_labels, predicted)
        scores["rec"].append(rec)
        print("Acc:", round(acc, 3), "Prec:", round(prec, 3), "Rec:", round(rec, 3))

    print_scores(scores)


def evaluate_naivebayes(datas):
    scores = {"acc": [], "prec": [], "rec": []}
    for i, (tra_data, val_data, tra_labels, val_labels) in enumerate(datas):

        model = MultinomialNB(alpha=1e-10)
        model.fit(tra_data, tra_labels)

        start_time = time.time()
        predicted = model.predict(val_data)
        print(time.time() - start_time, val_data.shape[0])

        acc = accuracy_score(val_labels, predicted)
        scores["acc"].append(acc)
        prec = precision_score(val_labels, predicted)
        scores["prec"].append(prec)
        rec = recall_score(val_labels, predicted)
        scores["rec"].append(rec)
        print("Acc:", round(acc, 3), "Prec:", round(prec, 3), "Rec:", round(rec, 3))

    print_scores(scores)


def get_featurized_folds(features, texts, labels):
    vectorizer = StemmedTfidfVectorizer(
        sublinear_tf=True, max_features=150000, stop_words=stopwords
    )

    skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    datas = []
    for train_index, val_index in skf.split(features, labels):

        tra_features, val_features = features[train_index], features[val_index]
        tra_texts, val_texts = texts[train_index], texts[val_index]
        tra_labels, val_labels = labels[train_index], labels[val_index]

        tra_texts = vectorizer.fit_transform(tra_texts)
        tra_data = sparse.hstack((tra_texts, tra_features))

        val_texts = vectorizer.transform(val_texts)
        val_data = sparse.hstack((val_texts, val_features))
        datas.append((tra_data, val_data, tra_labels, val_labels))

    return datas


def get(path):
    texts = []
    features = []
    with open(path, "r") as f:
        for line in f:
            text = line.split("\t")[2]
            texts.append(text)
            feature = get_features(text)
            features.append(feature)
    return texts, features


def print_scores(scores):
    print("Average")
    for k in scores:
        print(k + ":", round(np.mean(scores[k]), 3))


def main(polluter_path, legit_path):
    print("*" * 20 + "Processing Data" + "*" * 20)

    p_texts, p_features = get(polluter_path)
    n_texts, n_features = get(legit_path)
    texts = p_texts + n_texts
    features = p_features + n_features

    n_p, n_n = len(p_texts), len(n_texts)
    print("Polluter:", n_p, "Legit:", n_n)

    labels = np.array([1] * n_p + [0] * n_n)
    features = np.array(features)
    texts = np.array(texts)

    print("*" * 20 + "Preparing Features" + "*" * 20)
    datas = get_featurized_folds(features, texts, labels)
    print("*" * 20 + "Evaluating Models" + "*" * 20)
    print("-" * 20 + "Gradient Boosting" + "-" * 20)
    evaluate_lightgbm(datas)
    print("-" * 20 + "Random Forest" + "-" * 20)
    evaluate_randomforest(datas)
    print("-" * 20 + "Logit Regression" + "-" * 20)
    evaluate_logisticregression(datas)
    print("-" * 20 + "Naive Bayes" + "-" * 20)
    evaluate_naivebayes(datas)


if __name__ == "__main__":
    config = get_config()

    polluter_path = config["spam"]["spam"]
    legit_path = config["spam"]["nonspam"]

    main(polluter_path, legit_path)
