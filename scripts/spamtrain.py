from sklearn.linear_model import LogisticRegression
from joblib import dump
from scipy import sparse
import numpy as np

from eventdetector.spam import get_features, StemmedTfidfVectorizer
from eventdetector.constant import stopwords
from eventdetector.util import get_config


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


def main(polluter_path, legit_path, out_vectorizer, out_model):
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

    vectorizer = StemmedTfidfVectorizer(
        sublinear_tf=True, max_features=150000, stop_words=stopwords
    )

    model = LogisticRegression(
        solver="liblinear", random_state=0, max_iter=1000, C=1, penalty="l2"
    )

    print("*" * 20 + "Training Models" + "*" * 20)

    texts = vectorizer.fit_transform(texts)
    data = sparse.hstack((texts, features))
    model.fit(data, labels)

    print("*" * 20 + "Saving Models" + "*" * 20)

    vectorizer_dict = {
        "fixed_vocabulary_": vectorizer.fixed_vocabulary_,
        "stop_words_": vectorizer.stop_words_,
        "vocabulary_": vectorizer.vocabulary_,
        "_tfidf": vectorizer._tfidf,
    }
    dump(vectorizer_dict, out_vectorizer)
    dump(model, out_model)


if __name__ == "__main__":
    config = get_config()

    polluter_path = config["spam"]["spam"]
    legit_path = config["spam"]["nonspam"]
    out_vectorizer = config["spam"]["vectorizer"]
    out_model = config["spam"]["model"]

    main(polluter_path, legit_path, out_vectorizer, out_model)
