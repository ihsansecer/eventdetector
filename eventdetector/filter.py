from scipy import sparse
import numpy as np

from eventdetector.spam import get_features


def is_retweet(text):
    return text.startswith("RT @")


def is_valid_token(token, stopwords, allowed_pos, min_len=2):
    return (
        len(token.word) > min_len
        and token.ner == "O"
        and token.pos in allowed_pos
        and token.word.lower() not in stopwords
    )


def prepare_tokens_base(annotated, stopwords, allowed_pos):
    tokens = {}
    for sentence in annotated.sentence:
        for token in sentence.token:
            if is_valid_token(token, stopwords, allowed_pos):
                normalized_token = token.lemma.lower()
                tokens.setdefault(normalized_token, 0)
                tokens[normalized_token] += 1
    return tokens


def prepare_tokens_embedding(annotated, stopwords, allowed_pos, w2vmodel):
    tokens = {}
    for sentence in annotated.sentence:
        for token in sentence.token:
            if is_valid_token(token, stopwords, allowed_pos):
                normalized_token = token.lemma.lower()
                if normalized_token in w2vmodel.vocab:
                    tokens.setdefault(normalized_token, 0)
                    tokens[normalized_token] += 1
    return tokens


def is_valid_entity(mention, stopwords, stopkinds, min_len=0):
    return (
        len(mention.entityMentionText) > min_len
        and mention.entityMentionText.lower() not in stopwords
        and mention.ner.lower() not in stopkinds
    )


def prepare_entities(annotated, stopwords, stopkinds):
    entities = {}
    for mention in annotated.mentions:
        if is_valid_entity(mention, stopwords, stopkinds):
            string = mention.entityMentionText
            normalized_entity = string.lower()
            entities.setdefault(normalized_entity, 0)
            entities[normalized_entity] += 1
    return entities


def is_spam(text, model, vectorizer):
    features = get_features(text)
    features = np.array(features)
    text = vectorizer.transform([text])
    data = sparse.hstack((text, features))
    prediction = model.predict(data)[0]
    return prediction == 1
