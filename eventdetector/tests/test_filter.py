from collections import namedtuple

from eventdetector.filter import is_retweet, prepare_tokens_base, prepare_entities
from eventdetector.constant import stopwords, stopkinds, allowed_pos


Annotated = namedtuple("Annotated", ["sentence", "mentions"])
Sentence = namedtuple("Sentence", ["token"])
Tokens = namedtuple("Tokens", ["token"])
Token = namedtuple("Token", ["lemma", "word", "ner", "pos"])
Mention = namedtuple("Mentions", ["entityMentionText", "ner"])


def test_is_retweet():
    text = "RT @user: Lorem ipsum dolor sit amet"
    assert is_retweet(text)


def test_prepare_tokens_base():
    tokens = [
        Token(lemma="I", word="I", ner="?", pos="NN"),
        Token(lemma="am", word="am", ner="O", pos="?"),
        Token(lemma="go", word="going", ner="O", pos="VB"),
        Token(lemma="study", word="study", ner="O", pos="VB"),
        Token(lemma="study", word="study", ner="O", pos="VB"),
    ]
    sentences = [Sentence(token=tokens)]
    annotated = Annotated(sentence=sentences, mentions=None)

    result = prepare_tokens_base(annotated, stopwords, allowed_pos)
    expected = {"go": 1, "study": 2}
    assert result == expected


def test_prepare_entities():
    mentions = [
        Mention(entityMentionText="He", ner="person"),
        Mention(entityMentionText="Boris", ner="person"),
        Mention(entityMentionText="2020", ner="date"),
        Mention(entityMentionText="Boris", ner="person"),
        Mention(entityMentionText="UK", ner="country"),
    ]
    annotated = Annotated(sentence=None, mentions=mentions)

    result = prepare_entities(annotated, stopwords, stopkinds)
    expected = {"uk": 1, "boris": 2}
    assert result == expected
