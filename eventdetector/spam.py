from sklearn.feature_extraction.text import TfidfVectorizer

import Stemmer


def get_features(text):
    """
    Extract features of percentage of capital letters,
    numbers and currency symbols in given text.

    Returns
    -------
    capital_perc, number_perc, currency_perc:
        float, float, float
    """
    capital_count = 0
    number_count = 0
    currency_count = 0
    for c in text:
        if c.isupper():
            capital_count += 1
        if c.isdigit():
            number_count += 1
        if c == "$" or c == "€" or c == "£":
            currency_count += 1
    capital_perc = capital_count / len(text)
    number_perc = number_count / len(text)
    currency_perc = currency_count / len(text)
    return capital_perc, number_perc, currency_perc


class StemmedTfidfVectorizer(TfidfVectorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.english_stemmer = Stemmer.Stemmer("en")

    def build_analyzer(self):
        # ref: https://bit.ly/34wV8vb
        analyzer = super().build_analyzer()
        return lambda doc: self.english_stemmer.stemWords(analyzer(doc))
