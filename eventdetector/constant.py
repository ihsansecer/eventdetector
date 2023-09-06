from nltk import corpus


stopwords = set(corpus.stopwords.words("english"))
stopkinds = {
    "set",
    "time",
    "date",
    "duration",
    "url",
    "email",
    "money",
    "number",
    "ordinal",
    "percent",
}
allowed_pos = {
    "FW",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "NN",
    "NNS",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
}
