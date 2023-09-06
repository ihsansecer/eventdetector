import math

import numpy as np
import requests
import yaml


def get_wid(title):
    """
    Fetches Wikidata id for title using MediaWiki API.

    Parameters
    ----------
    title: str
        Name to be queried.

    Returns
    -------
    wid: str or None
        Wikidata id if found, otherwise None.
    """
    base_url = "https://en.wikipedia.org/w/api.php?"
    base_q = "action=query&prop=pageprops&redirects=1&format=json"

    q = base_url + base_q + "&titles=" + title
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    r = requests.post(q, headers=headers)

    try:
        json = r.json()
        pages = json["query"]["pages"]
        page = next(iter(pages))

        if page == "-1":
            return None
        if "disambiguation" in pages[page]["pageprops"]:
            return None

        wid = pages[page]["pageprops"]["wikibase_item"]
    except KeyError:
        return None
    return wid


def calculate_tfidf(tf, df, N):
    """
    Calculates tf-idf weight.

    Parameters
    ----------
    tf: int
        Term frequency
    df : int
        Document frequency
    N : int
        Total number of documents

    Returns
    -------
    tfidf: float
        Calculated IDF
    """
    tfidf = tf * (1 + math.log10(N / df))
    return tfidf


def get_top_k(values, elements, k):
    """
    Sorts `elements` using `values` and
    gets top `k` among them.

    Parameters
    ----------
    values: list of float or int
    elements: list of any

    Returns
    -------
    k_values, k_elements: 
        list of float or int, list of any
    """
    if len(values) <= k:
        return values, elements

    indices = np.argsort(values)[-k:]
    k_values = []
    k_elements = []
    for i in indices:
        k_values.append(values[i])
        k_elements.append(elements[i])
    return k_values, k_elements


def get_config():
    """
    Gets config dictionary from config.yaml file.

    Returns
    -------
    config: dictionary
        Config dictionary.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
