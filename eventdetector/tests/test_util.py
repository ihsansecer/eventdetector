from eventdetector.util import get_wid, calculate_tfidf, get_top_k


def test_get_wid():
    wid = get_wid("boris johnson")
    assert wid == "Q180589"


def test_tf_idf():
    tf_idf = calculate_tfidf(1, 10, 100)
    assert tf_idf == 2


def test_get_top_k():
    values = [3, 5, 2, 10, 4]
    elements = ["a", "b", "c", "d", "e"]
    top_values, top_elements = get_top_k(values, elements, 3)

    expected_values = [4, 5, 10]
    expected_elements = ["e", "b", "d"]
    assert top_values == expected_values
    assert top_elements == expected_elements
