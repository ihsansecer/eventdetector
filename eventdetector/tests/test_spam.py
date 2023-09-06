from eventdetector.spam import get_features


def test_get_features():
    text = "I am selling my car for 1000$."
    capital_perc, number_perc, currency_perc = get_features(text)

    assert capital_perc == 1 / len(text)
    assert number_perc == 4 / len(text)
    assert currency_perc == 1 / len(text)
