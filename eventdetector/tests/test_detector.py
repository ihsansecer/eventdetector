import datetime

import pytest

from eventdetector import EventDetectorBase


@pytest.mark.parametrize(
    "docs,min_threshold",
    [
        (
            [
                {
                    "id": 1,
                    "date": datetime.datetime(2020, 1, 1, 10, 0),
                    "terms": {"baseball": 1, "team": 1, "world": 1},
                    "entities": {"spain_country": 1},
                },
                {
                    "id": 2,
                    "date": datetime.datetime(2020, 1, 1, 10, 1),
                    "terms": {"baseball": 1, "team": 1, "going": 1},
                    "entities": {"spain_country": 1},
                },
            ],
            0.5,
        ),
        (
            [
                {
                    "id": 1,
                    "date": datetime.datetime(2020, 1, 1, 10, 0),
                    "terms": {"baseball": 1, "team": 1, "world": 1},
                    "entities": {"spain_country": 1},
                },
                {
                    "id": 2,
                    "date": datetime.datetime(2020, 1, 1, 10, 1),
                    "terms": {"baseball": 1, "team": 1, "world": 1},
                    "entities": {"spain_country": 1},
                },
            ],
            0.9,
        ),
    ],
)
def test_with_cluster(docs, min_threshold):
    ed = EventDetectorBase(min_threshold=min_threshold)
    for doc in docs:
        doc["terms"].update(doc["entities"])
        ed.fit(doc["id"], doc["date"], doc["terms"], doc["entities"])
    ed.flush()

    expected_cluster_id = 0
    for doc in docs:
        entity = next(iter(doc["entities"]))
        assert (
            ed.entity_doc_store_.get_cluster(entity, doc["id"]) == expected_cluster_id
        )


@pytest.mark.parametrize(
    "docs",
    [
        [
            {
                "id": 1,
                "date": datetime.datetime(2020, 1, 1, 10, 0),
                "terms": {"baseball": 1, "team": 1, "world": 1},
                "entities": {"spain_country": 1},
            },
            {
                "id": 2,
                "date": datetime.datetime(2020, 1, 1, 10, 1),
                "terms": {"account": 1, "checking": 1, "get": 1, "hand": 1},
                "entities": {"biden_person": 1, "president_title": 1},
            },
        ]
    ],
)
def test_with_no_cluster(docs):
    ed = EventDetectorBase()
    for doc in docs:
        doc["terms"].update(doc["entities"])
        ed.fit(doc["id"], doc["date"], doc["terms"], doc["entities"])
    ed.flush()

    for doc in docs:
        for entity in doc["entities"]:
            assert len(ed.entity_doc_store_.get_docs(entity)) == 1


@pytest.mark.parametrize(
    "docs",
    [
        [
            {
                "id": 1,
                "date": datetime.datetime(2020, 1, 1, 10, 00),
                "terms": {"baseball": 1, "team": 1, "world": 1},
                "entities": {"spain_country": 1},
            },
            {
                "id": 2,
                "date": datetime.datetime(2020, 1, 1, 11, 00),
                "terms": {"account": 1, "checking": 1, "get": 1, "hand": 1},
                "entities": {"biden_person": 1, "president_title": 1},
            },
        ]
    ],
)
def test_remove_expired(docs):
    ed = EventDetectorBase()
    for doc in docs:
        doc["terms"].update(doc["entities"])
        ed.fit(doc["id"], doc["date"], doc["terms"], doc["entities"])
    ed.flush()

    doc = docs[0]
    for entity in doc["entities"]:
        assert doc["id"] not in ed.entity_doc_store_.get_docs(entity)

        for term in ed.entity_term_store_.get_terms(entity):
            pass
            assert doc["id"] not in ed.entity_term_store_.get_docs(entity, term)

    assert doc["id"] not in ed.doc_store_.get_docs()
    assert len(ed.window_) == 1
