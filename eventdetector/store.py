class Store:
    def __init__(self):
        self.store = {}


class ClusterStoreBase(Store):
    """
    Keeps data required for each cluster.

    Store Attributes
    ----------------
    $main: dict
        Stores general data relevant to clusters.

        $doc: set
            Document ids.
        $sumlat: float
            Sum of latitudes.
        $sumlon: float
            Sum of longitudes.
        $locfreq: int
            Number of geo-documents.
    
    $wind: dict
        Stores size of cluster.
    """
    def setdefault(self, entity):
        self.store.setdefault("$main", {})
        self.store["$main"].setdefault(entity, {})
        self.store.setdefault("$wind", {})

    def add_cluster(self, entity, cluster_id):
        self.store["$main"].setdefault(entity, {})
        self.store["$main"][entity].setdefault(cluster_id, {})
        self.store["$main"][entity][cluster_id].setdefault("$doc", set())

        self.store["$main"][entity][cluster_id].setdefault("$sumlat", 0)
        self.store["$main"][entity][cluster_id].setdefault("$sumlon", 0)
        self.store["$main"][entity][cluster_id].setdefault("$locfreq", 0)

        self.store["$wind"].setdefault(entity, {})
        self.store["$wind"][entity].setdefault(cluster_id, 0)

    def add_doc(self, entity, cluster_id, doc_id, location):
        self.store["$main"][entity][cluster_id]["$doc"].add(doc_id)
        self.store["$wind"][entity][cluster_id] += 1

        if location is not None:
            lat, lon = location
            self.store["$main"][entity][cluster_id]["$sumlat"] += lat
            self.store["$main"][entity][cluster_id]["$sumlon"] += lon
            self.store["$main"][entity][cluster_id]["$locfreq"] += 1

    def remove_doc(self, entity, cluster_id, doc_id, location):
        self.store["$main"][entity][cluster_id]["$doc"].discard(doc_id)

        if len(self.store["$main"][entity][cluster_id]["$doc"]) == 0:
            del self.store["$main"][entity][cluster_id]
            return

        if location is not None:
            lat, lon = location
            self.store["$main"][entity][cluster_id]["$sumlat"] -= lat
            self.store["$main"][entity][cluster_id]["$sumlon"] -= lon
            self.store["$main"][entity][cluster_id]["$locfreq"] -= 1

    def remove_window(self):
        del self.store["$wind"]

    def get_clusters(self, entity):
        return self.store["$main"][entity]

    def get_cluster_size(self, entity, cluster_id):
        return len(self.store["$main"][entity][cluster_id]["$doc"])

    def get_docs(self, entity, cluster_id):
        return self.store["$main"][entity][cluster_id]["$doc"]

    def get_window(self, entity, cluster_id):
        return self.store["$wind"][entity][cluster_id]

    def get_centroid_location(self, entity, cluster_id):
        count = self.store["$main"][entity][cluster_id]["$locfreq"]
        lat_cent = self.store["$main"][entity][cluster_id]["$sumlat"] / count
        lon_cent = self.store["$main"][entity][cluster_id]["$sumlon"] / count
        return lat_cent, lon_cent

    def is_in_window(self, entity, cluster_id):
        return cluster_id in self.store["$wind"][entity]


class ClusterStoreEmbedding(ClusterStoreBase):
    """
    Keeps data required for each cluster.

    Store Attributes
    ----------------
    $main: dict
        Stores general data relevant to clusters.

        $doc: set
            Document ids.
        $sumlat: float
            Sum of latitudes.
        $sumlon: float
            Sum of longitudes.
        $locfreq: int
            Number of geo-documents.
        $tfsum: int
            Stores sum of term frequencies.
        $tfcount: dict
            Stores number of documents with a term.
    
    $wind: dict
        Stores size of cluster.
    """
    def add_cluster(self, entity, cluster_id):
        super(ClusterStoreEmbedding, self).add_cluster(entity, cluster_id)
        self.store["$main"][entity][cluster_id].setdefault("$tfsum", {})
        self.store["$main"][entity][cluster_id].setdefault("$tfcount", {})

    def add_term(self, entity, cluster_id, term, tf):
        self.store["$main"][entity][cluster_id]["$tfsum"].setdefault(term, 0)
        self.store["$main"][entity][cluster_id]["$tfcount"].setdefault(term, 0)
        self.store["$main"][entity][cluster_id]["$tfsum"][term] += tf
        self.store["$main"][entity][cluster_id]["$tfcount"][term] += 1

    def remove_term(self, entity, cluster_id, term, tf):
        self.store["$main"][entity][cluster_id]["$tfsum"][term] -= tf
        self.store["$main"][entity][cluster_id]["$tfcount"][term] -= 1

        if self.store["$main"][entity][cluster_id]["$tfcount"][term] == 0:
            del self.store["$main"][entity][cluster_id]["$tfsum"][term]
            del self.store["$main"][entity][cluster_id]["$tfcount"][term]

    def get_tf_sums(self, entity, cluster_id):
        return self.store["$main"][entity][cluster_id]["$tfsum"]

    def get_tf_counts(self, entity, cluster_id):
        return self.store["$main"][entity][cluster_id]["$tfcount"]


class EntityStore(Store):
    """
    Keeps data directly relevant to entities.

    Store Attributes
    ----------------
    $hist: dict
        Stores counts of documents with each entity at 
        different time steps.
    $stats: dict
        Stores statistics covering whole history kept in store.

        $sum: float
            Sum of counts of documents with entity.
        $sumsq: float
            Sum of squares of counts of documents with entity.
    """
    def __init__(self):
        super().__init__()
        self.store.setdefault("$hist", {})
        self.store.setdefault("$stats", {})

    def setdefault(self, time_step, entity):
        self.store["$hist"].setdefault(time_step, {})
        self.store["$hist"][time_step].setdefault(entity, 0)
        self.store["$stats"].setdefault(entity, {})
        self.store["$stats"][entity].setdefault("$sum", 0)
        self.store["$stats"][entity].setdefault("$sumsq", 0)

    def increment(self, time_step, entity):
        self.store["$hist"][time_step][entity] += 1

    def set_time_step(self, time_step):
        for entity in self.store["$hist"][time_step]:
            self.store["$stats"][entity]["$sum"] += self.store["$hist"][time_step][
                entity
            ]
            self.store["$stats"][entity]["$sumsq"] += (
                self.store["$hist"][time_step][entity] ** 2
            )

    def remove_time_step(self, time_step):
        for entity in self.store["$hist"][time_step]:
            self.store["$stats"][entity]["$sum"] -= self.store["$hist"][time_step][
                entity
            ]
            self.store["$stats"][entity]["$sumsq"] -= (
                self.store["$hist"][time_step][entity] ** 2
            )

        del self.store["$hist"][time_step]

    def get_hist(self):
        return self.store["$hist"]

    def get_time_step(self, time_step):
        return self.store["$hist"][time_step]

    def get_sum(self, entity):
        return self.store["$stats"][entity]["$sum"]

    def get_sumsq(self, entity):
        return self.store["$stats"][entity]["$sumsq"]


class EntityTermStoreBase(Store):
    """
    Keeps data relevant to entity-term relationship.

    Store Attributes
    ----------------
    $freq: int
        Document frequency of terms under entities.
    $index: dict
        Stores an inverted-index.
    """
    def setdefault(self, entity, term):
        self.store.setdefault(entity, {})
        self.store[entity].setdefault(term, {})
        self.store[entity][term].setdefault("$freq", 0)
        self.store[entity][term].setdefault("$index", {})

    def add_term(self, entity, term, doc_id, tf):
        self.store[entity][term]["$freq"] += 1
        self.store[entity][term]["$index"][doc_id] = tf

    def remove_term(self, entity, term, doc_id):
        self.store[entity][term]["$freq"] -= 1
        del self.store[entity][term]["$index"][doc_id]

    def get_document_frequency(self, entity, term):
        return self.store[entity][term]["$freq"]

    def get_term_frequency(self, entity, term, doc_id):
        return self.store[entity][term]["$index"][doc_id]

    def get_docs(self, entity, term):
        return self.store[entity][term]["$index"].keys()

    def get_terms(self, entity):
        return self.store[entity].keys()


class EntityTermStoreEmbedding(EntityTermStoreBase):
    """
    Keeps data relevant to entity-term relationship.

    Store Attributes
    ----------------
    $freq: int
        Document frequency of terms under entities.
    """
    def setdefault(self, entity, term):
        self.store.setdefault(entity, {})
        self.store[entity].setdefault(term, {})
        self.store[entity][term].setdefault("$freq", 0)

    def add_term(self, entity, term):
        self.store[entity][term]["$freq"] += 1

    def remove_term(self, entity, term):
        self.store[entity][term]["$freq"] -= 1

        if self.store[entity][term]["$freq"] == 0:
            del self.store[entity][term]


class EntityDocStoreBase(Store):
    """
    Keeps data relevant to entity-document relationship.

    Store Attributes
    ----------------
    $magn: float
        Magnitude of document.
    $clust: int
        Assigned cluster id of document.
    """
    def setdefault(self, entity, doc_id):
        self.store.setdefault(entity, {})
        self.store[entity].setdefault(doc_id, {})
        self.store[entity][doc_id].setdefault("$magn", 0)

    def remove_doc(self, entity, doc_id):
        del self.store[entity][doc_id]

    def magnitude_add(self, entity, doc_id, value):
        self.store[entity][doc_id]["$magn"] += value

    def set_magnitude(self, entity, doc_id, value):
        self.store[entity][doc_id]["$magn"] = value

    def get_magnitude(self, entity, doc_id):
        return self.store[entity][doc_id]["$magn"]

    def set_cluster(self, entity, doc_id, cluster_id):
        self.store[entity][doc_id]["$clust"] = cluster_id

    def get_cluster(self, entity, doc_id):
        return self.store[entity][doc_id]["$clust"]

    def get_docs(self, entity):
        return self.store[entity].keys()

    def get_entity_N(self, entity):
        return len(self.get_docs(entity))

    def has_entity(self, entity):
        return entity in self.store

    def has_doc(self, entity, doc_id):
        return self.has_entity(entity) and doc_id in self.store[entity]


class EntityDocStoreEmbedding(EntityDocStoreBase):
    """
    Keeps data relevant to entity-document relationship.

    Store Attributes
    ----------------
    $clust: int
        Assigned cluster id of document.
    """
    def setdefault(self, entity, doc_id):
        self.store.setdefault(entity, {})
        self.store[entity].setdefault(doc_id, {})


class DocStore(Store):
    """
    Keeps data directly relevant to documents.

    Store Attributes
    ----------------
    $term: dict
        Terms of document with its frequencies.
    $term: entity
        Entities of document with its frequencies.
    """
    def setdefault(self, doc_id):
        self.store.setdefault(doc_id, {})
        self.store[doc_id].setdefault("$term", {})
        self.store[doc_id].setdefault("$entity", {})

    def add_term(self, doc_id, term, tf):
        self.store[doc_id]["$term"][term] = tf

    def add_entity(self, doc_id, entity, tf):
        self.store[doc_id]["$entity"][entity] = tf

    def get_term_tf(self, doc_id, term):
        return self.store[doc_id]["$term"][term]

    def get_entity_tf(self, doc_id, entity):
        return self.store[doc_id]["$entity"][entity]

    def remove_doc(self, doc_id):
        del self.store[doc_id]

    def get_terms(self, doc_id):
        return self.store[doc_id]["$term"]

    def get_entities(self, doc_id):
        return self.store[doc_id]["$entity"]

    def get_docs(self):
        return self.store.keys()


class EntityMatchStore(Store):
    """
    Keeps data of matched entities.

    Store Attributes
    ----------------
    $wid: str
        Wikidata id of entity.
    $match: set
        Matched entities for entity.
    """
    def setdefault(self, entity, wid):
        self.store.setdefault(entity, {})
        self.store[entity]["$wid"] = wid
        self.store[entity]["$match"] = set()

    def add_match(self, entity, match):
        self.store[entity]["$match"].add(match)

    def get_wid(self, entity):
        return self.store[entity]["$wid"]

    def get_matches(self, entity):
        return self.store[entity]["$match"]

    def get_entities(self):
        return self.store.keys()

    def is_in(self, entity):
        return entity in self.store
