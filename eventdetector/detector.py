from collections import deque
from math import sqrt
import json

from scipy.spatial.distance import euclidean
import numpy as np

from eventdetector.util import get_wid, calculate_tfidf, get_top_k
from eventdetector.store import (
    EntityStore,
    EntityTermStoreBase,
    EntityTermStoreEmbedding,
    EntityDocStoreBase,
    EntityDocStoreEmbedding,
    DocStore,
    ClusterStoreBase,
    ClusterStoreEmbedding,
    EntityMatchStore,
)


class EventDetectorBase:
    """
    Implementation of inverted index-based event detection algorithm.
    
    Parameters
    ----------
    min_threshold: int or float, default=0.45
        Minimum similarity threshold required to form a cluster
        with another document.
    window_hours: int or float, default=1
        Length of event detection window in hours.
    slide_mins: int or float, default=1
        Slide frequency of window in minutes.
    burst_span: int, default=5
        Number of times window must be slid before burst detection.
    burst_threshold: int or float, default=1
        Minimum burtingness score required for a cluster to burst.
    cluster_limit: int, default=5
        Minimum document threshold required for a cluster to burst.
    link_percentage: float, default=0.5
        Minimum percentage of document required to be shared by two
        clusters with different entities to link them.
    burst_history: int, default=10
        Number of historical timeframes to use while calculating
        burtingness score.
    top_clusters: None or int, default=None
        Limits number of events to be returned. If None is given,
        all clusters are returned instead of top clusters.
    """
    def __init__(
        self,
        min_threshold=0.45,
        window_hours=1,
        slide_mins=1,
        burst_span=5,
        burst_threshold=1,
        cluster_limit=5,
        link_percentage=0.5,
        burst_history=10,
        top_clusters=None,
    ):
        self.min_threshold = min_threshold
        self.window_len = window_hours * 60 ** 2
        self.slide_freq = slide_mins * 60
        self.burst_span = burst_span
        self.burst_threshold = burst_threshold
        self.cluster_limit = cluster_limit
        self.link_percentage = link_percentage
        self.burst_history = burst_history
        self.top_clusters = top_clusters

        self.cluster_store_ = ClusterStoreBase()
        self.entity_doc_store_ = EntityDocStoreBase()
        self.entity_term_store_ = EntityTermStoreBase()
        self.entity_store_ = EntityStore()
        self.doc_store_ = DocStore()
        self.window_ = deque()
        self.cluster_id_ = 0
        self.update_buffer_ = deque()
        self.update_time_ = None
        self.burst_time_ = None
        self.span_ = 1
        self.time_step_ = 1

    def _magnitude_add(self, entity, term, doc_id):
        """
        Updates magnitude of document by adding term's squared idf.

        Parameters
        ----------
        entity: str
        term: str
        doc_id: int
        """
        df = self.entity_term_store_.get_document_frequency(entity, term)
        tf = self.entity_term_store_.get_term_frequency(entity, term, doc_id)
        entity_N = self.entity_doc_store_.get_entity_N(entity)
        tfidf = calculate_tfidf(tf, df, entity_N)
        self.entity_doc_store_.magnitude_add(entity, doc_id, tfidf ** 2)

    def _update_magnitude(self, entity, doc_id):
        """
        Recalculates magnitude of `doc_id` from scratch.

        Parameters
        ----------
        entity: str
        doc_id: int
        """
        entity_N = self.entity_doc_store_.get_entity_N(entity)
        terms = self.doc_store_.get_terms(doc_id)
        magnitude = 0
        for term in terms:
            df = self.entity_term_store_.get_document_frequency(entity, term)
            tf = self.entity_term_store_.get_term_frequency(entity, term, doc_id)
            magnitude += calculate_tfidf(tf, df, entity_N) ** 2

        entities = self.doc_store_.get_entities(doc_id)
        for other_entity in entities:
            if entity == other_entity:
                continue
            df = self.entity_term_store_.get_document_frequency(entity, other_entity)
            tf = self.entity_term_store_.get_term_frequency(
                entity, other_entity, doc_id
            )
            magnitude += calculate_tfidf(tf, df, entity_N) ** 2

        self.entity_doc_store_.set_magnitude(entity, doc_id, magnitude)

    def _calculate_term_similarity(self, entity, term, doc_id, other_id):
        """
        Calculate cosine similarity for one matching term. The actual cosine similarity
        between documents with `doc_id` and `other_id` is sum of all matching terms.

        Parameters
        ----------
        entity: str
        term: str
        doc_id: int
        other_id: int

        Returns
        -------
        similarity_t: float
            Calculated similarity for `term` under `entity`.
        """
        df = self.entity_term_store_.get_document_frequency(entity, term)
        tf = self.entity_term_store_.get_term_frequency(entity, term, doc_id)
        tf_other = self.entity_term_store_.get_term_frequency(entity, term, other_id)

        entity_N = self.entity_doc_store_.get_entity_N(entity)
        tfidf = calculate_tfidf(tf, df, entity_N)
        tfidf_other = calculate_tfidf(tf_other, df, entity_N)

        doc_magnitude = self.entity_doc_store_.get_magnitude(entity, doc_id)
        other_doc_magnitude = self.entity_doc_store_.get_magnitude(entity, other_id)

        similarity_t = (tfidf * tfidf_other) / (
            doc_magnitude * other_doc_magnitude
        ) ** 0.5
        return similarity_t

    def _get_most_similar_document(self, terms, entity, doc_id):
        """
        Gets the most similar document for `doc_id`. Uses cosine similarity.
        Entity whose inverted index is queried has no weight for
        similarity calculations.

        Parameters
        ----------
        terms: dict
        entity: str
        doc_id: int

        Returns
        -------
        max_similarity, similar_id: float, int
            Maximum calculated similarity and most similar document's id.
        """
        similarity = {}
        max_similarity = 0
        similar_id = None
        for term in terms:
            if term == entity:
                continue

            other_docs = self.entity_term_store_.get_docs(entity, term)
            for other_id in other_docs:
                if doc_id == other_id:
                    continue

                similarity_t = self._calculate_term_similarity(
                    entity, term, doc_id, other_id
                )
                similarity.setdefault(other_id, 0)
                similarity[other_id] += similarity_t

                if max_similarity < similarity[other_id]:
                    max_similarity = similarity[other_id]
                    similar_id = other_id

        return max_similarity, similar_id

    def _assign_cluster(self, entity, location, doc_id, similar_id, similarity):
        """
        Assign `doc_id` to a cluster given the most similar document. If similarity is
        lower than a threshold a new cluster is created.

        Parameters
        ----------
        entity: str
        location: tuple of float
        doc_id: int
        similar_id: int
        similarity: float
        """
        if similarity >= self.min_threshold:
            similar_cluster_id = self.entity_doc_store_.get_cluster(entity, similar_id)
            self.entity_doc_store_.set_cluster(entity, doc_id, similar_cluster_id)
            self.cluster_store_.add_cluster(entity, similar_cluster_id)
            self.cluster_store_.add_doc(entity, similar_cluster_id, doc_id, location)

        else:
            self.entity_doc_store_.set_cluster(entity, doc_id, self.cluster_id_)
            self.cluster_store_.add_cluster(entity, self.cluster_id_)
            self.cluster_store_.add_doc(entity, self.cluster_id_, doc_id, location)
            self.cluster_id_ += 1

    def _add_new_documents(self):
        """
        Process new documents for clustering. Updates inverted index for new
        document, finds most similar document and assigns it to a cluster.
        """
        while self.update_buffer_:
            doc_id, date, terms, entities, location = self.update_buffer_.pop()
            self.doc_store_.setdefault(doc_id)
            self.window_.append((doc_id, date, location))

            for entity in entities:
                self.entity_store_.setdefault(self.time_step_, entity)
                self.entity_store_.increment(self.time_step_, entity)

                self.entity_doc_store_.setdefault(entity, doc_id)
                tf = entities[entity]
                self.doc_store_.add_entity(doc_id, entity, tf)

                self.cluster_store_.setdefault(entity)

                docs_to_update = set()
                for term in terms:
                    if term == entity:
                        continue

                    self.entity_term_store_.setdefault(entity, term)
                    tf = terms[term]
                    self.entity_term_store_.add_term(entity, term, doc_id, tf)

                    if term not in entities:
                        self.doc_store_.add_term(doc_id, term, tf)

                    self._magnitude_add(entity, term, doc_id)

                    other_docs = self.entity_term_store_.get_docs(entity, term)
                    for other_id in other_docs:
                        if doc_id == other_id:
                            continue

                        docs_to_update.add(other_id)

                for other_id in docs_to_update:
                    self._update_magnitude(entity, other_id)

                max_similarity, similar_id = self._get_most_similar_document(
                    terms, entity, doc_id
                )
                self._assign_cluster(
                    entity, location, doc_id, similar_id, max_similarity
                )

    def _remove_expired_documents(self, date):
        """
        Removes expired documents from window along with the data
        related to those document.

        Parameters
        ----------
        date: datetime
            Time of current document.
        """
        while self.window_:
            doc_id, doc_date, doc_location = self.window_.popleft()
            doc_time_diff = (date - doc_date).total_seconds()
            if doc_time_diff >= self.window_len:
                entities = self.doc_store_.get_entities(doc_id)

                for entity in entities:
                    cluster_id = self.entity_doc_store_.get_cluster(entity, doc_id)
                    self.cluster_store_.remove_doc(
                        entity, cluster_id, doc_id, doc_location
                    )
                    self.entity_doc_store_.remove_doc(entity, doc_id)

                    for term in self.doc_store_.get_terms(doc_id):
                        self.entity_term_store_.remove_term(entity, term, doc_id)

                    for other_entity in entities:
                        if entity == other_entity:
                            continue

                        self.entity_term_store_.remove_term(
                            entity, other_entity, doc_id
                        )

                self.doc_store_.remove_doc(doc_id)

            else:
                self.window_.appendleft((doc_id, doc_date, doc_location))
                break

    def _rank_clusters(self):
        """
        Rank active clusters based on how bursty they are.

        Returns
        -------
        events: list of dict
            Detected events with entity, cluster id and burstingness score.
        """
        hist = self.entity_store_.get_hist()
        N_hist = len(hist)

        cluster_weight = {}
        cluster_count = 0
        current_window = self.entity_store_.get_time_step(self.time_step_)
        for entity in current_window:
            entity_current = current_window[entity]
            entity_sum = self.entity_store_.get_sum(entity) + entity_current
            entity_sumsq = self.entity_store_.get_sumsq(entity) + entity_current ** 2
            if entity_sum == 0 or entity_sumsq == 0:
                continue

            entity_mu = entity_sum / N_hist
            entity_sigma = sqrt((entity_sumsq - entity_sum ** 2 / N_hist) / (N_hist))
            if entity_sigma == 0:
                continue

            entity_z = (entity_current - entity_mu) / entity_sigma

            for cluster_id in self.cluster_store_.get_clusters(entity):
                if not self.cluster_store_.is_in_window(entity, cluster_id):
                    continue

                N_cluster = len(self.cluster_store_.get_docs(entity, cluster_id))
                if N_cluster < self.cluster_limit:
                    continue

                cluster_current = self.cluster_store_.get_window(entity, cluster_id)
                cluster_volume = cluster_current / entity_current

                B = entity_z * cluster_volume
                if B < self.burst_threshold:
                    continue
                cluster_count += 1

                cluster_weight[cluster_id] = {
                    "cluster_entity": {cluster_id: entity},
                    "B": B,
                }

        cluster_weight = self._link_clusters(cluster_weight)

        weights = []
        clusters = []
        for cluster_id in cluster_weight:
            weights.append(cluster_weight[cluster_id]["B"])
            clusters.append(cluster_weight[cluster_id]["cluster_entity"])

        if self.top_clusters is None:
            top_weights, top_clusters = get_top_k(weights, clusters, cluster_count)
        else:
            top_weights, top_clusters = get_top_k(weights, clusters, self.top_clusters)

        events = zip(top_weights, top_clusters)
        return events

    def _link_clusters(self, cluster_weight):
        """
        Link clusters based on percentage of documents they share.

        Parameters
        ----------
        cluster_weight: list of dict
            Clusters with entity, cluster id and burstingness score.

        Returns
        -------
        new_cluster_weight: list of dict
            Merged clusters.
        """
        new_cluster_weight = dict(cluster_weight)
        linked = set()

        def get_entity(cluster_id):
            return cluster_weight[cluster_id]["cluster_entity"][cluster_id]

        for cluster_id in cluster_weight:
            entity = get_entity(cluster_id)
            for other_cluster_id in cluster_weight:
                other_entity = get_entity(other_cluster_id)

                if entity == other_entity:
                    continue

                key = frozenset({cluster_id, other_cluster_id})
                if key in new_cluster_weight:
                    continue

                docs = self.cluster_store_.get_docs(entity, cluster_id)
                other_docs = self.cluster_store_.get_docs(
                    other_entity, other_cluster_id
                )

                N_intersection = len(docs & other_docs)
                N_union = len(docs | other_docs)

                if N_intersection / N_union >= self.link_percentage:
                    B_cluster = cluster_weight[cluster_id]["B"]
                    B_other_cluster = cluster_weight[other_cluster_id]["B"]

                    new_entity = {cluster_id: entity, other_cluster_id: other_entity}
                    new_cluster_weight[key] = {
                        "cluster_entity": new_entity,
                        "B": max(B_cluster, B_other_cluster),
                    }

                    linked.add(cluster_id)
                    linked.add(other_cluster_id)

        for cluster_id in linked:
            del new_cluster_weight[cluster_id]

        return new_cluster_weight

    def fit(self, doc_id, date, terms, entities, location=None):
        """
        Processes given document for event detection by adding it into relevant data stores,
        clustering them and detecting bursting clusters.

        Parameters
        ----------
        doc_id: int
            Document id.
        date: datetime
            Publication date and time.
        terms: dict
            Unique terms of document including entities with frequencies (e.g {"word": 2}).
        entities: dict
            Unique named entities with frequencies (e.g. {"boris": 1}).
        location: tuple of floats
            Latitude and longitude.

        Returns
        -------
        results: list of dict or list
            If clusters are ranked, detected events with entity, cluster id
            and burstingness score are returned else an empty list is returned.
        """
        if self.update_time_ is None:
            self.update_time_ = date

        if self.burst_time_ is None:
            self.burst_time_ = date

        results = []

        time_diff = (date - self.update_time_).total_seconds()
        if time_diff >= self.slide_freq:

            self._remove_expired_documents(date)
            self._add_new_documents()
            self.update_time_ = date

            if self.span_ == self.burst_span:

                if self.time_step_ >= self.burst_history + 1:
                    results = self._rank_clusters()

                    time_step_to_remove = (
                        self.time_step_ // self.burst_history - 1
                    ) * self.burst_history + self.time_step_ % self.burst_history

                    self.entity_store_.remove_time_step(time_step_to_remove)
                    self.cluster_store_.remove_window()

                self.entity_store_.set_time_step(self.time_step_)

                self.time_step_ += 1
                self.span_ = 1
                self.burst_time_ = date
            else:
                self.span_ += 1

        self.update_buffer_.appendleft((doc_id, date, terms, entities, location))

        return results

    def flush(self):
        """
        Removes expired documents and adds all documents in the buffer.
        """
        last_update_date = self.update_buffer_[0][1]

        self._remove_expired_documents(last_update_date)
        self._add_new_documents()


class EventDetectorEmbedding(EventDetectorBase):
    """
    Implementation of emebdding-based event detection algorithm.
    
    Parameters
    ----------
    w2vmodel: object
        Gensim Word2Vec model
    embeddings_path: str
        Path to entity embeddings file
    embeddings_map_path:
        Path to id map of entity embeddings file
    min_threshold: int or float, default=0.2
        Minimum similarity threshold required to add
        a document into a cluster.
    match_threshold: int or float, default=0.4
        Minimum similarity threshold required to form a
        cross-entity cluster.
    max_threshold: int or float, default=0.95
        Maximum similarity threshold to filter out a duplicated
        document.
    entity_threshold: int or float, default=4.5
        Upper distance threshold to match similar entities.
    window_hours: int or float, default=1
        Length of event detection window in hours.
    slide_mins: int or float, default=1
        Slide frequency of window in minutes.
    burst_span: int, default=5
        Number of times window must be slid before burst detection.
    burst_threshold: int or float, default=1
        Minimum burtingness score required for a cluster to burst.
    cluster_limit: int, default=5
        Minimum document threshold required for a cluster to burst.
    link_percentage: float, default=0.5
        Minimum percentage of document required to be shared by two
        clusters with different entities to link them.
    burst_history: int, default=10
        Number of historical timeframes to use while calculating
        burtingness score.
    top_clusters: None or int, default=None
        Limits number of events to be returned. If None is given,
        all clusters are returned instead of top clusters.
    top_terms: int, default=20
        Number of top terms to generate a pseudo-documents which
        represents clusters.
    entity_mmap_mode: str, default="r"
        mmap_mode for Numpy array of embeddings.
        See https://bit.ly/32Y5oeE for futher information.
    """
    def __init__(
        self,
        w2vmodel,
        embeddings_path,
        embeddings_map_path,
        min_threshold=0.2,
        match_threshold=0.4,
        max_threshold=0.95,
        entity_threshold=4.5,
        window_hours=1,
        slide_mins=1,
        burst_span=5,
        burst_threshold=1,
        cluster_limit=5,
        link_percentage=0.5,
        burst_history=10,
        top_clusters=None,
        top_terms=20,
        entity_mmap_mode="r",
    ):
        super(EventDetectorEmbedding, self).__init__(
            min_threshold,
            window_hours,
            slide_mins,
            burst_span,
            burst_threshold,
            cluster_limit,
            link_percentage,
            burst_history,
            top_clusters,
        )
        self.match_threshold = match_threshold
        self.entity_threshold = entity_threshold
        self.max_threshold = max_threshold
        self.top_terms = top_terms

        self.cluster_store_ = ClusterStoreEmbedding()
        self.entity_doc_store_ = EntityDocStoreEmbedding()
        self.entity_term_store_ = EntityTermStoreEmbedding()
        self.entity_match_store_ = EntityMatchStore()
        self.entities_wo_wid_ = set()

        self.w2vmodel = w2vmodel
        self.entityvec = np.load(embeddings_path, mmap_mode=entity_mmap_mode)
        with open(embeddings_map_path, "r") as f:
            entitylist = json.load(f)
            self.entitymap = dict(zip(entitylist, range(len(entitylist))))

    def _get_top_terms(self, entity, cluster_id):
        """
        Calculates top terms in a cluster based on tf-idf weights.

        Parameters
        ----------
        entity: dict
        cluster_id: int
        
        Returns
        -------
        top_terms: list of str
            Top terms of given cluster.
        """
        entity_N = self.entity_doc_store_.get_entity_N(entity)
        cluster_size = self.cluster_store_.get_cluster_size(entity, cluster_id)
        tf_sums = self.cluster_store_.get_tf_sums(entity, cluster_id)
        tf_counts = self.cluster_store_.get_tf_counts(entity, cluster_id)
        terms = list(tf_counts.keys())

        weights = []
        for term in terms:
            df = self.entity_term_store_.get_document_frequency(entity, term)
            tf_sum = tf_sums[term]
            tf_count = tf_counts[term]

            tfidfs = [
                calculate_tfidf(tf_sum / tf_count, df, entity_N)
                for _ in range(tf_count)
            ]
            weights.append(sum(tfidfs) / cluster_size)

        top_weights, top_terms = get_top_k(weights, terms, self.top_terms)

        return top_terms

    def _get_most_similar_cluster(self, terms, entity):
        """
        Gets the most similar cluster based on given terms and entity. Similarity score is one minus 
        Word Mover’s Distance between given terms and a pseudo-document for each cluster, which is
        generated using top terms in the cluster. 

        Parameters
        ----------
        terms: dict
        entity: str

        Returns
        -------
        max_similarity, similar_id: float, int
            Maximum calculated similarity and most similar cluster's id.
        """
        clusters = self.cluster_store_.get_clusters(entity)

        max_similarity = 0
        similar_id = None
        for cluster_id in clusters:
            top_terms = self._get_top_terms(entity, cluster_id)
            distance = self.w2vmodel.wmdistance(top_terms, terms)
            similarity = 1 - distance

            if similarity > max_similarity:
                max_similarity = similarity
                similar_id = cluster_id

        return max_similarity, similar_id

    def _assign_cluster(
        self,
        entity,
        location,
        terms,
        doc_id,
        similar_id,
        similarity,
        min_threshold,
        new_cluster,
    ):
        """
        Assign `doc_id` to a cluster given the most similar document. If `new_cluster` parameter
        is True and similarity score is lower than similarity threshold a new cluster is created.

        Parameters
        ----------
        entity: str
        location: tuple of float
        terms: dict
        doc_id: int
        similar_id: int
        similarity: float
        min_threshold: float
        new_cluster: bool
        """
        if similarity >= min_threshold:
            self.entity_doc_store_.set_cluster(entity, doc_id, similar_id)
            self.cluster_store_.add_cluster(entity, similar_id)
            self.cluster_store_.add_doc(entity, similar_id, doc_id, location)
            for term in terms:
                tf = terms[term]
                self.cluster_store_.add_term(entity, similar_id, term, tf)
        elif new_cluster:
            self.entity_doc_store_.set_cluster(entity, doc_id, self.cluster_id_)
            self.cluster_store_.add_cluster(entity, self.cluster_id_)
            self.cluster_store_.add_doc(entity, self.cluster_id_, doc_id, location)
            for term in terms:
                tf = terms[term]
                self.cluster_store_.add_term(entity, self.cluster_id_, term, tf)
            self.cluster_id_ += 1

    def _get_wikidata_key(self, wid):
        """
        Formats Wikidata id in the same form used in embedding map.

        Parameters
        ----------
        wid: str

        Returns
        -------
        key: str
        """
        key = "<http://www.wikidata.org/entity/" + wid + ">"
        return key

    def _find_wid(self, entity):
        """
        Fetches Wikidata id using MediaWiki API.

        Parameters
        ----------
        entity: str

        Returns
        -------
        str or None:
            Wikidata id for given entity 
            or None if it is not found or not in entity map.
        """
        wid = get_wid(entity)

        if wid is None:
            return None

        key = self._get_wikidata_key(wid)

        if key in self.entitymap:
            return wid

        return None

    def _get_embedding(self, wid):
        """
        Gets embedding vector for given Wikidata id.

        Parameters
        ----------
        wid: str

        Returns
        -------
        embedding: numpy.ndarray
        """
        key = self._get_wikidata_key(wid)
        index = self.entitymap[key]
        embedding = self.entityvec[index]
        return embedding

    def _calculate_entity_matches(self, entity, wid):
        """
        Calculates similar entities and stores it into entity match store.

        Parameters
        ----------
        entity: str
        wid: str
        """
        entities = self.entity_match_store_.get_entities()
        for other_entity in entities:
            if entity == other_entity:
                continue

            other_wid = self.entity_match_store_.get_wid(entity)

            embedding = self._get_embedding(wid)
            other_embedding = self._get_embedding(other_wid)
            distance = euclidean(embedding, other_embedding)

            if distance <= self.entity_threshold:
                self.entity_match_store_.add_match(entity, other_entity)

                if self.entity_match_store_.is_in(other_entity):
                    self.entity_match_store_.add_match(other_entity, other_entity)

    def _add_to_entity(
        self, entity, doc_id, location, terms, min_threshold, new_cluster=True
    ):
        """
        Clusters document for given entity. If `new_cluster` is True, a new cluster can be
        created, otherwise the document could only be assigned to existing clusters. If
        similarty score of the most similar cluster is higher than a threshold no
        assignment is made.

        Parameters
        ----------
        entity: str
        doc_id: int
        location: tuple of float
        terms: dict
        min_threshold: float
        new_cluster: bool
        
        Returns
        -------
        bool:
            True if document is assigned to a cluster, otherwise False. 
        """
        self.cluster_store_.setdefault(entity)

        max_similarity, similar_id = self._get_most_similar_cluster(terms, entity)

        if not new_cluster and max_similarity < min_threshold:
            return False

        if max_similarity > self.max_threshold:
            return False

        self.entity_store_.setdefault(self.time_step_, entity)
        self.entity_store_.increment(self.time_step_, entity)

        self.entity_doc_store_.setdefault(entity, doc_id)

        for term in terms:
            self.entity_term_store_.setdefault(entity, term)
            self.entity_term_store_.add_term(entity, term)

        self._assign_cluster(
            entity,
            location,
            terms,
            doc_id,
            similar_id,
            max_similarity,
            min_threshold,
            new_cluster,
        )
        return True

    def _add_new_documents(self):
        """
        Process new documents for clustering. Updates tf-idf values for generating
        pseudo-documents, finds most similar document under same entity, assigns it
        to a cluster and employs cross-clustering if there is a match between entities.
        """
        while self.update_buffer_:
            doc_id, date, terms, entities, location = self.update_buffer_.pop()
            self.doc_store_.setdefault(doc_id)
            self.window_.append((doc_id, date, location))

            for term in terms:
                if term not in entities:
                    tf = terms[term]
                    self.doc_store_.add_term(doc_id, term, tf)

            for entity in entities:
                terms_wo_entity = dict(terms)
                del terms_wo_entity[entity]

                tf = entities[entity]
                self.doc_store_.add_entity(doc_id, entity, tf)

                result = self._add_to_entity(
                    entity, doc_id, location, terms_wo_entity, self.min_threshold
                )
                if not result:
                    continue

                if entity in self.entities_wo_wid_:
                    continue

                if not self.entity_match_store_.is_in(entity):

                    wid = self._find_wid(entity)
                    if wid is None:
                        self.entities_wo_wid_.add(entity)
                        continue

                    self.entity_match_store_.setdefault(entity, wid)
                    self._calculate_entity_matches(entity, wid)

                matches = self.entity_match_store_.get_matches(entity)
                for match in matches:
                    if match in entities:
                        continue

                    if not self.entity_doc_store_.has_entity(match):
                        continue

                    self._add_to_entity(
                        match,
                        doc_id,
                        location,
                        terms_wo_entity,
                        self.match_threshold,
                        new_cluster=False,
                    )

    def _remove_from_entity(self, entity, doc_id, location, terms, entities):
        """
        Removes information of document linked to given entity.

        Parameters
        ----------
        entity: str
        doc_id: int
        location: tuple of float
        terms: dict
        entities: dict
        """
        cluster_id = self.entity_doc_store_.get_cluster(entity, doc_id)
        self.entity_doc_store_.remove_doc(entity, doc_id)

        for term in terms:
            self.entity_term_store_.remove_term(entity, term)
            tf = self.doc_store_.get_term_tf(doc_id, term)
            self.cluster_store_.remove_term(entity, cluster_id, term, tf)

        for other_entity in entities:
            self.entity_term_store_.remove_term(entity, other_entity)
            tf = self.doc_store_.get_entity_tf(doc_id, other_entity)
            self.cluster_store_.remove_term(entity, cluster_id, other_entity, tf)

        self.cluster_store_.remove_doc(entity, cluster_id, doc_id, location)

    def _remove_expired_documents(self, date):
        """
        Removes expired documents from window along with the information
        related to those document.

        Parameters
        ----------
        date: datetime
            Time of current document.
        """
        while self.window_:
            doc_id, doc_date, doc_location = self.window_.popleft()
            doc_time_diff = (date - doc_date).total_seconds()
            if doc_time_diff >= self.window_len:
                entities = self.doc_store_.get_entities(doc_id)

                terms = self.doc_store_.get_terms(doc_id)
                for entity in entities:
                    other_entities = dict(entities)
                    del other_entities[entity]

                    if not self.entity_doc_store_.has_doc(entity, doc_id):
                        continue

                    self._remove_from_entity(
                        entity, doc_id, doc_location, terms, other_entities
                    )
                    if self.entity_match_store_.is_in(entity):
                        matches = self.entity_match_store_.get_matches(entity)
                        for match in matches:
                            if match in entities:
                                continue

                            if self.entity_doc_store_.has_doc(entity, doc_id):
                                self._remove_from_entity(
                                    match, doc_id, doc_location, terms, other_entities
                                )

                self.doc_store_.remove_doc(doc_id)

            else:
                self.window_.appendleft((doc_id, doc_date, doc_location))
                break
