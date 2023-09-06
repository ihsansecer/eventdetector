import csv
import json
import time
import cProfile
from datetime import datetime

from requests.exceptions import ReadTimeout
from stanza.server import CoreNLPClient
from gensim import downloader
from joblib import load
from tqdm import tqdm

from eventdetector.filter import (
    is_retweet,
    is_spam,
    prepare_entities,
    prepare_tokens_base,
    prepare_tokens_embedding,
)
from eventdetector.constant import stopwords, stopkinds, allowed_pos
from eventdetector.spam import StemmedTfidfVectorizer
from eventdetector.util import get_config
from eventdetector import EventDetectorBase, EventDetectorEmbedding


def parse_data(data):
    text = data["text"]
    doc_id = data["_id"]
    date_str = data["date"]["$date"]
    date = datetime.strptime(date_str[:-1], "%Y-%m-%dT%H:%M:%S")

    location = None
    if "coordinates" in data:
        location = data["coordinates"]

    source = data["source"]

    return doc_id, text, date, location, source


def write_events(ed, events, date, writer, stats):
    for event in events:
        score, cluster_entity = event
        docs = set()
        for cluster_id in cluster_entity:
            entity = cluster_entity[cluster_id]
            docs |= ed.cluster_store_.get_docs(entity, cluster_id)

        writer.writerow(
            [
                ed.time_step_,
                date,
                set(cluster_entity.keys()),
                set(cluster_entity.values()),
                score,
                list(docs),
            ]
        )

        stats["n_bursting"] += 1


def get_kind_stats(annotated, is_geo, stats):
    for mention in annotated.mentions:
        kind = mention.ner.lower()
        if is_geo:
            stats["geo"]["kind"].setdefault(kind, 0)
            stats["geo"]["kind"][kind] += 1
        else:
            stats["nongeo"]["kind"].setdefault(kind, 0)
            stats["nongeo"]["kind"][kind] += 1


def get_filter_count(is_geo, stats, key):
    if is_geo:
        stats["geo"][key] += 1
    else:
        stats["nongeo"][key] += 1


def get_source_count(source, is_geo, stats):
    if is_geo:
        stats["geo"]["source"].setdefault(source, 0)
        stats["geo"]["source"][source] += 1
    else:
        stats["nongeo"]["source"].setdefault(source, 0)
        stats["nongeo"]["source"][source] += 1


def save_stats(stats, out_stats):
    with open(out_stats, "w") as f:
        json.dump(stats, f)


def main(
    algorithm,
    in_path,
    in_vectorizer,
    in_model,
    out_path,
    out_stats,
    embeddings_path,
    embeddings_map_path,
    config,
):
    stats = {
        "geo": {
            "n": 0,
            "rt": 0,
            "spam": 0,
            "noentity": 0,
            "notoken": 0,
            "kind": {},
            "source": {},
        },
        "nongeo": {
            "n": 0,
            "rt": 0,
            "spam": 0,
            "noentity": 0,
            "notoken": 0,
            "kind": {},
            "source": {},
        },
        "ent_time": 0,
        "ent_proc": 0,
        "spam_time": 0,
        "spam_proc": 0,
        "n_bursting": 0,
        "ed_bust_time": [],
        "ed_norm_time": [],
    }

    start_time = time.time()

    nlp = CoreNLPClient(start_server=False, be_quiet=False)

    if algorithm == "base":
        ed = EventDetectorBase(**config[algorithm]["parameters"])
    elif algorithm == "embedding":
        w2vmodel = downloader.load("glove-twitter-200")
        w2vmodel.init_sims(replace=True)
        ed = EventDetectorEmbedding(
            w2vmodel,
            embeddings_path,
            embeddings_map_path,
            **config[algorithm]["parameters"]
        )

    spam_transformer = StemmedTfidfVectorizer(
        sublinear_tf=True, max_features=150000, stop_words=stopwords
    )
    vectorizer_dict = load(in_vectorizer)
    for key in vectorizer_dict:
        setattr(spam_transformer, key, vectorizer_dict[key])
    spam_model = load(in_model)

    print("Start-up time:", time.time() - start_time)

    if config["profile"]:
        pr1 = cProfile.Profile()
        pr2 = cProfile.Profile()

    with nlp:
        with open(in_path, "r") as infile, open(out_path, "w") as outfile:
            writer = csv.writer(outfile)

            for i, line in enumerate(tqdm(infile)):
                if config["line_limit"] is not None and i == config["line_limit"]:
                    break

                if config["profile"]:
                    pr1.enable()

                data = json.loads(line)
                doc_id, text, date, location, source = parse_data(data)
                is_geo = location is not None

                if is_retweet(text):
                    get_filter_count(is_geo, stats, "rt")
                    continue

                start_time = time.time()
                spam = is_spam(text, spam_model, spam_transformer)
                stats["spam_time"] += time.time() - start_time
                stats["spam_proc"] += 1

                if spam:
                    get_filter_count(is_geo, stats, "spam")
                    continue

                try:
                    start_time = time.time()
                    annotated = nlp.annotate(text)
                    stats["ent_time"] += time.time() - start_time
                    stats["ent_proc"] += 1
                except ReadTimeout:
                    print("Nlp server throwed timeout exception")
                    continue
                except Exception:
                    print("Nlp server throwed another exception")
                    continue

                entities = prepare_entities(annotated, stopwords, stopkinds)

                get_kind_stats(annotated, is_geo, stats)

                if len(entities) < 1:
                    get_filter_count(is_geo, stats, "noentity")
                    continue

                if algorithm == "base":
                    tokens = prepare_tokens_base(annotated, stopwords, allowed_pos)
                elif algorithm == "embedding":
                    tokens = prepare_tokens_embedding(
                        annotated, stopwords, allowed_pos, w2vmodel
                    )

                if len(tokens) < config["token_limit"]:
                    get_filter_count(is_geo, stats, "notoken")
                    continue

                tokens.update(entities)

                get_filter_count(is_geo, stats, "n")
                get_source_count(source, is_geo, stats)

                start_time = time.time()

                if config["profile"]:
                    pr2.enable()

                events = ed.fit(doc_id, date, tokens, entities, location=location)

                if config["profile"]:
                    pr2.disable()

                time_took = time.time() - start_time

                if events:
                    write_events(ed, events, date, writer, stats)
                    stats["ed_bust_time"].append(time_took)
                else:
                    stats["ed_norm_time"].append(time_took)

                if config["profile"]:
                    pr1.disable()

                outfile.flush()

                if i != 1 and i % 10000 == 0:
                    stats["n_clusters"] = ed.cluster_id_
                    save_stats(stats, out_stats)

    stats["n_clusters"] = ed.cluster_id_
    save_stats(stats, out_stats)

    if config["profile"]:
        pr1.dump_stats("1" + config["profile_path"])
        pr2.dump_stats("2" + config["profile_path"])


if __name__ == "__main__":
    config = get_config()

    algorithm = config["algorithm"]

    in_path = config["twitter"]["input"]
    in_vectorizer = config["spam"]["vectorizer"]
    in_model = config["spam"]["model"]
    out_path = config["output"]["event"]
    out_stats = config["output"]["eventstats"]

    embeddings_path = config["wikidata"]["embedding"]
    embeddings_map_path = config["wikidata"]["titlemap"]

    main(
        algorithm,
        in_path,
        in_vectorizer,
        in_model,
        out_path,
        out_stats,
        embeddings_path,
        embeddings_map_path,
        config,
    )
