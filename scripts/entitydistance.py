import json
import random
import itertools

from tqdm import tqdm
from scipy import spatial, stats
import numpy as np

from eventdetector.util import get_config


def get_data(embeddings_path, embeddings_map_path, parent_path):
    embeddings = np.load(embeddings_path, mmap_mode="r")

    with open(embeddings_map_path, "r") as f:
        embeddings_list = json.load(f)
        embeddings_map = dict(zip(embeddings_list, range(len(embeddings_list))))

    with open(parent_path, "r") as f:
        parent_map = json.load(f)

    return embeddings, embeddings_map, parent_map


def get_nonchlidren(parent_ids, children_ids, n):
    parent_ids = set(parent_ids)

    count = 0
    selected_ids = set()
    while parent_ids:
        selected_id = random.sample(parent_ids, k=1)[0]
        parent_ids.discard(selected_id)
        if selected_id not in children_ids:
            selected_ids.add(selected_id)
            count += 1
        if count == n:
            break
    return selected_ids


def get_embedding(wid, embeddings, embeddings_map):
    name = "<http://www.wikidata.org/entity/" + wid + ">"
    if name not in embeddings_map:
        return None
    offset = embeddings_map[name]
    return embeddings[offset]


def set_values(name, distance_chi, distance_non, vals):
    if np.isnan(distance_chi) or np.isnan(distance_chi):
        return None

    vals.setdefault(name, {})
    vals[name].setdefault("child", [])
    vals[name].setdefault("nonchild", [])
    vals[name].setdefault("n", 0)
    vals[name]["n"] += 1

    if distance_chi < distance_non:
        vals[name].setdefault("correct", 0)
        vals[name]["child"].append(distance_chi)
        vals[name]["nonchild"].append(distance_non)
        vals[name]["correct"] += 1
    else:
        vals[name].setdefault("wrong", 0)
        vals[name]["wrong"] += 1


def print_results(vals):
    print("Distances are computed")

    for name in vals:
        child = vals[name]["child"]
        nonchild = vals[name]["nonchild"]
        n = vals[name]["n"]

        print("Function:", name)
        print("Correct:", vals[name]["correct"], "Wrong:", vals[name]["wrong"])
        print("Correct percentage:", round(vals[name]["correct"] / n, 3))
        print(
            "Mean child:",
            round(np.mean(child), 3),
            "Std child:",
            round(np.std(child), 3),
            "Median child:",
            round(np.median(child), 3),
            "Mad child:",
            round(stats.median_absolute_deviation(child), 3),
        )
        print(
            "Mean nonchild:",
            round(np.mean(nonchild), 3),
            "Std nonchild:",
            round(np.std(nonchild), 3),
            "Median nonchild:",
            round(np.median(nonchild), 3),
            "Mad nonchild:",
            round(stats.median_absolute_deviation(nonchild), 3),
        )


def main(embeddings_path, embeddings_map_path, parent_path):
    print("Loading data")
    embeddings, embeddings_map, parent_map = get_data(
        embeddings_path, embeddings_map_path, parent_path
    )
    parent_ids = set(parent_map.keys())
    print("Data is loaded")

    vals = {}
    functions = [
        spatial.distance.cosine,
        spatial.distance.euclidean,
        spatial.distance.cityblock,
        spatial.distance.correlation,
        spatial.distance.chebyshev,
    ]

    print("Computing distances")
    for parent_id in tqdm(parent_ids):

        parent_embedding = get_embedding(parent_id, embeddings, embeddings_map)
        if parent_embedding is None:
            continue

        children_ids = parent_map[parent_id]
        size = min(len(children_ids), 50)
        children_ids = children_ids[:size]

        nonchildren_ids = get_nonchlidren(parent_ids, children_ids, size)
        if not nonchildren_ids:
            continue

        product = list(itertools.product(children_ids, nonchildren_ids))

        for chi_id, non_id in product:
            chi_embedding = get_embedding(chi_id, embeddings, embeddings_map)
            non_embedding = get_embedding(non_id, embeddings, embeddings_map)
            if chi_embedding is None or non_embedding is None:
                continue

            for function in functions:

                distance_chi = function(chi_embedding, parent_embedding)
                distance_non = function(non_embedding, parent_embedding)

                name = function.__name__
                set_values(name, distance_chi, distance_non, vals)

    print_results(vals)


if __name__ == "__main__":
    config = get_config()

    embeddings_path = config["wikidata"]["embedding"]
    embeddings_map_path = config["wikidata"]["titlemap"]
    parent_path = config["wikidata"]["parentmap"]

    main(embeddings_path, embeddings_map_path, parent_path)
