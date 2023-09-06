import csv
import json

from eventdetector.util import get_config


def main(result_path, data_path):
    doc_text = {}
    doc_geo = {}
    geo_count = 0
    with open(result_path, "r") as result_file, open(data_path, "r") as data_file:
        reader = csv.reader(result_file)

        for line in reader:
            time_step, date, entities, cluster_ids, score, docs = line
            entities, cluster_ids, docs = eval(entities), eval(cluster_ids), eval(docs)
            print(
                "Time step:",
                time_step,
                "Date:",
                date,
                "Entity:",
                entities,
                "Cluster id:",
                cluster_ids,
                "Score",
                score,
            )
            print("-" * 70)

            has_geo = False
            for doc_id in docs:
                if doc_id in doc_text:
                    print(doc_text[doc_id])
                    if doc_geo[doc_id] is not None:
                        has_geo = True
                else:
                    for line in data_file:
                        data = json.loads(line)
                        other_doc_id = data["_id"]
                        text = data["text"]

                        location = None
                        if "coordinates" in data:
                            location = data["coordinates"]
                            doc_geo[other_doc_id] = location
                        else:
                            doc_geo[other_doc_id] = None

                        doc_text[other_doc_id] = text

                        if other_doc_id == doc_id:
                            print(text)
                            if location is not None:
                                has_geo = True
                            break

            if has_geo:
                geo_count += 1

            print("*" * 70)
            print("Has geo:", has_geo)
            print("-" * 70)

    print("Geo clusters:", geo_count)


if __name__ == "__main__":
    config = get_config()

    result_path = config["output"]["event"]
    data_path = config["twitter"]["input"]

    main(result_path, data_path)
