from contextlib import ExitStack
from datetime import datetime
import json

from tqdm import tqdm

from eventdetector.util import get_config


def get_date(data):
    date_str = data["date"]["$date"]
    date = datetime.strptime(date_str[:-1], "%Y-%m-%dT%H:%M:%S")
    return date


def add_source(data, key, stats):
    source = data["source"]
    stats[key].setdefault(source, 0)
    stats[key][source] += 1


def main(geo_path, nongeo_path, out_path, stats_path):
    pre_geo_line = ""
    pre_nongeo_line = ""

    stats = {"geo": 0, "nongeo": 0, "geosource": {}, "nongeosource": {}}

    pbar = tqdm()
    with ExitStack() as stack:
        geo_file = stack.enter_context(open(geo_path, "r"))
        nongeo_file = stack.enter_context(open(nongeo_path, "r"))
        out_file = stack.enter_context(open(out_path, "w"))

        while True:
            if pre_geo_line:
                geo_line = pre_geo_line
            else:
                geo_line = geo_file.readline()

            if pre_nongeo_line:
                nongeo_line = pre_nongeo_line
            else:
                nongeo_line = nongeo_file.readline()

            if not geo_line and not nongeo_line:
                break

            if geo_line:
                geo_data = json.loads(geo_line)
                geo_date = get_date(geo_data)

            if nongeo_line:
                nongeo_data = json.loads(nongeo_line)
                nongeo_date = get_date(nongeo_data)

            if geo_line and nongeo_line:
                if nongeo_date < geo_date:
                    out_file.write(nongeo_line)
                    pre_geo_line = geo_line
                    pre_nongeo_line = ""

                    add_source(nongeo_data, "nongeosource", stats)
                    stats["nongeo"] += 1
                else:
                    out_file.write(geo_line)
                    pre_nongeo_line = nongeo_line
                    pre_geo_line = ""

                    add_source(geo_data, "geosource", stats)
                    stats["geo"] += 1

            elif nongeo_line:
                out_file.write(nongeo_line)
                pre_geo_line = geo_line
                pre_nongeo_line = ""

                add_source(nongeo_data, "nongeosource", stats)
                stats["nongeo"] += 1
            elif geo_line:
                out_file.write(geo_line)
                pre_nongeo_line = nongeo_line
                pre_geo_line = ""

                add_source(geo_data, "geosource", stats)
                stats["geo"] += 1

            pbar.update(1)

    with open(stats_path, "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    config = get_config()

    geo_path = config["twitter"]["geo"]
    nongeo_path = config["twitter"]["nongeo"]
    out_path = config["twitter"]["input"]
    stats_path = config["output"]["mergestats"]

    main(geo_path, nongeo_path, out_path, stats_path)
