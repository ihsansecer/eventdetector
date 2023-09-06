# EventDetector

An implementation of work presented in the dissertation, "Real-Time Entity-Based Event Detection
in Geolocated Social Text Streams" in part fulfilment of the requirements of the
Degree of Master of Science at the University of Glasgow.

## Quick Start

To install library:

`pip install .`

To setup config file copy provided sample as `config.yaml` and edit accordingly (check [this section](#config) for more information):

`cp config.sample.yaml config.yaml && nano config.yaml`

To run `main.py` first start a Stanford CoreNLP Server. Check [this section](#stanford-corenlp-server) for more information on CoreNLP.

## Project Structure

`eventdetector/` includes core classes and functions implementing our event detection approach:
- `conftest.py` is an empty file required for pytest.
- `constant.py` includes a list of allowed POS tags, stopwords and eliminated entity kinds.
- `detector.py` includes inverted index-based and embedding-based event detector classes.
- `filter.py` includes filtering related functions.
- `spam.py` includes some utilities for spam classification.
- `store.py` includes classes of stores which are responsible to keep most of information required for event detection process.
- `util.py` includes some utility functions.
- `tests/` includes unit tests.

`scripts/` includes long-running experimentation code:
- `entitydistance.py` runs experimentation for comparing distances between entities.
- `main.py` runs event detection pipeline for our experiments.
- `merge.py` merges geo and non-geo-datasets.
- `showcluster.py` prints cluster of documents from output of `main.py`.
- `spamexperiment.py` runs spam classification code on whole dataset.
- `spamtrain.py` trains selected logistic regression model and saves neccesary data into disk.

`notebooks/` includes short-running experimentation code:
- `eda-geo.ipynb` is the code for exploratory analysis of geo-data.
- `spam.ipynb` is the code for hyper-parameter tuning of spam classifier.
- `stats.ipynb` is the code for analysing statistics from results of event detectors.

## Config

Explanation of parameters for base (inverted index) and embedding detectors could be found in `eventdetector/detector.py`.
- algorithm: use `base` for inverted index and `embedding` for embedding methods.
- token_limit: use 5 for embedding method to reproduce experiments.
- line_limit: use if you want to run limited number of documents.
- profile: assign `true` to activate profiling, otherwise `false`.
- profile_path: path to profiling outputs.
- twitter: merged, geo and nongeo documents for event detection.
- wikidata: Wikidata datasets (check [this section](#datasets) for datasets).
- spam: paths to spam dataset and to save trained models.
- output: output data collected from event detection process.

## Datasets

- Geo-datasets could be found [here](https://drive.google.com/file/d/1qep2wWRLT-iBP0c5Ve1BXraD6GWWjTLh/view) for London and [here](https://drive.google.com/file/d/1T4nLwJIFkPL5zWmYxgc97Y81RoMwG3Cd/view) for New York. Non-geo-datasets are not uploaded due to their large size.
- Training dataset for spam classifier could be found [here](http://infolab.tamu.edu/data/social_honeypot_icwsm_2011.zip).
- Wikidata id for embeddings could be found [here](https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_names.json.gz) and Numpy arrays of embedding vectors could be downloaded from [here](https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_vectors.npy.gz).
- Parent-child relationships (`par_child_dict.json`) used for `entitydistance.py` could be found [heare](https://zenodo.org/record/3364465#.X2yu1s9Kjg4).
- Stopwords must be downloaded by using command `python -m nltk.downloader stopwords` after installation of eventdetector.
- Word embeddings will be downloaded automatically by Gensim library.

## Running Unit Tests

Install development requirements:

`pip install -r requirements-dev.txt`

Run pytest:

`pytest --cov=eventdetector`

## Stanford CoreNLP Server

It can be downloaded from [here](https://stanfordnlp.github.io/CoreNLP/download.html).

To ensure reproducibility while running `main.py`:
- Use annotators as `-annotators "tokenize,ssplit,pos,lemma,ner"`
- Use a server properties file `-serverProperties propertiesFile` including lines:
  - `pos.model = ./twittermodels/gate-EN-twitter.model`
  - `parse.model = edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz`
  - `ner.model = edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz, edu/stanford/nlp/models/ner/english.muc.7class.caseless.distsim.crf.ser.gz, edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz`

Twitter POS-tagger could be found [here](https://gate.ac.uk/wiki/twitter-postagger.html). 

After these steps it could be started using a command similar to:

`java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,ner" -port 9000 -timeout 15000 -threads 4 -serverProperties server.properties`

## Coding Standards 

`black` and `flake8` libraries are used to preserve code quality and coding standards compatible to pep8.

Most of the classes and functions under `eventdetector/` folder has docstrings which describe their functionality.
