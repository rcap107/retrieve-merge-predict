Benchmarking Join Suggestions
===
This repository contains the code for implementing and running the pipeline described in the paper "A Benchmarking Data
Lake for Join Discovery and Learning with Relational Data".

The objective is modeling a situation where an user is trying to execute ML tasks on some base data, enriching it by
using new tables found in a data lake through Join Discovery methods.

The candidates produced by the join discovery methods are used to augment the base table, then the performance of the
joined tables is compared to that of the base table by training a regressor with Catboost and comparing the R2 score
measured before and after joining.

We use YADL as our data lake, a synthetic data lake based on the YAGO3 knowledge base. The YADL variants used in the paper
are available on Zenodo: https://zenodo.org/record/8015298

The code for preparing the YADL variants can be found in this repo: https://github.com/rcap107/prepare-data-lakes

# Installing the requirements
We strongly recommend to use conda environments to fetch the required packages. File `environment.yaml` contains the
required dependencies and allows to create directly a conda environment:
```
conda env create --file=environment.yaml
conda activate bench-repro
```
Then, install the remaining dependencies with pip:
```
pip install -r requirements.txt
```

# Preparing the environment
Once the required python environment has been prepared it is necessary to prepare the files required
for the execution of the pipeline.

For efficiency reasons and to avoid running unnecessary operations when testing different components, the pipeline has
been split in different modules that have to be run in sequence.

## Preparing the metadata
Given a data lake version to evaluate, the first step is preparing a metadata file for each table in the data lake. This
metadata is used in all steps of the pipeline.

The script `prepare_metadata.py`is used to generate the files for a given data lake case.

Use the command:
```
python prepare_metadata.py DATA_FOLDER
```
where `DATA_FOLDER` is the root path of the data lake.

The script will recursively scan all folders found in `DATA_FOLDER` and generate a json file for each parquet file
encountered.

## Preparing the Join Discovery methods
This step is an offline operation during which the join discovery methods are prepared by building the data structures they rely on to function. The preparation of these data structures can require a substantial amount of
time and disk space and is not required for the querying step, as such it can be executed only once for each data lake.

Different JD methods require different data structures and different starting configurations, which should be stored in `config/join_discovery`.

Here are some sample configurations for the supported join discovery methods:

```
[["lazo"]]
data_lake_variant="wordnet_big"
host="localhost"
port=15449

[["minhash"]]
data_lake_variant="wordnet_big"
thresholds=20
oneshot=true
num_perm=128
n_jobs=-1
```

The JD methods `CountVectorizerIndex` and `ExactMatchingIndex` work only for single query columns. As such,
each case `queryt_table-query_column` must be defined independently:

```
[["count_vectorizer"]]
data_dir="data/metadata/wordnet_big"
base_table_path="data/source_tables/us-presidential-results-yadl.parquet"
query_column="col_to_embed"
n_jobs=-1

[["count_vectorizer"]]
data_dir="data/metadata/open_data_us"
base_table_path="data/source_tables/us-presidential-results-yadl.parquet"
query_column="county_name"
n_jobs=-1
```

The configuration parser will prepare the data structures (specifically, the counts) for each case provided in the configuration file.

Configuration files whose name start with `prepare` in `config/join_discovery` are example configuration files for the index preparation step.

## Querying the JD methods
Because of how some methods are implemented, the querying operation can incur in significant costs because of the need to load the index structures in memory.

For this work we are not directly interested in the querying performance (i.e., the time required to query the index), so for convenience all queries are executed offline and persisted on disk so that they can be loaded at runtime during the execution of the pipeline.



Configuration files whose name start with `query` in `config/join_discovery` are example configuration files for the index query step.
