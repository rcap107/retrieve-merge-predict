Benchmarking Join Suggestions
===
This repository contains the code for implementing and running the pipeline described in the paper "Retrieve, Merge, Predict: Augmenting Tables with Data Lakes
(Experiment, Analysis & Benchmark Paper).

The objective is modeling a situation where an user is trying to execute ML tasks on some base data, enriching it by
using new tables found in a data lake using retrieval methods.

The join candidates are merged with the base table under study, before training a ML model (either Catboost, or a linear model) to evaluate the performance
before and after the merge.

We use YADL as our data lake, a synthetic data lake based on the YAGO3 knowledge base. The YADL variants used in the paper
are available [on Zenodo](https://zenodo.org/doi/10.5281/zenodo.10600047).

The code for preparing the YADL variants can be found in [this repo](https://github.com/rcap107/prepare-data-lakes).

# Installing the requirements
We strongly recommend to use conda environments to fetch the required packages. File `environment.yaml` contains the
required dependencies and allows to create directly a conda environment:
```
conda env create --file=environment.yaml
conda activate bench
```
Then, install the remaining dependencies with pip:
```
pip install -r requirements.txt
```

# Downloading YADL
It is possible to download YADL from [the zenodo repository](https://zenodo.org/doi/10.5281/zenodo.10600047) using `wget` in the root folder:
```sh
wget -O data/binary_update.tar.gz https://zenodo.org/records/10600048/files/binary_update.tar.gz
wget -O data/wordnet_full.tar.gz https://zenodo.org/records/10600048/files/wordnet_full.tar.gz
```
Additional files may be downloaded from zenodo using the same command:
```sh
wget -O destination_file_name path_to_file
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

## Preparing the Retrieval methods
This step is an offline operation during which the retrieval methods are prepared by building the data structures they rely on to function. The preparation of these data structures can require a substantial amount of
time and disk space and is not required for the querying step, as such it can be executed only once for each data lake.

Different retrieval methods require different data structures and different starting configurations, which should be stored in `config/retrieval/prepare`. In all configurations,
`n_jobs` is the number of parallel jobs that will be executed; if it set to -1, all available
CPU cores will be used.

### Execution
```
python prepare_retrieval_methods.py [--repeats REPEATS] config_file
```
`config_file` is the path to the configuration file. `repeats` is a parameter that can be
added to re-run the current configuration `repeats` times to track the time.

### Config files
Here is a sample configuration for MinHash.
```toml
[["minhash"]]
data_lake_variant="wordnet_full"
thresholds=20
oneshot=true
num_perm=128
n_jobs=-1
```

`ExactMatchingIndex` work only for single query columns. As such,
each case `queryt_table-query_column` must be defined independently:

```toml
[["exact_matching"]]
data_dir="data/metadata/wordnet_full"
base_table_path="data/source_tables/yadl/company_employees-yadl.parquet"
query_column="col_to_embed"
n_jobs=-1

[["exact_matching"]]
data_dir="data/metadata/open_data_us"
base_table_path="data/source_tables/housing_prices-open_data.parquet"
query_column="County"
n_jobs=-1
```

The configuration parser will prepare the data structures (specifically, the counts) for each case provided in the configuration file.

Configuration files whose name start with `prepare` in `config/retrieval/prepare` are example configuration files for the index preparation step.

## Querying the retrieval methods
Because of how some methods are implemented, the querying operation can incur in significant costs because of the need to load the index structures in memory.

For conveniences, queries are executed offline and persisted on disk so that they can be loaded at runtime during the execution of the pipeline.

Configuration files stored in `config/retrieval/query` are example configuration files for the index query step.

# Executing the pipeline
