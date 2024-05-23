---
permalink: /docs/preparation
layout: page
title: Preparing the environment
prev_page: /docs/resources
next_page: /docs/execution
---

- TOC
{:toc}
Once the required python environment has been built, it is necessary to prepare the files required
for the execution of the pipeline.

For efficiency reasons and to avoid running unnecessary operations when testing different components, the pipeline has
been split in different modules that have to be run in sequence.

## Preparing the metadata
Given a data lake version to evaluate, the first step is preparing a metadata file for each table in the data lake. This
metadata is used in all steps of the pipeline.

The script `prepare_metadata.py`is used to generate the files for a given data lake case.

**NOTE:** This scripts assumes that all tables are saved in `.parquet` format, and will raise an error if it finds no `.parquet`
files in the given path. Please convert your files to parquet before running this script. The tables we provide are already in parquet. 

Use the command:
```
python metadata_creation.py PATH_DATA_FOLDER
```
where `PATH_DATA_FOLDER` is the root path of the data lake. The stem of `PATH_DATA_FOLDER` will be used as identifier for 
the data lake throughout the program (e.g., for `data/binary_update`, the data lake will be stored under the name `binary_update`).

The script will recursively scan all folders found in `PATH_DATA_FOLDER` and generate a json file for each parquet file
encountered. By providing the `--flat` parameter, it is possible to scan only the files in the root directory rather than 
working on all folders and files. 

Metadata will be saved in `data/metadata/DATA_LAKE_NAME`, with an auxiliary file stored in `data/metadata/_mdi/md_index_DATA_LAKE_NAME.pickle`.

## Preparing the Retrieval methods
This step is an offline operation during which the retrieval methods are prepared by building the data structures they rely on to function. This operation can require a long time and a large amount of disk space depending on the method (refer to the paper for more detail). Preparing each retrieval method must be done once for each different data lake variant; however, once the preparation is done querying may be done without repeating the preparation. 

Different retrieval methods require different data structures and different starting configurations, which should be stored in `config/retrieval/prepare`. In all configurations,
`n_jobs` is the number of parallel jobs that will be executed; if it set to -1, all available
CPU cores will be used.

```sh
python prepare_retrieval_methods.py [--repeats REPEATS] config_file
```
`config_file` is the path to the configuration file. `repeats` is a parameter that can be
added to re-run the current configuration `repeats` times (this should be used only for measuring the time required 
for running the indexing operation).

### Config files
Here is a sample configuration for MinHash.
```toml
[["minhash"]]
data_lake_variant="wordnet_full"
num_perm=128
n_jobs=-1
```

`ExactMatchingIndex` works only for single query columns. As such,
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

The configuration parser will prepare the data structures for each case provided in the configuration file.

Configuration files whose name start with `prepare` in `config/retrieval/prepare` are example configuration files for the index preparation step.

To prepare the retrieval methods for data lake `binary_update`:
```sh
python prepare_retrieval_methods.py config/retrieval/prepare/prepare-exact_matching-binary_update.toml
python prepare_retrieval_methods.py config/retrieval/prepare/prepare-minhash-binary_update.toml
```
This will create the index structures for the different retrieval methods in `data/metadata/_indices/binary_update`. 

Data lake preparation should be repeated for any new data lake, and each data lake will have its own directory in `data/metadata/_indices/`.

## Querying the retrieval methods
The querying operation is decoupled from the indexing step for practical reasons (querying is much faster than indexing). 
Moreover, methods such as MinHash attempt to optimize the query operation by building the data structures offline in the indexing
step. 

For these reason, querying is done using the `query_indices.py` script and is based on the configurations in `config/retrieval/query`.

In principle, queries could be done at runtime during the pipeline execution. For efficiency and simplicity, they are executed
offline and stored in `results/query_results`. The pipeline then loads the appropriate query at runtime. 

To build the queries for `binary_update`:
```sh
python query_indices.py config/retrieval/query/query-minhash-binary_update.toml
python query_indices.py config/retrieval/query/query-minhash_hybrid-binary_update.toml
python query_indices.py config/retrieval/query/query-exact_matching-binary_update.toml
```

### Hybrid MinHash
To use the Hybrid MinHash variant, the `query` configuration file should include the parameter `hybrid=true`: the re-ranking
operation is done at query time. 

## Starmie
We use Starmie in some of our experiments, however executing the code is not straightforward due to package incompatibilities. For this reason, Starmie preparation for each data lake should be done directly from the Starmie repository we adapted to our pipeline (available [here](https://github.com/rcap107/starmie)). 

We refer to the readme in the linked repo for more detail on how to run Starmie in our environment. 

Once the Starmie model has been prepared, it is possible to wrap the query results prepared by Starmie in a format that
can be handled by the pipeline using the script `import_from_starmie.py`:

```py
python import_from_starmie.py BASE_PATH
```
where `BASE_PATH` is the path to the folder that contains the query results, e.g., `results/metadata/binary_update`. 
This will produce the query data structures that are used by the pipeline and allows to run a full batch of experiments
using Starmie query results. 

