---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
This repository contains the code for implementing and running the pipeline described in the paper "Retrieve, Merge, Predict: Augmenting Tables with Data Lakes
(Experiment, Analysis & Benchmark Paper).

- TOC
{:toc}

The objective is modeling a situation where an user is trying to execute ML tasks on some base data, enriching it by
using new tables found in a data lake using retrieval methods.

The join candidates are merged with the base table under study, before training a ML model (either Catboost, or a linear model) to evaluate the performance
before and after the merge.

We use YADL as our data lake, a synthetic data lake based on the YAGO3 knowledge base. The YADL variants used in the paper
are available [on Zenodo](https://zenodo.org/doi/10.5281/zenodo.10600047).

The code for preparing the YADL variants can be found in [this repo](https://github.com/rcap107/prepare-data-lakes).

The base tables used for the experiments are provided in the repository 

**NOTE:** The repository relies heavily on the `parquet` format [ref](https://parquet.apache.org/docs/file-format/), and will expect all tables (both source tables, and data lake
tables) to be stored in `parquet` format. Please convert your data to parquet before working on the pipeline. 

**NOTE:** We recommend to use the smaller `binary_update` data lake and its corresponding configurations to set up the data structures and debug potential issues, as all preparation steps are significantly faster than with larger data lakes. 


## A simple example

![alice-example](/assets/img/alice-example.drawio.png)

Alice is working on a table that contains information about movies. She has also access to a data lake, or a collection 
of other tables on all sorts of subjects. 

She is looking for a way to predict the box office revenue of a movie based on as much information as possible, so she
would like to leverage the information stored in the data to improve the performance of her model. 

The problem is that, while the information is indeed available, it is mixed with a huge amount of unrelated data. Alice's
problem is thus figuring out how to find those tables that are actually relevant, and how to join them with her starting
table. 

This toy example was our starting point for creating the following pipeline, where we illustrate the various step that 
Alice may need in order to predict the revenue. 

![pipeline](/assets/img/benchmark-pipeline-v6.png)

The objective is modeling a situation where an user is trying to execute ML tasks on some base data, enriching it by
using new tables found in a data lake through Join Discovery methods.

The candidates produced by the join discovery methods are used to augment the base table, then the performance of the
joined tables is compared to that of the base table by training a regressor with Catboost and comparing the R2 score
measured before and after joining.

We use YADL as our data lake, a synthetic data lake based on the YAGO3 knowledge base. 

<!-- [Resources]({% post_url 2023-07-05-resources %}) -->

# Accessing the resources
## Code repositories
The repository containing the pipeline and the code required to run the experiments 
can be found in the [pipeline repository][pipeline_repo].
The code for preparing the YADL variants can be found in the [preparation repository][prepare_repo]. 

## Base tables
The base tables used for the experiments are available in the [pipeline repository][pipeline_repo], in folder `data/source_tables`.

## YADL
It is possible to download YADL from [the zenodo repository][zenodo_link] manually or by using `wget` in the root folder:
```sh
wget -O data/binary_update.tar.gz https://zenodo.org/records/10600048/files/binary_update.tar.gz
wget -O data/wordnet_full.tar.gz https://zenodo.org/records/10600048/files/wordnet_full.tar.gz
```

# Installing the requirements
We recommend to use conda environments to fetch the required packages. File `environment.yaml` contains the
required dependencies and allows to create directly a conda environment:
```
conda env create --file=environment.yaml
conda activate bench
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

**NOTE:** This scripts assumes that all tables are saved in `.parquet` format, and will raise an error if it finds no `.parquet`
files in the given path. Please convert your files to parquet before running this script. 

Use the command:
```
python prepare_metadata.py PATH_DATA_FOLDER
```
where `PATH_DATA_FOLDER` is the root path of the data lake. The stem of `PATH_DATA_FOLDER` will be used as identifier for 
the data lake throughout the program (e.g., for `data/binary_update`, the data lake will be stored under the name `binary_update`).

The script will recursively scan all folders found in `PATH_DATA_FOLDER` and generate a json file for each parquet file
encountered. By providing the `--flat` parameter, it is possible to scan only the files in the root directory rather than 
working on all folders and files. 

Metadata will be saved in `data/metadata/DATA_LAKE_NAME`, with an auxiliary file stored in `data/metadata/_mdi/md_index_DATA_LAKE_NAME.pickle`.

## Preparing the Retrieval methods
This step is an offline operation during which the retrieval methods are prepared by building the data structures they rely on to function. This operation can require a long time and a large amount of disk space (depending on the method); it is not required for 
the querying step and thus it can be executed only once for each data lake (and retrieval method).

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

The configuration parser will prepare the data structures (specifically, the counts) for each case provided in the configuration file.

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

# Executing the pipeline
The configurations used to run the experiments in the paper are available in directory `config/evaluation`. 

The experiment configurations that tested default parameters are stored in `config/evaluation/general`; experiment configurations 
testing aggregation are in `config/evaluation/aggregation`; additional experiments that test specific parameters and scenarios are in `config/evaluation/other`.

To run experiments with `binary_update`:
```sh
python main.py config/evaluation/general/config-binary.toml
```

[zenodo_link]: https://zenodo.org/doi/10.5281/zenodo.10600047
[prepare_repo]: https://github.com/rcap107/YADL
[pipeline_repo]: https://github.com/rcap107/benchmark-join-suggestions