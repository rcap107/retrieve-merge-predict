---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
Benchmarking learning on data lakes with YADL
===
This repository contains the code for implementing and running the pipeline described in the paper "A Benchmarking Data
Lake for Join Discovery and Learning with Relational Data".

- TOC
{:toc}


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

![pipeline](/assets/img/benchmark-pipeline-v5.png)

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
can be found in the [pipeline repository](pipeline_repo).
The code for preparing the YADL variants can be found in the [preparation repository][prepare_repo]. 

## Base tables
The base tables used for the experiments are available in the [pipeline repository](pipeline_repo), in folder `data/source_tables`.

## YADL
The YADL variants used in the paper are [available on Zenodo][zenodo_link]. 

<!-- ## Precomputed indices -->

# Installing the requirements
We strongly recommend to use conda environments to fetch the required packages. File `environment.yaml` contains the
required dependencies and allows to create directly a conda environment:
```sh
conda env create --file=environment.yaml
conda activate bench-repro
```
Then, install the remaining dependencies with pip:
```
pip install -r requirements.txt
```

# Running the pipeline
## Creating the indices
Before running the pipeline, it is necessary to set up the indices and the metadata of the tables in the data lake.

Extract the variants to folder `data/yadl/`, then run the script `prepare_metadata.py`.
```
python prepare_metadata.py -s CASE PATH
```
`CASE` is the tag to be given to the index (e.g., `binary` or `wordnet`).

`PATH` is the path to the root folder containing all the tables (saved in parquet) to be added to the metadata index and
to the indices.

```
# wordnet case
python prepare_metadata.py -s wordnet data/wordnet_big/
# binary case
python prepare_metadata.py -s binary data/binary/
```
Running the indexing step on the given tables takes about ~15 minutes on our cluster.

## Running the experiments
The sample script `example_config.sh` runs a single, shortened run: the results will not necessarily be accurate, but it
simplifies debugging.

To run the experiments reported in the paper, run the `./run_experiments.sh` script. Note that running all the experiments
can take a very long time.

## Studying the results
Results are stored in the `results/logs` folder.

`main_log.log` logs one line for each execution of the `main_pipeline.py` script, mainly for logging run-level 
parameters and timers.

`runs_log.log` keeps track of each singular fold (i.e. train/test split over the base table), and the main steps in the
training: results using only the base table, best result when joining a single candidate, join over all candidates and
join over the top `k` candidates.

Each outer fold will run the sequence "base table", "candidates", "full join", "sampled full join" on the same train/test
split.

The run results are available in 
[this spreadsheet](https://docs.google.com/spreadsheets/d/1a8YcpMxhr5MXkOLGepAZyDWcikySoL0zvqgWv1-Uv4c/edit?usp=sharing)

[zenodo_link]: https://zenodo.org/record/8015298
[prepare_repo]: https://github.com/rcap107/prepare-data-lakes
[pipeline_repo]: https://github.com/rcap107/benchmark-join-suggestions