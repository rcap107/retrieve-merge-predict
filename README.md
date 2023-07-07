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

# Downloading YADL
It is possible to download YADL from zenodo using `wget` in the root folder:
```
wget -O data/YADL_binary.tar.gz https://zenodo.org/record/8015298/files/YADL_binary.tar.gz
wget -O data/YADL_wordnet.tar.gz https://zenodo.org/record/8015298/files/YADL_wordnet.tar.gz
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
Running the indexing step takes about ~15 minutes on our cluster.

## Running the experiments
The sample script `example_config.sh` runs a single, shortened run: the results will not necessarily be accurate, but it
simplifies debugging.

To run the experiments reported in the paper, run the `./run_experiments.sh` script. Note that running all the experiments
can take a very long time.

## Studying the results
Results are stored in the `results/logs` folder.

`main_log.log` keeps track of each "high level execution" (as invoked
from the command line), mainly logging run-level parameters and timers.

`runs_log.log` keeps track of each singular fold (i.e. train/test split over the base table), and the main steps in the
training: results using only the base table, best result when joining a single candidate, join over all candidates and
join over the top `k` candidates.

Each outer fold will run the sequence "base table", "candidates", "full join", "sampled full join" on the same train/test
split.

Results were prepared using Google Sheets:
https://docs.google.com/spreadsheets/d/1a8YcpMxhr5MXkOLGepAZyDWcikySoL0zvqgWv1-Uv4c/edit?usp=sharing
