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


# Running the pipeline
Before running the pipeline, it is necessary to set up the indices and the metadata of the tables in the data lake. 

Extract the variants to folder `data/yadl/`, then run the script `prepare_metadata.py`.
```
python prepare_metadata.py -s CASE PATH
```
`CASE` is the tag to be given to the index (e.g., `binary` or `wordnet`). 

`PATH` is the path to the root folder containing all the tables (saved in parquet) to be added to the metadata index and 
to the indices.

The actual pipeline is run on a single `CASE` at a time. 

To run the experiments reported in the paper, run the `./run_experiments.sh` script. 
