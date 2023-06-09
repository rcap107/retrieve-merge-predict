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

# Running the pipeline
To run the experiments reported in the paper, run the `./run_experiments.sh` script. 

# Dir structure



## Logging

# How to execute

## Preparing the metadata

## Running the indexing

## Testing the performance

