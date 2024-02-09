---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
This repository contains the code for implementing and running the pipeline described in the paper "Retrieve, Merge, Predict: Augmenting Tables with Data Lakes
(Experiment, Analysis & Benchmark Paper).

## [Accessing the resources](docs/resources)
## [Installing the requirements](docs/installation)
## [Preparing the environment](docs/preparation)
## [Running the pipeline](docs/execution)
## [Dataset info](docs/datasets)
<!-- ## [Experimental results](docs/results) -->


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

The candidates produced by the join discovery methods are used to augment the base table, then the performance of the
joined tables is compared to that of the base table by training a regressor with Catboost and comparing the R2 score
measured before and after joining.

We use YADL as our data lake, a synthetic data lake based on the YAGO3 knowledge base. The YADL variants used in the paper
are available [on Zenodo](https://zenodo.org/doi/10.5281/zenodo.10600047).

The code for preparing the YADL variants can be found in [this repo](https://github.com/rcap107/prepare-data-lakes).

The base tables used for the experiments are provided in the repository 

**NOTE:** The repository relies heavily on the `parquet` format [[ref](https://parquet.apache.org/docs/file-format/)], and will expect all tables (both source tables, and data lake
tables) to be stored in `parquet` format. Please convert your data to parquet before working on the pipeline. 

