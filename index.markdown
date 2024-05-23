---
layout: home
# title: Retrieve, Merge, Predict
---
This repository contains the code for implementing and running the pipeline described in the paper "Retrieve, Merge, Predict: Augmenting Tables with Data Lakes".

**Code repositories**
The repository containing the pipeline and the code required to run the experiments 
can be found in the [pipeline repository][pipeline_repo].
The code for preparing the YADL variants can be found in the [preparation repository][prepare_repo]. 

Retrieve the code either by cloning the main repository:
```
git clone https://github.com/rcap107/retrieve-merge-predict.git
```
or by downloading the latest release from [tags](https://github.com/rcap107/retrieve-merge-predict/tags). 

**Base tables**
The base tables used for the experiments are available in the [pipeline repository][pipeline_repo], in folder `data/source_tables`. More information is reported in [Dataset info](docs/datasets). 

**Data lakes**
The data lakes used for our experiments are stored on Zenodo. Follow the instructions in [Downloading the data lakes](docs/resources) to prepare them. 

**Requirements** 
We recommend to use conda environments to fetch the required packages. File `environment.yaml` contains the
required dependencies and allows to create directly a conda environment:
```
conda env create --file=environment.yaml
conda activate bench
```
Then, from within the `bench` environment install the remaining dependencies with pip:
```
pip install -r requirements.txt
```

**Preparation of the testing environment**
Running the experiments requires some preparation to build the data structures that are required by the retrieval step. This is reported in [Preparing the environment](docs/preparation). 

**Pipeline execution**
The page [Running the pipeline](docs/execution) contains the information required for preparing configurations, running the pipeline, recovering from a pipeline crash, and exploring the profiling results. 

Make sure that the preparation has been run properly, since the pipeline is relying on data structures that are assumed to have been build during the previous step. 


<!-- ## [Experimental results](docs/results) -->


## A simple example

![pipeline](/assets/img/benchmark-pipeline-v6.png)
<!-- ![alice-example](/assets/img/alice-example.drawio.png) -->

Alice is working on a table that contains information about movies. She has also access to a data lake, or a collection 
of other tables on all sorts of subjects. 

She is looking for a way to predict the box office revenue of a movie based on as much information as possible, so she
would like to leverage the information stored in the data to improve the performance of her model. 

The problem is that, while the information is indeed available, it is mixed with a huge amount of unrelated data. Alice's
problem is thus figuring out how to find those tables that are actually relevant, and how to join them with her starting
table. 

This toy example was our starting point for creating the our pipeline, where we illustrate the various step that 
Alice may need in order to predict the revenue. 

The candidates produced by the join discovery methods are used to augment the base table, then the performance of the
joined tables is compared to that of the base table by training a regressor with Catboost and comparing the R2 score
measured before and after joining.

We use YADL as our data lake, a synthetic data lake based on the YAGO3 knowledge base. The YADL variants used in the paper
are available [on Zenodo](https://zenodo.org/doi/10.5281/zenodo.10600047).

The code for preparing the YADL variants can be found in [this repo](https://github.com/rcap107/YADL).

The base tables used for the experiments are provided in the repository 

**NOTE:** The repository relies heavily on the `parquet` format [[ref](https://parquet.apache.org/docs/file-format/)], and will expect all tables (both source tables, and data lake
tables) to be stored in `parquet` format. Please convert your data to parquet before working on the pipeline. 


[zenodo_link]: https://zenodo.org/doi/10.5281/zenodo.10600047
[prepare_repo]: https://github.com/rcap107/YADL
[pipeline_repo]: https://github.com/rcap107/benchmark-join-suggestions