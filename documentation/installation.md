---
permalink: /docs/installation
layout: page
---
**Installation**

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
