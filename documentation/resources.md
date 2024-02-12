---
permalink: /docs/resources
layout: page
---
**Code repositories**

The repository containing the pipeline and the code required to run the experiments 
can be found in the [pipeline repository][pipeline_repo].
The code for preparing the YADL variants can be found in the [preparation repository][prepare_repo]. 

**Base tables**

The base tables used for the experiments are available in the [pipeline repository][pipeline_repo], in folder `data/source_tables`.

**YADL**

It is possible to download YADL from [the zenodo repository][zenodo_link] manually or by using `wget` in the root folder:
```sh
wget -O data/binary_update.tar.gz https://zenodo.org/records/10600048/files/binary_update.tar.gz
wget -O data/wordnet_full.tar.gz https://zenodo.org/records/10600048/files/wordnet_full.tar.gz
```

[zenodo_link]: https://zenodo.org/doi/10.5281/zenodo.10600047
[prepare_repo]: https://github.com/rcap107/YADL
[pipeline_repo]: https://github.com/rcap107/benchmark-join-suggestions