---
permalink: /docs/resources
layout: page
next_page: /docs/preparation
# previous_page: 
title: Downloading the data lakes 
---
All data lakes are available at `https://zenodo.org/doi/10.5281/zenodo.10600047`.

Archives provided here follow the notation used for the experiment configration, which is different from what is reported in the paper. The four YADL versions available here are:

- `binary_update` (YADL Binary)
- `wordnet_full` (YADL Base)
- `wordnet_vldb_10` (YADL 10k)
- `wordnet_vldb_50` (YADL 50k)

All YADL variants are synthesized from YAGO using the code in [YADL][YADL].

It is possible to download YADL from [the zenodo repository][zenodo_link] manually or by using `wget` in the root folder:
```sh
wget -O data/binary_update.tar.gz https://zenodo.org/records/10624396/files/binary_update.tar.gz
wget -O data/wordnet_full.tar.gz https://zenodo.org/records/10624396/files/wordnet_full.tar.gz
wget -O data/wordnet_vldb_10.tar.gz https://zenodo.org/records/10624396/files/wordnet_vldb_10.tar.gz
wget -O data/wordnet_vldb_50.tar.gz https://zenodo.org/records/10624396/files/wordnet_vldb_50.tar.gz
```

Once the archive has been downloaded, uncompress it to the `data` folder and execute the preparation step. 

[zenodo_link]: https://zenodo.org/doi/10.5281/zenodo.10600047
[YADL]: https://github.com/rcap107/YADL
[pipeline_repo]: https://github.com/rcap107/benchmark-join-suggestions