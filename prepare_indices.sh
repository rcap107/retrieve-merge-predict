#!/bin/bash

# MINHASH
# python prepare_indices.py -s binary data/binary --selected_indices minhash
# python prepare_indices.py -s binary data/wordnet_big --selected_indices minhash
# python prepare_indices.py -s binary data/wordnet_big_num_cp --selected_indices minhash

# MANUAL
## binary
for tab in `ls data/source_tables`;
do
echo $tab;
python prepare_metadata.py -s binary data/binary --selected_indices manual --base_table data/source_tables/$tab --n_jobs 8
done

## wordnet_big
for tab in `ls data/source_tables`;
do
echo $tab;
python prepare_metadata.py -s wordnet_big data/wordnet_big --selected_indices manual --base_table data/source_tables/$tab --n_jobs 8
done

## wordnet_big_num_cp
for tab in `ls data/source_tables`;
do
echo $tab;
python prepare_metadata.py -s wordnet_big_num_cp data/wordnet_big_num_cp --selected_indices manual --base_table data/source_tables/$tab --n_jobs 8
done
