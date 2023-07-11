#!/bin/bash

# MINHASH
# python prepare_indices.py -s binary data/binary --selected_indices minhash
# python prepare_indices.py -s binary data/wordnet_big --selected_indices minhash
# python prepare_indices.py -s binary data/wordnet_big_num_cp --selected_indices minhash

# MANUAL
for tab in `ls data/source_tables`;
do
echo $tab;
python prepare_metadata.py -s binary data/binary --selected_indices manual --base_table data/source_tables/$tab --n_jobs 8
break
done
