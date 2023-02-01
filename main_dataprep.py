"""This script is used to fetch the join candidates for the given datasets 
"""

import argparse
from src.data_preparation.data_preparation import read_dataset_paths, query_datamart
from pathlib import Path

def parse_args(dummy=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_file", action="store", 
                        help="Path to the file that contains the list of datasets to query for.")
    parser.add_argument("--query_limit", action="store", default=5,  type=int,
                        help="Limit over the number of queried results.")
    parser.add_argument("--query_timeout", action="store",  default=None,
                        help="Timeout for the querying operation (by dataset).")
    parser.add_argument("--debug", action="store_true", 
                        help="If set, run only the first benchmark dataset rather than all those in the folder.")
    
    if dummy:
        args = parser.parse_args(args=[
        "data/target_datasets_small.txt",
        "--query_limit", "20"])
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args(dummy=True)
    
    query_results = query_datamart(Path(args.dataset_file), args.query_limit, args.query_timeout, debug=False)
