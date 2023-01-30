"""This script is used to fetch the join candidates for the given datasets 
"""

import argparse
from src.data_preparation import reading_dataset_paths, query_datamart
from pathlib import Path

def parse_args(dummy=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_file", action="store", 
                        help="Path to the file that contains the list of datasets to query for.")
    parser.add_argument("--query_limit", action="store", default=5,  type=int,
                        help="Limit over the number of queried results.")
    parser.add_argument("--query_timeout", action="store",  default=None,
                        help="Timeout for the querying operation (by dataset).")
    # parser.add_argument("dataset_file", action="store", 
    #                     help="Path to the file that contains the list of datasets to query for.")
    
    if dummy:
        args = parser.parse_args(args=[
        "data/target_datasets_small.txt",
        "--query_limit", "5"])
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args(dummy=True)

    dataset_list = reading_dataset_paths(Path(args.dataset_file))
    
    query_results = query_datamart(dataset_list, args.query_limit, args.query_timeout)
    