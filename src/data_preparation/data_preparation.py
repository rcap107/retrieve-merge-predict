from d3m import container
import datamart
import datamart_rest
import datetime
from pathlib import Path
import requests
import json
from tqdm import tqdm
import pandas as pd
from typing import Union, Iterable, Dict
import shutil
import os


REST_API_PATH = "https://auctus.vida-nyu.org/api/v1"

def build_dir_tree(candidate_paths: Iterable[Path]):
    """Function for creating the directory tree for all target tables. 

    Args:
        candidate_paths (Iterable[Path]): List of paths to candidate tables in the full repository.
    """
    base_path = Path("data/benchmark-datasets/")
    os.makedirs(base_path, exist_ok=True)
    destination_paths = []
    for pth in candidate_paths:
        ds_name = pth.stem
        src_dataset_path = Path("data")/pth
        dest_dataset_path = base_path/Path(ds_name)
        shutil.copytree(src_dataset_path, dest_dataset_path)
        os.makedirs(dest_dataset_path/Path(f"{ds_name}_candidates"), exist_ok=True)
        destination_paths.append(dest_dataset_path)

    return destination_paths

def reading_dataset_paths(VALID_PATH):
    valid_paths = []
    with open(VALID_PATH, "r") as fp:
        n_paths = int(fp.readline().strip())
        for idx, row in enumerate(fp):
            valid_paths.append(Path(row.strip()))
    return valid_paths


def query_datamart(dataset_paths: Path, query_limit: int, query_timeout: Union[int, None]):
    # All dataset results by dataset
    results_by_dataset = {}
    # Datasets for which querying fails (for any reason)
    failed_datasets = []

    # Connecting to the API
    client = datamart_rest.RESTDatamart(REST_API_PATH)

    for v_path in tqdm(dataset_paths):
        ds_name = v_path.stem
        target_dataset_learning_data = Path("data")/v_path/f"{ds_name}_dataset"/Path("tables/learningData.csv")
        assert target_dataset_learning_data.exists()

        # Loading the D3M representation
        full_container = container.Dataset.load(target_dataset_learning_data.absolute().as_uri()) 
        
        try:
            cursor = client.search_with_data(query={}, supplied_data=full_container)
            results = cursor.get_next_page(limit=query_limit, timeout=query_timeout)
            results_by_dataset[v_path] = results
        except Exception as e:    
            print(f"Server error for {v_path}")
            failed_datasets.append(v_path)
    
    return results_by_dataset

def download_candidates(query_results: Dict):
    for dataset_name, ds_results in query_results.items():
        if len(ds_results) > 0:
            print(f"Dataset {dataset_name} does not have candidates. Skipping")
            continue
        
        for single_result in ds_results:
            pass