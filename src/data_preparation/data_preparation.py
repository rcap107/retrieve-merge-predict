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
from d3m.container.utils import save_container
import zipfile
import io

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
        src_dataset_path = Path("data") / pth
        dest_dataset_path = base_path / Path(ds_name)
        shutil.copytree(src_dataset_path, dest_dataset_path)
        os.makedirs(dest_dataset_path / Path(f"{ds_name}_candidates"), exist_ok=True)
        destination_paths.append(dest_dataset_path)

    return destination_paths


def reading_dataset_paths(VALID_PATH):
    valid_paths = []
    with open(VALID_PATH, "r") as fp:
        n_paths = int(fp.readline().strip())
        for idx, row in enumerate(fp):
            valid_paths.append(Path(row.strip()))
    return valid_paths


def fallback_download(dataset_id, dest_path, dataset_metadata):
    # response = requests.post(
    #     'https://auctus.vida-nyu.org/api/v1/search',
    #     files={
    #         'query': json.dumps({'keywords': dataset_id}).encode('utf-8'),
    #     },
    # )
    # response.raise_for_status()
    # for result in response.json()['results']:
    #     print(result['score'], result['name'], result['id'])

    response = requests.get(
        f"https://auctus.vida-nyu.org/api/v1/download/{dataset_id}",
        files={"format": "d3m"},
    )
    response.raise_for_status()

    if response.status_code == 200:
        try:
            dummy = io.BytesIO(response.content)
            zf = zipfile.ZipFile(dummy)
            zf.extractall(dest_path)
            dummy.close()
            json.dump(dataset_metadata, open(Path(dest_path, "metadata.json"), "w"))

            return dataset_id
        except Exception as e:
            raise e
            return None
    elif response.status_code == 404:
        return None


class Dataset:
    def __init__(self, df_id) -> None:
        self.id = df_id
        self.passed = True
        self.failed_candidates = []

    def set_failed(self):
        self.passed = False
        return

    def add_failed(self, failed_id):
        self.failed_candidates.append(failed_id)

    def to_dict(self):
        return {
            "id": self.id,
            "passed": self.passed,
            "failed_candidates": self.failed_candidates,
        }


def query_datamart(
    dataset_paths: Path, query_limit: int, query_timeout: Union[int, None], debug=False
):

    if debug:
        limit = 1
    else:
        limit = -1

    # All dataset results by dataset
    results_by_dataset = {}
    # Datasets for which querying fails (for any reason)
    failed_datasets = []

    # Connecting to the API
    client = datamart_rest.RESTDatamart(REST_API_PATH)

    data_path = Path("data/benchmark-datasets")
    datasets_to_check = os.listdir(data_path)[:limit]

    list_datasets = []
    for ds_name in tqdm(
        datasets_to_check, total=len(datasets_to_check), position=0, leave=False
    ):
        ds_path = Path(data_path, f"{ds_name}")
        target_dataset_learning_data = Path(
            ds_path, f"{ds_name}_dataset", Path("tables/learningData.csv")
        )
        assert target_dataset_learning_data.exists()

        # Loading the D3M representation
        full_container = container.Dataset.load(
            target_dataset_learning_data.absolute().as_uri()
        )

        ds_instance = Dataset(ds_name)
        try:
            # Probing Auctus with the full container
            cursor = client.search_with_data(query={}, supplied_data=full_container)
            # Fetching results
            results = cursor.get_next_page(limit=query_limit, timeout=query_timeout)
            results_by_dataset[ds_name] = results
            # Download each candidate in a different folder
            for res in tqdm(results, total=len(results), position=1, leave=False):
                res_mdata = res.get_json_metadata()
                res_id = res_mdata["id"]
                print(res_id)
                res_path = Path(ds_path, f"{ds_name}_candidates", res_id)
                try:
                    res_dw = res.download(supplied_data=None)
                    # res_mdata = res_dw.to_json_structure()
                    try:
                        save_container(res_dw, res_path)
                    except FileExistsError:
                        shutil.rmtree(res_path)
                        save_container(res_dw, res_path)
                except ValueError as ve:
                    # Some datasets raise exceptions, I force the download by using the REST API. It will be a problem for later. 
                    print("Downloader failed: using fallback method.")
                    if fallback_download(res_id, res_path, res_mdata) != res_id:
                        print("Fallback method failed. ")
                        ds_instance.add_failed(res_id)

        except Exception as e:
            # progress_overall.write(f"Server error for {ds_name}")
            failed_datasets.append(ds_instance)
            ds_instance.set_failed()
        list_datasets.append(ds_instance)
    return results_by_dataset, list_datasets


def download_candidates(query_results: Dict):
    for dataset_name, ds_results in query_results.items():
        if len(ds_results) > 0:
            print(f"Dataset {dataset_name} does not have candidates. Skipping")
            continue

        for single_result in ds_results:
            pass
