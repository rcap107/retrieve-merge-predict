import datetime
import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Union

import datamart
import datamart_rest
import pandas as pd
import requests
from d3m import container
from d3m.container.utils import save_container
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="a",
    filename="data/logging.txt",
    level=logging.DEBUG,
)


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


def read_dataset_paths(dataset_list_path):
    valid_paths = []
    with open(dataset_list_path, "r") as fp:
        n_paths = int(fp.readline().strip())
        for idx, row in enumerate(fp):
            valid_paths.append(Path(row.strip()))
    stems = [pth.stem for pth in valid_paths]
    return valid_paths, stems


def fallback_download(dataset_id, dest_path, dataset_metadata):
    response = requests.get(
        f"https://auctus.vida-nyu.org/api/v1/download/{dataset_id}",
        files={"format": "d3m"},
    )
    response.raise_for_status()

    if response.status_code == 200:
        try:
            # The dummy is needed because zip needs some kind of file pointer to extract, which we don't have.
            dummy = io.BytesIO(response.content)
            zf = zipfile.ZipFile(dummy)
            zf.extractall(dest_path)
            dummy.close()
            json.dump(dataset_metadata, open(Path(dest_path, "metadata.json"), "w"))

            return dataset_id
        except Exception as e:
            # raise e
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
    dataset_list_path: Path,
    query_limit: int,
    query_timeout: Union[int, None],
    debug=False,
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

    _, list_ds_names = read_dataset_paths(dataset_list_path=dataset_list_path)

    data_path = Path("data/benchmark-datasets")

    list_datasets = []
    print("=" * 60)
    for idx, ds_name in enumerate(list_ds_names):
        print(f"{idx+1}/{len(list_ds_names)} - {ds_name}")
        target_dataset_learning_data = Path(
            data_path,
            f"{ds_name}",
            f"{ds_name}_dataset",
            Path("tables/learningData.csv"),
        )
        assert target_dataset_learning_data.exists()

        # Loading the D3M representation
        full_container = container.Dataset.load(
            target_dataset_learning_data.absolute().as_uri()
        )
        logging.info(f"Reading {ds_name}")
        ds_instance = Dataset(ds_name)
        try:
            # Probing Auctus with the full container
            cursor = client.search_with_data(query={}, supplied_data=full_container)
            # Fetching results
            results = cursor.get_next_page(limit=query_limit, timeout=query_timeout)
            results_by_dataset[ds_name] = results
            # Download each candidate in a different folder
            for id2, res in enumerate(results):
                res_mdata = res.get_json_metadata()
                res_id = res_mdata["id"]
                res_path = Path(ds_path, f"{ds_name}_candidates", res_id)

                logging.info(f"{ds_name}  - Downloading {res_id}")
                print(f"{res_id}", end="\r", flush=True)
                try:
                    res_dw = res.download(supplied_data=None)
                    # res_mdata = res_dw.to_json_structure()
                    try:
                        save_container(res_dw, res_path)
                        logging.debug(f"{ds_name} - {res_id} - Writing")
                    except FileExistsError:
                        shutil.rmtree(res_path)
                        save_container(res_dw, res_path)
                        logging.debug(f"{ds_name} - {res_id} - Overwriting")
                    finally:
                        print(f"{res_id} - Standard", end="\r", flush=True)
                        logging.info(f"{ds_name}  - {res_id} - Standard")
                except ValueError as ve:
                    # Some datasets raise exceptions, I force the download by using the REST API. It will be a problem for later.
                    print(f"{res_id} - Fallback", end="\r", flush=True)

                    if fallback_download(res_id, res_path, res_mdata) != res_id:
                        print("Fallback method failed. ")
                        ds_instance.add_failed(res_id)
                        logging.error(f"{ds_name}  - {res_id} - Failed with fallback")
                    else:
                        logging.info(f"{ds_name}  - {res_id} - Fallback")
                        logging.warning(f"{ds_name}  - {res_id} - Fallback")

                except Exception as ge:
                    print(
                        f"Uncaught exception for dataset {ds_name} and candidate {res_id}"
                    )
                    logging.error(f"{ds_name}  - {res_id} - {ge}")
                print()
        except Exception as e:
            # progress_overall.write(f"Server error for {ds_name}")
            failed_datasets.append(ds_instance)
            ds_instance.set_failed()
            logging.error(f"{ds_name}  - Failed querying")

        list_datasets.append(ds_instance)
        logging.info(f"{ds_name} - Complete")

        print("=" * 60)
    return results_by_dataset, list_datasets


def download_candidates(query_results: Dict):
    for dataset_name, ds_results in query_results.items():
        if len(ds_results) > 0:
            print(f"Dataset {dataset_name} does not have candidates. Skipping")
            continue

        for single_result in ds_results:
            pass
