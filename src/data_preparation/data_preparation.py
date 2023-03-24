import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Union

import datamart_rest
import requests
from d3m import container
from d3m.container.utils import save_container

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="a",
    filename="data/logging.txt",
    level=logging.DEBUG,
)


REST_API_PATH = "https://auctus.vida-nyu.org/api/v1"


def build_dir_tree(dataset_list_path: Path, base_path=Path("data/benchmark-datasets/")):
    """Function for creating the directory tree for all target tables.

    Args:
        candidate_paths (Iterable[Path]): List of paths to candidate tables in the full repository.
    """
    os.makedirs(base_path, exist_ok=True)
    candidate_paths, _ = read_dataset_paths(dataset_list_path)
    destination_paths = []
    for pth in candidate_paths:
        ds_name = pth.stem
        src_dataset_path = Path("data") / pth
        dest_dataset_path = base_path / Path(ds_name)
        shutil.copytree(src_dataset_path, dest_dataset_path)
        os.makedirs(dest_dataset_path / Path(f"{ds_name}_candidates"), exist_ok=True)
        destination_paths.append(dest_dataset_path)

    return destination_paths


def read_dataset_paths(dataset_list_path: Path):
    valid_paths = []
    with open(dataset_list_path, "r") as fp:
        n_paths = int(fp.readline().strip())
        for idx, row in enumerate(fp):
            valid_paths.append(Path(row.strip()))
    stems = [pth.stem for pth in valid_paths]
    return valid_paths, stems


def fallback_download(dataset_id, dest_path, dataset_metadata):
    """Function for forcing the download of the dataset in case the high-level
    function raises an exception.

    Args:
        dataset_id (str): ID of the dataset to download.
        dest_path (Path): Destination path where to save the dataset.
        dataset_metadata (dict): Metadata of the dataset.

    Returns:
        Union[str, None]: If successful, returns `dataset_id`. If not, returns None.
    """
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
    def __init__(self, df_name, df_path) -> None:
        self.path = df_path
        self.name = df_name
        self.passed = True
        self.failed_candidates = []
        self.candidate_datasets = {}

    def set_failed(self):
        self.passed = False
        return

    def add_candidate(self, response_candidate):
        res_mdata = response_candidate.get_json_metadata()
        res_id = res_mdata["id"]
        res_aug = res_mdata["augmentation"]
        res_path = Path(self.path, f"{self.name}_candidates", res_id)

        if res_id not in self.candidate_datasets:
            self.candidate_datasets[res_id] = CandidateDataset(
                res_id, self.name, res_path
            )
            self.candidate_datasets[res_id].add_to_metadata(res_aug)
        else:
            self.candidate_datasets[res_id].add_to_metadata(res_aug)
        return res_id

    def add_failed(self, failed_id):
        self.failed_candidates.append(failed_id)

    def to_dict(self):
        return {
            "name": self.name,
            "passed": self.passed,
            "failed_candidates": self.failed_candidates,
            "candidate_datasets": {
                res_id: cand.to_dict()
                for res_id, cand in self.candidate_datasets.items()
            },
        }

    def save_to_json(self):
        json.dump(
            self.to_dict(),
            open(Path(self.path, f"{self.name}_candidates", f"queryResults.json"), "w"),
            indent=2,
        )


class CandidateDataset:
    def __init__(self, dataset_id, target_id, dataset_path, method=None):
        self.dataset_id = dataset_id
        self.target_id = target_id
        self.path = dataset_path
        self.metadata = []
        self.dl_mode = None

    def add_to_metadata(self, new_mdata):
        self.metadata.append(new_mdata)

    def set_dl_mode(self, dl_mode):
        self.dl_mode = dl_mode

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "target_id": self.target_id,
            "dl_mode": self.dl_mode,
            "metadata": self.metadata,
        }

    def save_to_json(self):
        json.dump(
            self.to_dict(),
            open(Path(self.path, f"{self.dataset_id}_metadata.json"), "w"),
            indent=2,
        )


def prepare_container(ds_name, ds_path):
    # ds_path = Path(download_folder_path, f"{ds_name}")
    if not ds_path.exists():
        logging.error(f"ERROR - {ds_name} - Dataset not found")
        return None

    else:
        # Loading the D3M representation
        full_container = container.Dataset.load(ds_path.absolute().as_uri())
        logging.info(f"INFO - Reading {ds_name}")
        return full_container


def query_single_dataset(
    client,
    ds_name,
    ds_path,
    container: container.Dataset,
    query_limit=20,
    query_timeout=60,
    only_metadata=False,
):
    ds_instance = Dataset(ds_name, ds_path)

    try:
        # Probing Auctus with the full container
        cursor = client.search_with_data(query={}, supplied_data=container)
        # Fetching results
        results = cursor.get_next_page(limit=query_limit, timeout=query_timeout)

        for id_res, res in enumerate(results):
            res_id = ds_instance.add_candidate(res)

            if not only_metadata:
                #TODO: debug whatever problem 
                raise NotImplementedError("Needs fixing")
                logging.info(f"INFO - {ds_name}  - Downloading {res_id}")
                print(f"{res_id}", end="\r", flush=True)
                try:
                    res_dw = res.download(supplied_data=None)
                    ds_instance.candidate_datasets[res_id].set_dl_mode("standard")
                    # res_mdata = res_dw.to_json_structure()
                    try:
                        save_container(res_dw, res_path)
                        logging.debug(f"DEBUG - {ds_name} - {res_id} - Writing")
                    except FileExistsError:
                        shutil.rmtree(res_path)
                        save_container(res_dw, res_path)
                        logging.debug(f"DEBUG - {ds_name} - {res_id} - Overwriting")
                    finally:
                        print(f"{res_id} - Standard", end="\r", flush=True)
                        logging.info(f"INFO - {ds_name}  - {res_id} - Standard")
                except ValueError as ve:
                    # Some datasets raise exceptions, I force the download by using the REST API. It will be a problem for later.
                    print(f"{res_id} - Fallback", end="\r", flush=True)

                    if fallback_download(res_id, res_path, res_mdata) != res_id:
                        print("Fallback method failed. ")
                        ds_instance.add_failed(res_id)
                        logging.error(
                            f"ERROR - {ds_name}  - {res_id} - Failed with fallback"
                        )
                    else:
                        ds_instance.candidate_datasets[res_id].set_dl_mode("fallback")
                        # logging.info(f"INFO - {ds_name}  - {res_id} - Fallback")
                        logging.warning(f"INFO - {ds_name}  - {res_id} - Fallback")

                except Exception as ge:
                    print(
                        f"Uncaught exception for dataset {ds_name} and candidate {res_id}"
                    )
                    logging.error(f"ERROR - {ds_name}  - {res_id} - {ge}")
                print()

        ds_instance.save_to_json()

        # for cand_id, candidate in ds_instance.candidate_datasets.items():
        #     candidate.save_to_json()

    except Exception as e:
        # progress_overall.write(f"Server error for {ds_name}")
        ds_instance.set_failed()
        logging.error(f"ERROR - {ds_name}  - Failed querying")
    return ds_instance


def query_datamart(
    dataset_list_path: Path,
    query_limit: int,
    query_timeout: Union[int, None],
    debug=False,
    data_folder=Path("data/benchmark-datasets"),
    only_metadata=True,
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

    list_ds_paths, list_ds_names = read_dataset_paths(
        dataset_list_path=dataset_list_path
    )

    download_folder_path = Path(data_folder)

    assert download_folder_path.exists()

    list_datasets = []
    list_failed_datasets = []
    print("=" * 60)
    for idx, (_, ds_name) in enumerate(zip(list_ds_paths, list_ds_names)):
        print(f"{idx+1}/{len(list_ds_names)} - {ds_name}")
        ds_path = Path(data_folder, ds_name)
        target_dataset_learning_data = Path(
            ds_path, f"{ds_name}_dataset", "tables", "learningData.csv"
        )
        full_container = prepare_container(ds_name, target_dataset_learning_data)
        if full_container is None:
            list_failed_datasets.append(ds_name)

        ds_instance = query_single_dataset(
            client, ds_name, ds_path, full_container, only_metadata=only_metadata,
            query_limit=query_limit, query_timeout=query_timeout
        )
        list_datasets.append(ds_instance)
        logging.info(f"INFO - {ds_name} - Complete")

        print("=" * 60)
    return results_by_dataset, list_datasets
