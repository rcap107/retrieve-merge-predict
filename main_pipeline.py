# %%
# %load_ext autoreload
# %autoreload 2
import json
import pickle
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.candidate_discovery.utils_lazo import LazoIndex
from src.candidate_discovery.utils_minhash import MinHashIndex
from src.data_preparation.data_structures import CandidateJoin, RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins
import src.evaluation.evaluation_methods as em

def prepare_metadata(data_dir, source_data_lake, mdata_out_dir, mdata_index_fname):
    for dataset_path in data_dir.glob("**/*.parquet"):
        ds = RawDataset(dataset_path, source_data_lake, mdata_out_dir)
        ds.save_metadata_to_json()

    metadata_index = MetadataIndex(mdata_out_dir)
    metadata_index.save_index(mdata_index_fname)

    return metadata_index


def prepare_default_configs(data_dir, selected_indices=None):
    """Prepare default configurations for various indexing methods and provide the
    data directory that contains the metadata of the tables to be indexed.

    Args:
        data_dir (str): Path to the directory that contains the metadata.
        selected_indices (str): If provided, prepare and run only the selected indices. 

    Raises:
        IOError: Raise IOError if `data_dir` is incorrect.

    Returns:
        dict: Configuration dictionary
    """
    if Path(data_dir).exists():
        configs = {
            "lazo": {
                "data_dir": data_dir,
                "partition_size": 50_000,
                "host": "localhost",
                "port": 15449,
            },
            "minhash": {
                "data_dir": data_dir,
                "thresholds": [20, 40, 80],
                "oneshot": True,
            },
        }
        if selected_indices is not None:
            return {
                index_name: config
                for index_name, config in configs.items()
                if index_name in selected_indices
            }
        else:
            return configs
    else:
        raise IOError(f"Invalid path {data_dir}")


def prepare_indices(index_configurations: dict):
    """Given a dict of index configurations, initialize the required indices.

    Args:
        index_configurations (dict): Dictionary that contains the required configurations.

    Raises:
        NotImplementedError: Raise NotImplementedError when providing an index that is not recognized.

    Returns:
        dict: Dictionary that contains the initialized indices.
    """
    index_dict = {}
    for index, config in index_configurations.items():
        if index == "lazo":
            index_dict[index] = LazoIndex(**config)
        elif index == "minhash":
            index_dict[index] = MinHashIndex(**config)
        else:
            raise NotImplementedError

    return index_dict


def save_indices(index_dict, index_dir):
    """Save all the indices found in `index_dict` in separate pickle files, in the
    directory provided in `index_dir`.

    Args:
        index_dict (dict): Dictionary containing the indices.
        index_dir (str): Path where the dictionaries will be saved.
    """
    if Path(index_dir).exists():
        for index_name, index in index_dict.items():
            print(f"Saving index {index_name}")
            filename = f"{index_name}_index.pickle"
            fpath = Path(index_dir, filename)
            index.save_index(fpath)
    else:
        raise ValueError(f"Invalid `index_dir` {index_dir}")


def load_indices(index_dir):
    """Given `index_dir`, scan the directory and load all the indices in an index dictionary.

    Args:
        index_dir (str): Path to the directory containing the indices.

    Raises:
        IOError: Raise IOError if the index dir does not exist.

    Returns:
        dict: The `index_dict` containing the loaded indices.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise IOError(f"Index dir {index_dir} does not exist.")
    index_dict = {}

    for index_path in index_dir.glob("**/*.pickle"):
        with open(index_path, "rb") as fp:
            input_dict = pickle.load(fp)
            iname = input_dict["index_name"]
            if iname == "minhash":
                index = MinHashIndex()
                index.load_index(index_dict=input_dict)
            elif iname == "lazo":
                index = LazoIndex()
                index.load_index(index_path)
            else:
                raise ValueError(f"Unknown index {iname}.")
                
            index_dict[iname] = index

    return index_dict


def generate_candidates(
    index_name: str,
    index_result: list,
    metadata_index: MetadataIndex,
    mdata_source: dict,
    query_column: str,
):
    """Given the index results in `index_result`, generate a candidate join for each of them. The candidate join will
    not execute the join operation: it holds the information (path, join columns) necessary for it.

    This information is extracted using `metadata_index` for the join columns and `mdata_source` for the source
    table. `query_column` is the join column in the source table.

    Args:
        index_name (str): Name of the index that generated the candidates.
        index_result (list): List of potential join candidates as they were provided by an index.
        metadata_index (MetadataIndex): Metadata Index that holds the metadata of all tables in the data lake.
        mdata_source (dict): Metadata of the source table.
        query_column (str): Label of the query column.

    Returns:
        dict: Dictionary containing the candidates, the candidate id is the index.
    """
    candidates = {}
    for res in index_result:
        hash_, column, similarity = res
        mdata_cand = metadata_index.query_by_hash(hash_)
        cjoin = CandidateJoin(
            indexing_method=index_name,
            source_table_metadata=mdata_source.info,
            candidate_table_metadata=mdata_cand,
            how="left",
            left_on=query_column,
            right_on=column,
            similarity_score=similarity,
        )
        candidates[cjoin.candidate_id] = cjoin
    return candidates


def querying(
    mdata_source: dict,
    source_column: str,
    query: list,
    indices: dict,
    mdata_index: MetadataIndex,
):
    """Query all indices for the given values in `query`, then generate the join candidates.

    Args:
        mdata_source (dict): Metadata of the source table.
        source_column (str): Column that is the target of the query.
        query (list): List of values in the query column.
        indices (dict): Dictionary containing all the indices.
        mdata_index (MetadataIndex): Metadata Index that contains information on tables in the data lake.

    Returns:
        (dict, dict): Returns the result of the queries on each index and the candidate joins for all results.
    """
    query_results = {}
    for index_name, index in indices.items():
        print(f"Querying index {index_name}.")
        index_res = index.query_index(query)
        query_results[index.index_name] = index_res

    candidates_by_index = {}
    for index, index_res in query_results.items():
        candidates = generate_candidates(index,
            index_res, mdata_index, mdata_source, source_column
        )
        candidates_by_index[index] = candidates

    return query_results, candidates_by_index


def evaluate_joins(source_table, join_candidates: dict, num_features=None, verbose=1):
    result_list = []
    source_table, num_features, cat_features = em.prepare_table_for_evaluation(source_table, num_features)

    # Run on source table alone
    print("Running on base table.")
    results = em.run_on_table(source_table, num_features, cat_features, verbose=verbose)
    rmse = em.measure_rmse(results["y_test"], results["y_pred"])
    r2 = em.measure_rmse(results["y_test"], results["y_pred"])
    # TODO add proper name + hash
    result_list.append(("base", "base", rmse, r2))

    # Run on all candidates
    print("Running on candidates, one at a time.")
    for index_name, index_cand in join_candidates.items():
        rmse_dict = em.execute_on_candidates(index_cand, source_table, num_features, cat_features, verbose=verbose)
        for k,v in rmse_dict.items():
            result_list.append((index_name, k, v))

    # Execute full join by index, then evaluate
    # print("Running on the fully-joined table. ")
    # for index_name, index_cand in join_candidates.items():
    #     merged_table = em.execute_full_join(index_cand, source_table, num_features, cat_features)
    #     cat_features = [col for col in merged_table.columns if col not in num_features]
    #     merged_table = merged_table.fill_null("")
    #     merged_table = merged_table.fill_nan("")
    #     results = em.run_on_table(merged_table, num_features, cat_features, verbose=verbose)
    #     rmse = em.measure_rmse(results["y_test"], results["y_pred"])
    #     result_list.append((index_name, "full_merge", rmse))    

    return pl.from_records(result_list, orient="row").to_pandas()

# %%
if __name__ == "__main__":
    source_data_lake = "yago3-dl"
    metadata_dir = "data/metadata/binary"
    mdata_index_path = Path("data/metadata/mdi/md_index_binary.pickle")

    index_dir = "data/metadata/indices"

    precomputed_indices = True  # if true, load indices from disk

    print(f"Reading metadata from {mdata_index_path}")
    if not mdata_index_path.exists():
        raise FileNotFoundError(f"Path to metadata index {mdata_index_path} is invalid.")
    else:
        mdata_index = MetadataIndex(index_path=mdata_index_path)

    selected_indices = ["minhash", "lazo"]

    if not precomputed_indices:
        index_configurations = prepare_default_configs(metadata_dir, selected_indices)
        print("Preparing indices.")
        indices = prepare_indices(index_configurations)
        print("Saving indices.")
        save_indices(indices, index_dir)
    else:
        print("Loading indices.")
        indices = load_indices(index_dir)

    # Query index
    print("Querying.")
    query_data_path = Path("data/source_tables/ken_datasets/presidential-results")
    tab_name = "presidential-results-prepared"
    tab_path = Path(query_data_path, f"{tab_name}.parquet")
    df = pl.read_parquet(tab_path)
    
    print(f"Querying from dataset {tab_path}")
    query_metadata = RawDataset(tab_path.resolve(), "queries", "data/metadata/queries")
    query_metadata.save_metadata_to_json()
    
    query_column = "col_to_embed"
    # query = df[query_column].sample(3000).drop_nulls()
    query = df[query_column].drop_nulls()

    query_results, candidates_by_index = querying(
        query_metadata, query_column, query, indices, mdata_index
    )

    with open(f"generated_candidates_{tab_name}.pickle", "wb") as fp:
        pickle.dump(candidates_by_index, fp)

    print("Profiling results.")
    profiling_results = profile_joins(candidates_by_index)
    profiling_results.to_csv("results/profiling_results.csv", index=False, mode="a")
    
    print("Evaluating join results.")
    # num_features = [
    #     "budget", 
    #     "popularity",
    #     "release_date", 
    #     "runtime",
    #     "vote_average",
    #     "vote_count",
    #     "target"
    # ]

    # TODO: Somewhere in here there still are issues with types 
    results = evaluate_joins(df, join_candidates=candidates_by_index, num_features=None, verbose=0)
    results.to_csv("results/run_results.csv", mode="a")


# # %%

# # %%
# # Load prepared indices
# mh_index = pickle.load(open("mh_index.pickle", "rb"))
# metadata_index = MetadataIndex(index_path="debug_metadata_index.pickle")
# # %%
# # Query index
# data_path = Path("data/yago3-dl/wordnet")
# metadata_path = Path("data/metadata/debug")

# tab_name = "yago_seltab_wordnet_movie"
# tab_path = Path(data_path, f"{tab_name}.parquet")
# df = pl.read_parquet(tab_path)
# query_column = "isLocatedIn"
# query = df[query_column].sample(50000).drop_nulls()

# mh_result = mh_index.query_index(query)

# # %%
# mdata_source = RawDataset(tab_path, "yago3-dl", metadata_path)


# # %%

# %%
