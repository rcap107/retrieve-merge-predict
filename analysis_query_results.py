import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from joblib import load

from src.utils.joining import execute_join_with_aggregation

DEFAULT_QUERY_RESULT_DIR = Path("results/query_results")


def load_query_result(yadl_version, index_name, tab_name, query_column, top_k):
    query_result_path = "{}__{}__{}__{}.pickle".format(
        yadl_version,
        index_name,
        tab_name,
        query_column,
    )

    with open(Path(DEFAULT_QUERY_RESULT_DIR, query_result_path), "rb") as fp:
        query_result = pickle.load(fp)

    query_result.select_top_k(top_k)
    return query_result


def load_exact_matching(data_lake_version, table_name, column_name):
    path = Path("data/metadata/_indices", data_lake_version)
    iname = f"em_index_{table_name}_{column_name}.pickle"
    with open(Path(path, iname), "rb") as fp:
        d = load(fp)
    counts = d["counts"]
    return counts


def test_joining(
    data_lake_version, index_name, table_name, query_column, top_k, aggregation
):
    query_result = load_query_result(
        data_lake_version, index_name, table_name, query_column, 0
    )
    base_table = pl.read_parquet(f"data/source_tables/yadl/{table_name}.parquet")
    df_counts = load_exact_matching(
        data_lake_version=data_lake_version,
        table_name=table_name,
        column_name=query_column,
    )
    query_result.select_top_k(top_k)
    total_time = 0
    list_stats = []
    base_results = {
        "retrieval_method": index_name,
        "data_lake_version": data_lake_version,
        "table_name": table_name,
        "query_column": query_column,
        "aggregation": aggregation,
        "top_k": "",
        "rank": "",
        "cnd_table": "",
        "cnd_column": "",
        "containment": "",
        "src_nrows": "",
        "src_ncols": "",
        "cnd_nrows": "",
        "cnd_ncols": "",
        "join_time": "",
    }
    for rank, (c_id, cand) in enumerate(query_result.candidates.items()):
        r_dict = dict(base_results)
        _, cnd_md, left_on, right_on = cand.get_join_information()
        cand_table = pl.read_parquet(cnd_md["full_path"])

        cont = df_counts.filter(
            (pl.col("hash") == cnd_md["hash"]) & (pl.col("col") == right_on)
        )["containment"].item()

        start_time = datetime.datetime.now()
        merge = execute_join_with_aggregation(
            base_table,
            cand_table,
            left_on=left_on,
            right_on=right_on,
            how="left",
            aggregation=aggregation,
        )
        end_time = datetime.datetime.now()
        time_required = (end_time - start_time).total_seconds()
        total_time += time_required

        r_dict["cnd_table"] = cnd_md["hash"]
        r_dict["cnd_column"] = right_on[0]
        r_dict["containment"] = cont
        r_dict["src_nrows"], r_dict["src_ncols"] = base_table.shape
        r_dict["cnd_nrows"], r_dict["cnd_ncols"] = cand_table.shape
        r_dict["join_time"] = time_required
        r_dict["top_k"] = top_k
        r_dict["rank"] = rank

        list_stats.append(r_dict)
    print(f"{data_lake_version} {table_name} {aggregation} {top_k} {total_time:.2f}")
    return list_stats


if __name__ == "__main__":
    data_lake_version = "wordnet_full"
    query_column = "col_to_embed"
    top_k = 0
    index_names = ["exact_matching", "minhash", "minhash_hybrid"]
    keys = ["index_name", "tab_name", "top_k", "join_time", "avg_cont"]
    results = []
    tabs = [
        "company_employees",
        "housing_prices",
        "movies_vote",
        "movies",
        "us_accidents",
        "us_county_population",
        "us_elections",
    ]

    for tab in tabs:
        tab_name = f"{tab}-yadl-depleted"
        for i in index_names:
            print(i)
            for k in [30, 200]:
                for aggr in ["first", "mean"]:
                    this_res = test_joining(
                        data_lake_version,
                        i,
                        tab_name,
                        query_column,
                        k,
                        aggregation=aggr,
                    )
                    results += this_res
    df = pl.from_dicts(results)
    df.write_csv("analysis_query_results.csv")
