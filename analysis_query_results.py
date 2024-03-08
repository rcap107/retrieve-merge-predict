import datetime
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from joblib import load
from tqdm import tqdm

from src.utils.joining import execute_join_with_aggregation

DEFAULT_QUERY_RESULT_DIR = Path("results/query_results")


def load_query_result(yadl_version, index_name, tab_name, query_column, top_k):
    query_result_path = "{}__{}__{}__{}.pickle".format(
        yadl_version,
        index_name,
        tab_name,
        query_column,
    )

    with open(
        Path(DEFAULT_QUERY_RESULT_DIR, yadl_version, query_result_path), "rb"
    ) as fp:
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
    data_lake_version,
    index_name,
    table_name,
    base_table,
    query_column,
    top_k,
    aggregation,
):
    query_result = load_query_result(
        data_lake_version, index_name, table_name, query_column, 0
    )
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


def test_aggr(
    data_lake_version,
    index_name,
    table_name,
    base_table,
    query_column,
    top_k,
):
    query_result = load_query_result(
        data_lake_version, index_name, table_name, query_column, 0
    )
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

    base_columns = base_table.columns

    for rank, (c_id, cand) in enumerate(query_result.candidates.items()):
        r_dict = dict(base_results)
        _, cnd_md, left_on, right_on = cand.get_join_information()
        cand_table = pl.read_parquet(cnd_md["full_path"])

        cont = df_counts.filter(
            (pl.col("hash") == cnd_md["hash"]) & (pl.col("col") == right_on)
        )["containment"].item()

        merges = {}
        for agg_case in ["first", "mean"]:
            merges[agg_case] = execute_join_with_aggregation(
                base_table,
                cand_table,
                left_on=left_on,
                right_on=right_on,
                how="left",
                aggregation=agg_case,
            )

        equals_dict = {
            "frac_equals": [],
            "nn_first": [],
            "nn_second": [],
            "unique_first": [],
            "unique_second": [],
        }

        for comb in combinations(merges, 2):
            first = merges[comb[0]].select(cs.string())
            second = merges[comb[1]].select(cs.string())

            for col in [c for c in first.columns if c not in base_columns]:
                unique_first = len(first.select(pl.col(col).unique()))
                unique_second = len(second.select(pl.col(col).unique()))

                nn_first = first.filter(pl.col(col).is_not_null())[col]
                nn_second = second.filter(pl.col(col).is_not_null())[col]
                max_nn = max(len(nn_first), len(nn_second))
                equals = first.filter(pl.col(col) == second[col]).shape[0]

                frac_equals = equals / max_nn if max_nn > 0 else 0
                equals_dict["frac_equals"].append(frac_equals)
                equals_dict["nn_first"].append(len(nn_first))
                equals_dict["nn_second"].append(len(nn_second))
                equals_dict["unique_first"].append(unique_first)
                equals_dict["unique_second"].append(unique_second)

        r_dict["cnd_table"] = cnd_md["hash"]
        r_dict["cnd_column"] = right_on[0]
        r_dict["containment"] = cont
        r_dict["src_nrows"], r_dict["src_ncols"] = base_table.shape
        r_dict["cnd_nrows"], r_dict["cnd_ncols"] = cand_table.shape
        r_dict["top_k"] = top_k
        r_dict["rank"] = rank
        r_dict["avg_frac_equals"] = np.mean(equals_dict["frac_equals"])
        r_dict["avg_nn_first"] = np.mean(equals_dict["nn_first"])
        r_dict["avg_nn_second"] = np.mean(equals_dict["nn_second"])
        r_dict["avg_unique_first"] = np.mean(equals_dict["unique_first"])
        r_dict["avg_unique_second"] = np.mean(equals_dict["unique_second"])

        list_stats.append(r_dict)
    print(f"{data_lake_version} {table_name} {top_k} {total_time:.2f}")
    return list_stats


def test_col_stats(
    data_lake_version,
    index_name,
    table_name,
    base_table,
    query_column,
    top_k,
):
    query_result = load_query_result(
        data_lake_version, index_name, table_name, query_column, 0
    )
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

    base_columns = base_table.columns

    col_stats = []
    single_col_stats_ls = []

    for rank, (c_id, cand) in enumerate(query_result.candidates.items()):
        r_dict = dict(base_results)
        _, cnd_md, left_on, right_on = cand.get_join_information()
        this_cand = pl.read_parquet(cnd_md["full_path"])
        cand_table = this_cand.filter(pl.col(right_on).is_in(base_table[left_on]))
        cat_cols = cand_table.select(cs.string()).columns
        cat_cols = [_ for _ in cat_cols if _ not in right_on]

        cont = df_counts.filter(
            (pl.col("hash") == cnd_md["hash"]) & (pl.col("col") == right_on)
        )["containment"].item()

        for col in cat_cols:
            d_stats = {
                "candidate_hash": cnd_md["hash"],
                "candidate_name": cnd_md["df_name"],
                "column_name": col,
                "column_cardinality": [],
                "column_frac_in_mode": [],
                "column_frac_nulls": [],
            }
            d_stats["column_cardinality"] = cand_table.select(
                pl.col(col).n_unique()
            ).item()
            d_stats["column_frac_in_mode"] = cand_table.filter(
                pl.col(col) == pl.col(col).mode().sort(descending=True).first()
            ).shape[0] / len(cand_table)
            d_stats["column_frac_nulls"] = cand_table.filter(
                pl.col(col).is_null()
            ).shape[0] / len(cand_table)

        # list_stats.append(r_dict)
    print(f"{data_lake_version} {table_name} {top_k} {total_time:.2f}")
    return single_col_stats_ls
    # return list_stats


def test_group_stats(
    data_lake_version: str,
    index_name: str,
    table_name: str,
    base_table: pl.DataFrame,
    query_column: str,
    top_k: int,
):
    query_result = load_query_result(
        data_lake_version, index_name, table_name, query_column, 0
    )
    query_result.select_top_k(top_k)
    total_time = 0

    list_stats = []

    for rank, (c_id, cand) in tqdm(
        enumerate(query_result.candidates.items()),
        total=len(query_result.candidates),
        position=2,
    ):
        _, cnd_md, left_on, right_on = cand.get_join_information()
        this_cand = pl.read_parquet(cnd_md["full_path"])
        cand_table = this_cand.filter(pl.col(right_on).is_in(base_table[left_on]))
        cat_cols = cand_table.select(cs.string()).columns
        cat_cols = [_ for _ in cat_cols if _ not in right_on]

        dict_stats = {
            col: {
                "table_name": table_name,
                "cand_hash": cnd_md["hash"],
                "cand_table": cnd_md["df_name"],
                "col_name": col,
                "in_mode": 0,
                "equal_aggr": 0,
                "nulls": 0,
                "unique": 0,
                "grp_size": 0,
            }
            for col in cat_cols
        }

        for col in tqdm(cat_cols, total=len(cat_cols), position=1, leave=False):
            if col in right_on:
                continue
            subtable = cand_table.select(right_on + [col])
            this_col_stats = {
                "in_mode": [],
                "equal_aggr": [],
                "nulls": [],
                "unique": [],
                "grp_size": [],
            }
            for gidx, group in tqdm(
                subtable.group_by(right_on),
                position=0,
                total=len(cand_table),
                leave=False,
            ):
                _stats = (
                    group.lazy()
                    .with_columns(
                        pl.col(col).null_count().alias("nulls"),
                    )
                    .fill_null(f"null_{gidx[0]}")
                    .with_columns(
                        pl.col(col).mode().first().alias("mode"),
                        pl.col(col).first().alias("first"),
                        (
                            pl.col(col)
                            .value_counts(sort=True)
                            .struct.rename_fields(["val", "count"])
                            .struct.field("count")
                            / len(group)
                        ).alias("in_mode"),
                        pl.col(col).n_unique().alias("unique"),
                        pl.lit(len(group)).alias("grp_size"),
                    )
                    .with_columns(
                        (pl.col("mode") == pl.col("first")).alias("equal_aggr")
                    )
                    .select(["in_mode", "unique", "nulls", "equal_aggr", "grp_size"])
                    .collect()
                    .to_dict(as_series=False)
                )
                for key in this_col_stats:
                    this_col_stats[key] += _stats[key]
            dict_stats[col].update(
                {key: np.mean(value) for key, value in this_col_stats.items()}
            )
        list_stats += list(dict_stats.values())
    print(f"{data_lake_version} {table_name} {top_k} {total_time:.2f}")
    return list_stats


if __name__ == "__main__":
    data_lake_version = "wordnet_full"
    query_column = "col_to_embed"
    top_k = 0
    index_names = [
        "exact_matching"
        # "minhash", "minhash_hybrid", "exact_matching"
    ]
    keys = ["index_name", "tab_name", "top_k", "join_time", "avg_cont"]
    results = []

    mode = "group_stats"

    version = "yadl"  # or open_data_us
    base_path = Path(f"data/source_tables/{version}")

    queries = [
        ("company_employees", "col_to_embed"),
        ("housing_prices", "col_to_embed"),
        ("us_elections", "col_to_embed"),
        ("movies", "col_to_embed"),
        # ("movies_vote", "col_to_embed"),
        ("us_accidents", "col_to_embed"),
    ]
    # queries = [
    #     ("company_employees", "name"),
    #     ("housing_prices", "County"),
    #     ("us_elections", "county_name"),
    #     ("movies", "title"),
    #     ("movies_vote", "title"),
    #     ("us_accidents", "County"),
    #     ("schools", "col_to_embed"),
    # ]

    for query in queries:
        for iname in index_names:
            print(iname)
            for k in [5]:
                aggr = "first"
                tab, query_column = query
                table_name = f"{tab}-yadl"
                # for aggr in ["first", "mean"]:
                base_table = pl.read_parquet(
                    Path(f"data/source_tables/{version}/{table_name}.parquet")
                )

                params = {
                    "data_lake_version": data_lake_version,
                    "index_name": iname,
                    "table_name": table_name,
                    "base_table": base_table,
                    "query_column": query_column,
                    "top_k": k,
                }

                if mode == "stats":
                    params.update({"aggregation": aggr})
                    this_res = test_joining(**params)

                    results += this_res
                elif mode == "aggr":
                    this_res = test_aggr(**params)

                    results += this_res
                elif mode == "col_stats":
                    col_stats = test_col_stats(**params)
                    results += col_stats

                elif mode == "group_stats":
                    col_stats = test_group_stats(**params)
                    results += col_stats

    df = pl.from_dicts(results)
    out_path = Path(f"analysis_query_results_{data_lake_version}_{mode}_all.csv")
    if out_path.exists():
        df.write_csv(open(out_path, "a"), include_header=False)
    else:
        df.write_csv(open(out_path, "w"))
