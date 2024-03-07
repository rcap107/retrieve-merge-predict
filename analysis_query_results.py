import datetime
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
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

        # start_time = datetime.datetime.now()
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

        equals_dict = {}
        for comb in combinations(merges, 2):
            first = merges[comb[0]].select(cs.string())
            second = merges[comb[1]].select(cs.string())

            for col in [c for c in first.columns if c not in base_columns]:
                nn_first = first.filter(pl.col(col).is_not_null())[col]
                nn_second = second.filter(pl.col(col).is_not_null())[col]
                max_nn = max(len(nn_first), len(nn_second))
                equals = first.filter(pl.col(col) == second[col]).shape[0]

                frac_equals = equals / max_nn if max_nn > 0 else 0
                equals_dict[f"frac_{comb[0]}_{comb[1]}"] = frac_equals
                equals_dict[f"nn_first"] = len(nn_first)
                equals_dict[f"nn_second"] = len(nn_second)
        # end_time = datetime.datetime.now()
        # time_required = (end_time - start_time).total_seconds()
        # total_time += time_required

        r_dict["cnd_table"] = cnd_md["hash"]
        r_dict["cnd_column"] = right_on[0]
        r_dict["containment"] = cont
        r_dict["src_nrows"], r_dict["src_ncols"] = base_table.shape
        r_dict["cnd_nrows"], r_dict["cnd_ncols"] = cand_table.shape
        # r_dict["join_time"] = time_required
        r_dict["top_k"] = top_k
        r_dict["rank"] = rank
        r_dict.update(equals_dict)

        list_stats.append(r_dict)
    print(f"{data_lake_version} {table_name} {top_k} {total_time:.2f}")
    return list_stats


if __name__ == "__main__":
    data_lake_version = "wordnet_full"
    query_column = "col_to_embed"
    top_k = 0
    index_names = [
        "minhash_hybrid",
    ]
    keys = ["index_name", "tab_name", "top_k", "join_time", "avg_cont"]
    results = []

    mode = "aggr"

    version = "yadl"  # or open_data_us
    base_path = Path(f"data/source_tables/{version}")

    queries = [
        ("company_employees", "col_to_embed"),
        ("housing_prices", "col_to_embed"),
        ("us_elections", "col_to_embed"),
        ("movies", "col_to_embed"),
        ("movies_vote", "col_to_embed"),
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
            for k in [30]:
                aggr = "first"
                tab, query_column = query
                table_name = f"{tab}-yadl"
                # for aggr in ["first", "mean"]:
                base_table = pl.read_parquet(
                    Path(f"data/source_tables/{version}/{table_name}.parquet")
                )
                if mode == "stats":
                    this_res = test_joining(
                        data_lake_version,
                        iname,
                        table_name,
                        base_table,
                        query_column,
                        k,
                        aggregation=aggr,
                    )

                    results += this_res
                elif mode == "aggr":
                    this_res = test_aggr(
                        data_lake_version,
                        iname,
                        table_name,
                        base_table,
                        query_column,
                        k,
                    )

                    results += this_res

    df = pl.from_dicts(results)
    df.write_csv(open(f"analysis_query_results_{data_lake_version}_{mode}.csv", "a"))
