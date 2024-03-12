import datetime
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
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
        desc="Candidate: ",
    ):
        _, cnd_md, left_on, right_on = cand.get_join_information()
        this_cand = pl.read_parquet(cnd_md["full_path"])
        cand_table = this_cand.filter(pl.col(right_on).is_in(base_table[left_on]))
        cat_cols = cand_table.select(cs.string()).columns
        cat_cols = [_ for _ in cat_cols if _ not in right_on]

        dict_stats = {
            col: {
                "data_lake_version": data_lake_version,
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

        for col in tqdm(
            cat_cols, total=len(cat_cols), position=1, leave=False, desc="Column: "
        ):
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
            n_gr = subtable.select(pl.col(right_on).n_unique()).item()
            for gidx, group in tqdm(
                subtable.group_by(right_on),
                position=0,
                total=n_gr,
                leave=False,
                desc="Group: ",
            ):
                _stats = {}
                _stats["nulls"] = group.select(pl.col(col).null_count()).item()
                group = group.fill_null(f"null_{gidx[0]}")
                _eq = group.select(
                    pl.col(col).mode().first().alias("mode")
                    == pl.col(col).first().alias("first")
                ).item()
                _stats["equal_aggr"] = _eq
                _stats["in_mode"] = group.select(
                    pl.col(col)
                    .value_counts(sort=True)
                    .struct.rename_fields(["val", "count"])
                    .struct.field("count")
                    .first()
                    / len(group)
                ).item()
                _stats["unique"] = group.select(
                    pl.col(col).n_unique().alias("unique")
                ).item()
                _stats["grp_size"] = len(group)

                for key, val in this_col_stats.items():
                    val.append(_stats[key])

                # for key in this_col_stats:
                #     this_col_stats[key].append(_stats[key])
            dict_stats[col].update(
                {key: np.mean(value) for key, value in this_col_stats.items()}
            )
        list_stats += list(dict_stats.values())

    return list_stats


if __name__ == "__main__":
    data_lake_version = "binary_update"
    print("Data lake: ", data_lake_version)
    index_names = [
        "exact_matching"
        # "minhash", "minhash_hybrid", "exact_matching"
    ]
    keys = ["index_name", "tab_name", "top_k", "join_time", "avg_cont"]
    results = []

    mode = "group_stats"

    version = "yadl"  # or open_data_us
    base_path = Path(f"data/source_tables/{version}")

    queries = {
        "open_data_us": [
            ("company_employees", "name"),
            ("housing_prices", "County"),
            ("us_elections", "county_name"),
            ("movies", "title"),
            ("us_accidents", "County"),
            ("schools", "col_to_embed"),
        ],
        "yadl": [
            ("company_employees", "col_to_embed"),
            ("housing_prices", "col_to_embed"),
            ("us_elections", "col_to_embed"),
            ("movies", "col_to_embed"),
            ("us_accidents", "col_to_embed"),
        ],
    }

    for query in queries[version]:
        for iname in index_names:
            print(iname)
            for k in [30]:
                aggr = "first"
                tab, query_column = query
                # table_name = f"{tab}-open_data"
                table_name = f"{tab}-yadl-depleted"
                print(f"{data_lake_version} {table_name}")
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
                elif mode == "group_stats":
                    col_stats = test_group_stats(**params)
                    results += col_stats

    df = pl.from_dicts(results)
    out_path = Path(f"analysis_query_results_{data_lake_version}_{mode}_all.csv")
    if out_path.exists():
        df.write_csv(open(out_path, "a"), include_header=False)
    else:
        df.write_csv(open(out_path, "w"))
