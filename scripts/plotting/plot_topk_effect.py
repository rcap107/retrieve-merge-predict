# %%
import polars as pl
from pathlib import Path
import json

# %%
exp_ids = ["0709", "0710", "0711", "0712", "0713", "0714"]


def unpack_dict(log_dict):
    dd = dict()
    n_splits = log_dict["n_splits"]
    for k, v in log_dict["query_info"].items():
        dd[k] = [v] * n_splits

    if log_dict["task"] == "classification":
        dd["results"] = [x["f1"] for x in log_dict["results"]]
    else:
        dd["results"] = [x["r2"] for x in log_dict["results"]]
    dd["estimator"] = [x["estimator"] for x in log_dict["results"]]
    return {
        k: dd[k]
        for k in [
            "data_lake",
            "join_discovery_method",
            "estimator",
            "table_path",
            "query_column",
            "top_k",
            "results",
        ]
    }


# %%
paths = []
for f in Path("../../results/logs").iterdir():
    if f.stem.split("-")[0] in exp_ids:
        paths.append(f)
# %%
all_dfs = []

for f in paths:
    for _log in Path(f, "json").iterdir():
        if _log.is_dir():
            continue
        log_dict = json.load(open(_log, "r"))
        all_dfs.append(pl.DataFrame(unpack_dict(log_dict)))

# %%
df = pl.concat(all_dfs)
df = df.with_columns(
    table=pl.col("table_path").str.split("/").list.last().str.split("-").list.first()
).filter(pl.col("table") != "us_accidents_large")
df
# %%
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# fig, ax = plt.subplots(1,1, squeeze=True, layout="constrained")
sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="results",
    hue="table",
    row="estimator",
    col="data_lake",
    marker="o",
    kind="line",
    facet_kws={"sharey": True}
)
# %%
sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="results",
    hue="data_lake",
    row="estimator",
    col="table",
    marker="o",
    kind="line",
    facet_kws={"sharey": True}
)

# %%
