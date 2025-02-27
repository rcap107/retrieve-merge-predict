# %%
# %cd ~/bench
# %%
import polars as pl
from pathlib import Path
import json

import polars.selectors as cs


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.patches import Rectangle


from src.utils import constants

sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
XTICKS_INSET = [1, 2, 3, 5, 8, 10]

# %%
exp_ids = [
    "0709",
    "0710",
    "0711",
    "0712",
    "0713",
    "0714",
    "0715",
    "0716",
    "0717",
    "0718",
    "0719",
    "0720",
]


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
def get_run_df(run_path):
    dfs = []
    with open(Path(run_path, "scenario_id"), "r") as fp:
        scenario_id = int(fp.read())
    for _i in range(scenario_id + 1):
        json_path = Path(run_path, "json", str(_i) + ".json")
        log_path = Path(run_path, "run_logs", str(_i) + ".log")
        log_json = json.load(open(json_path, "r"))

        top_k = log_json["top_k"]
        exp_task = log_json["task"]

        _df = pl.read_csv(log_path)
        _df = _df.with_columns(top_k=pl.lit(top_k).alias("top_k"))
        if exp_task == "regression":
            _df = _df.with_columns(
                pl.lit(0.0).alias("auc"),
                pl.lit(0.0).alias("f1score"),
                prediction_metric=pl.col("r2score"),
            )
        else:
            _df = _df.with_columns(
                pl.lit(0.0).alias("r2score"),
                pl.lit(0.0).alias("rmse"),
                prediction_metric=pl.col("f1score"),
            )

        dfs.append(_df)
    return pl.concat(dfs)


# %%
paths = []
for f in Path("results/logs").iterdir():
    if f.stem.split("-")[0] in exp_ids:
        paths.append(f)

all_dfs = []
for f in paths:
    x = get_run_df(f)
    all_dfs.append(x)
df = pl.concat(all_dfs)
df = df.with_columns(table=pl.col("base_table").str.split("-").list.first()).filter(
    (pl.col("table") != "us_accidents_large") & (pl.col("estimator") == "full_join")
)


# %%
import numpy as np


def major_gigabyte_formatter(x, pos):
    return f"{x/1e3:.0f} GB"


def minor_gigabyte_formatter(x, pos):
    return f"{x/1e3:.0f}"


def major_time_formatter(x, pos):
    # return f"{x/60:.0f}min"
    if x > 60:
        return f"{x/60:.0f}min"
    else:
        return f"{x:.0f}s"


# Create a FuncFormatter object
major_gb_formatter = FuncFormatter(major_gigabyte_formatter)
minor_gb_formatter = FuncFormatter(minor_gigabyte_formatter)

# Fixed locators
# major_time_locator = FixedLocator([10, 120, 600, 3600])
gb_locator = FixedLocator(np.linspace(2000, 16000, 5))
# %%
# fig, ax = plt.subplots(1,1, squeeze=True, layout="constrained")
g = sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="prediction_metric",
    hue="table",
    # row="estimator",
    col="target_dl",
    marker="o",
    kind="line",
    style="estimator",
    facet_kws={"sharey": True},
)

for ax in g.axes.flat:
    col_value = ax.get_title().split(" = ")[-1]  # Extract the value after ' = '
    cv = constants.LABEL_MAPPING["target_dl"][col_value]
    ax.set_title(f"{cv}", fontsize=14)
    ax.set_ylabel("Prediction performance")
    ax.set_xlabel("Number of retrieved candidates")

g.savefig("")

# %%
# fig, ax = plt.subplots(1,1, squeeze=True, layout="constrained")
g = sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="time_run",
    hue="table",
    # row="estimator",
    col="target_dl",
    marker="o",
    kind="line",
    facet_kws={"sharey": True},
)

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(major_time_formatter)
    col_value = ax.get_title().split(" = ")[-1]  # Extract the value after ' = '
    cv = constants.LABEL_MAPPING["target_dl"][col_value]
    ax.set_title(f"{cv}", fontsize=14)
    ax.set_ylabel("Single fold runtime")
    ax.set_xlabel("Number of retrieved candidates")

    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper right")
    sns.lineplot(
        x="top_k",
        y="time_run",
        ax=ax_inset,
        data=df.filter(target_dl=col_value),
        hue="table",
        legend=None,
        marker="o",
    )

    ax_inset.set_xlim(0, 11)
    ax_inset.set_ylim(0, 50)

    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")

    ax_inset.yaxis.set_major_formatter(major_time_formatter)

    ax_inset.set_xticks(XTICKS_INSET)
    ax_inset.xaxis.set_minor_locator(
        AutoMinorLocator(1)
    )  # Minor ticks between major ticks on x-axis

    rect = Rectangle(
        (0, 0), 11, 50, linewidth=1, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)


# %%
# fig, ax = plt.subplots(1,1, squeeze=True, layout="constrained")
g = sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="peak_fit",
    hue="table",
    # row="estimator",
    col="target_dl",
    marker="o",
    kind="line",
    facet_kws={"sharey": True, "sharex": True},
)


for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(major_gb_formatter)
    ax.yaxis.set_minor_formatter(minor_gb_formatter)
    col_value = ax.get_title().split(" = ")[-1]  # Extract the value after ' = '
    cv = constants.LABEL_MAPPING["target_dl"][col_value]
    ax.set_title(f"{cv}", fontsize=14)
    ax.set_ylabel("Peak RAM fit")
    ax.set_xlabel("Number of retrieved candidates")

    loc = "upper right"

    ax_inset = inset_axes(ax, width="30%", height="30%", loc=loc)
    sns.lineplot(
        x="top_k",
        y="peak_fit",
        ax=ax_inset,
        data=df.filter(target_dl=col_value),
        hue="table",
        legend=None,
        marker="o",
    )

    ax_inset.set_xlim(0, 11)
    ax_inset.set_ylim(1000, 5000)

    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")

    ax_inset.yaxis.set_major_formatter(major_gb_formatter)
    ax_inset.yaxis.set_minor_formatter(minor_gb_formatter)

    ax_inset.set_xticks(XTICKS_INSET)
    ax_inset.xaxis.set_minor_locator(
        AutoMinorLocator(1)
    )  # Minor ticks between major ticks on x-axis

    rect = Rectangle(
        (0, 1000),
        11,
        5000,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


# %%
df.sort("top_k").pivot(
    on="top_k",
    index="target_dl",
    values="prediction_metric",
    aggregate_function="mean",
).with_columns(
    pl.col("target_dl").replace(constants.LABEL_MAPPING["target_dl"])
).with_columns(
    cs.numeric().round(3)
)
# %%
# %%
df.sort("top_k").pivot(
    on="top_k",
    index="target_dl",
    values="peak_fit",
    aggregate_function="mean",
).with_columns(
    pl.col("target_dl").replace(constants.LABEL_MAPPING["target_dl"])
).with_columns(
    cs.numeric().round(0)
)
# %%
df.sort("top_k").pivot(
    on="top_k",
    index="target_dl",
    values="time_run",
    aggregate_function="mean",
).with_columns(
    pl.col("target_dl").replace(constants.LABEL_MAPPING["target_dl"])
).with_columns(
    cs.numeric().round(3)
)

# %%
from src.utils.plotting import pareto_frontier_plot

# %%
df_pareto = (
    df.group_by("top_k")
    .agg(pl.mean("time_run"), pl.mean("prediction_metric"))
    .sort("top_k")
)
hue_order = df_pareto["top_k"].to_numpy()

df_sem = (
    df.group_by("top_k")
    .agg(
        pl.count("prediction_metric").alias("count"),
        pl.std("prediction_metric").alias("std_pred"),
        pl.std("time_run").alias("std_time"),
    )
    .with_columns(
        sem_time=pl.col("std_time") / (pl.col("count") ** 0.5),
        sem_pred=pl.col("std_pred") / (pl.col("count") ** 0.5),
    )
    .sort("top_k")
)
xerr = df_sem["sem_time"].to_numpy()
yerr = df_sem["sem_pred"].to_numpy()

# %%
fig, ax = plt.subplots(1, 1, squeeze=True)
pareto_frontier_plot(
    df_pareto.to_pandas(),
    x_var="time_run",
    y_var="prediction_metric",
    hue_var="top_k",
    palette="tab10",
    hue_order=hue_order,
    ax=ax,
    ax_title="",
    ax_xlabel="",
)
ax.set_ylim([0.4, 0.6])
ax.set_ylabel("Prediction performance")
ax.set_xlabel("Time run (s)")

scatter = ax.collections[0]  # Get the first collection (scatter points)
colors = (
    scatter.get_facecolor()
)  # Get the facecolors (which represent the colors for each point)

for idx, c in enumerate(colors):
    ax.errorbar(
        x=df_pareto["time_run"][idx],
        y=df_pareto["prediction_metric"][idx],
        xerr=xerr[idx],
        yerr=yerr[idx],
        fmt="none",
        ecolor=c,
    )
# %%

# %%
