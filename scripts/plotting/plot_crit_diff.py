# %% plot critical difference figures
# %cd ~/bench

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scikit_posthocs as sp
import seaborn as sns

from src.utils import constants, plotting
from src.utils.critical_difference_plot import critical_difference_diagram
from src.utils.logging import read_and_process

sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
# %%
df = pl.read_parquet("results/temp_results_retrieval.parquet")
df = df.with_columns(
    pl.when(pl.col("prediction_metric") < -1)
    .then(-1)
    .otherwise(pl.col("prediction_metric"))
    .alias("y")
)

df = df.filter(
    # (pl.col("chosen_model") == "catboost") &
    (pl.col("estimator") != "nojoin")
)

keys = ["jd_method", "estimator", "chosen_model"]
exp_keys = ["base_table", "target_dl"]


names = df.unique(keys).select(keys).sort(keys).with_row_index("model")

df = df.join(names, on=keys, how="left")
experiments = (
    df.unique(exp_keys).select(exp_keys).sort(exp_keys).with_row_index("experiment")
)
df = df.join(experiments, on=exp_keys, how="left")

ranks = (
    df.group_by(["experiment", "model"])
    .agg(pl.mean("y"))
    .with_columns(
        pl.col("y").rank("ordinal", descending=True).over(["experiment"]).alias("rank")
    )
    .select("model", "rank")
)

rank_by_model = (
    df.group_by(["experiment", "model"])
    .agg(pl.mean("y"))
    .with_columns(
        pl.col("y").rank("ordinal", descending=True).over(["experiment"]).alias("rank")
    )
    .group_by("model")
    .agg(
        rank_median=pl.median("rank"),
        rank_mean=pl.mean("rank"),
    )
)

df_avg = df.group_by(keys).agg(pl.mean("y")).sort(keys).with_row_index(name="model")
p_values = sp.posthoc_conover(df.to_pandas(), val_col="y", group_col="model")

n = (
    names.with_columns(
        jd_method=pl.col("jd_method").replace(constants.LABEL_MAPPING["jd_method"]),
        estimator=pl.col("estimator").replace(constants.LABEL_MAPPING["estimator"]),
        chosen_model=pl.col("chosen_model").replace(
            constants.LABEL_MAPPING["chosen_model"]
        ),
    )
    .with_columns(
        prepared_name=pl.col("jd_method")
        + "-"
        + pl.col("estimator")
        # + "-"
        # + pl.col("aggregation")
        + "-"
        + pl.col("chosen_model")
    )
    .select(pl.col("model", "prepared_name"))
)
this_d = (
    ranks.group_by("model")
    .agg(pl.mean("rank"))
    .join(n, on="model")
    .select(["prepared_name", "rank"])
    # .with_columns(rank=-pl.col("rank"))
)
dd = dict(zip(*this_d.to_dict().values()))

# %% Coloring by retrieval method
colors = ["green", "orange", "red", "blue"]
mp = dict(zip(constants.LABEL_MAPPING["jd_method"].values(), colors))
print(mp)
color_props = {k: mp[k.split("-")[0]] for k in dd}

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(12, 10))
_ = critical_difference_diagram(
    dd, p_values, color_palette=color_props, ascending=False
)
fig.subplots_adjust(left=0, right=2)
fig.savefig(
    "images/crit_diff_plot_retrieval_method.png",
    bbox_inches="tight",
    # pad_inches=0.1
)
fig.savefig(
    "images/crit_diff_plot_retrieval_method.pdf",
    bbox_inches="tight",
)
# %% Coloring by selector
colors = ["green", "orange", "red", "blue"]
mp = dict(zip(constants.LABEL_MAPPING["estimator"].values(), colors))
color_props = {k: mp[k.split("-")[1]] for k in dd}
print(mp)

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(12, 10))
_ = critical_difference_diagram(
    dd, p_values, color_palette=color_props, ascending=False
)
fig.subplots_adjust(left=0, right=2)
fig.savefig(
    "images/crit_diff_plot_selector.png",
    bbox_inches="tight",
    # pad_inches=0.1
)
fig.savefig(
    "images/crit_diff_plot_selector.pdf",
    bbox_inches="tight",
)

# %% Coloring by ML model
colors = ["green", "orange", "red", "blue"]
mp = dict(zip(constants.LABEL_MAPPING["chosen_model"].values(), colors))
color_props = {k: mp[k.split("-")[2]] for k in dd}
print(mp)

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(12, 10))
_ = critical_difference_diagram(
    dd, p_values, color_palette=color_props, ascending=False
)
fig.subplots_adjust(left=0, right=2)
fig.savefig(
    "images/crit_diff_plot_ml_model.png",
    bbox_inches="tight",
    # pad_inches=0.1
)
fig.savefig(
    "images/crit_diff_plot_ml_model.pdf",
    bbox_inches="tight",
)

# %%
