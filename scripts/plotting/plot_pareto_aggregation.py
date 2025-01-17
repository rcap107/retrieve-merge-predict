#%%
# %cd ~/bench
#%%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src.utils import constants, plotting
from src.utils.critical_difference_plot import critical_difference_diagram
from src.utils.logging import read_and_process

sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")


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
major_time_locator = FixedLocator([10, 120, 600, 3600])
gb_locator = FixedLocator([2000, 3000, 4000, 5000, 7000, 10000])

# Setting constants
hues = {0: "estimator", 1: "chosen_model", 2: "jd_method"}
titles = {0: "Selector", 1: "Prediction Model", 2: "Retrieval method"}
palettes = {0: "tab10", 1: "tab10", 2: "tab10"}


df = pl.read_parquet("results/results_aggregation.parquet")
df = df.with_columns(
    pl.when(pl.col("prediction_metric") < -1)
    .then(-1)
    .otherwise(pl.col("prediction_metric"))
    .alias("y")
).filter(pl.col("estimator") != "nojoin")

keys = ["jd_method", "estimator", "aggregation", "chosen_model"]
exp_keys = ["base_table", "target_dl"]
names = df.unique(keys).select(keys).sort(keys).with_row_index("model")
df = df.join(names, on=keys, how="left")
experiments = (
    df.unique(exp_keys).select(exp_keys).sort(exp_keys).with_row_index("experiment")
)
df = df.join(experiments, on=exp_keys, how="left")
res = df.group_by(keys).agg(
    pl.mean("y"),
    pl.mean("time_run"),
    pl.mean("peak_fit"),
)
data = res.to_pandas()
#%%
hue_order = {
    "estimator": ["highest_containment", "best_single_join"],
    "chosen_model": ["catboost", "ridgecv", "resnet", "realmlp"],
    "jd_method": ["exact_matching", "minhash", "minhash_hybrid"],
    "aggregation": ["first", "mean", "dfs"],
}

#%% Time
fig, axs = plt.subplots(
    1,
    4,
    squeeze=True,
    sharey=True,
    sharex=True,
    figsize=(14, 4),
    # gridspec_kw={"hspace": 0.4},
    layout="constrained",
)

variable = "time_run"
y_var = "y"


groups = ["jd_method", "estimator", "aggregation", "chosen_model"]
for pl_id in range(4):
    group_variable = groups[pl_id]
    # ax = axs[pl_id // 2][pl_id % 2]
    ax = axs[pl_id]
    ax.set_xscale("log")
    idx_ = pl_id
    (h, l), _ = plotting.pareto_frontier_plot(
        data,
        variable,
        y_var,
        hue_var=group_variable,
        palette="tab10",
        hue_order=hue_order[group_variable],
        ax=ax,
        ax_title="",
        ax_xlabel="",
    )
    l = [constants.LABEL_MAPPING[groups[pl_id]][_] for _ in l]

    ax.legend(
        h,
        l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        title=constants.LABEL_MAPPING["variables"][groups[pl_id]],
        ncols=2,
        mode="expand",
        edgecolor="white",
        # columnspacing=50
    )
    ax.xaxis.set_major_formatter(major_time_formatter)
    ax.xaxis.set_major_locator(major_time_locator)

fig.supylabel("Prediction Performance")
fig.supxlabel("Time run (s)")

fig.savefig("images/pareto_aggregation_time.png")
fig.savefig("images/pareto_aggregation_time.pdf", bbox_inches="tight")

#%% Peak RAM
fig, axs = plt.subplots(
    1,
    4,
    squeeze=True,
    sharey=True,
    sharex=True,
    figsize=(14, 4),
    # gridspec_kw={"hspace": 0.4},
    layout="constrained",
)

variable = "peak_fit"
y_var = "y"


groups = ["jd_method", "estimator", "aggregation", "chosen_model"]
for pl_id in range(4):
    group_variable = groups[pl_id]
    # ax = axs[pl_id // 2][pl_id % 2]
    ax = axs[pl_id]
    ax.set_xscale("log")
    idx_ = pl_id
    (h, l), _ = plotting.pareto_frontier_plot(
        data,
        variable,
        y_var,
        hue_var=group_variable,
        palette="tab10",
        hue_order=hue_order[group_variable],
        ax=ax,
        ax_title="",
        ax_xlabel="",
    )
    l = [constants.LABEL_MAPPING[groups[pl_id]][_] for _ in l]

    ax.legend(
        h,
        l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        title=constants.LABEL_MAPPING["variables"][groups[pl_id]],
        ncols=2,
        mode="expand",
        edgecolor="white",
        # columnspacing=50
    )
    ax.xaxis.set_major_formatter(major_gigabyte_formatter)
    ax.xaxis.set_major_locator(gb_locator)
    ax.xaxis.set_minor_locator(NullLocator())


fig.supylabel("Prediction Performance")
fig.supxlabel("Peak RAM (MB)")


fig.savefig("images/pareto_aggregation_ram.png")
fig.savefig("images/pareto_aggregation_ram.pdf", bbox_inches="tight")
#%% Only aggregation time for main body
fig, ax = plt.subplots(1, 1, squeeze=True, sharey=True, sharex=True, figsize=(5, 3))

variable = "time_run"
y_var = "y"


group_variable = "aggregation"
ax.set_xscale("log")
idx_ = pl_id
(h, l), _ = plotting.pareto_frontier_plot(
    data,
    variable,
    y_var,
    hue_var=group_variable,
    palette="tab10",
    hue_order=hue_order[group_variable],
    ax=ax,
    ax_title="",
    ax_xlabel="",
)
l = [constants.LABEL_MAPPING["aggregation"][_] for _ in l]

ax.legend(
    h,
    l,
    # loc="upper center",
    # bbox_to_anchor=(0.5, 1.4),
    title="Aggregation",
    ncols=1,
    # mode="expand",
    edgecolor="white",
    # columnspacing=50
)
ax.xaxis.set_major_formatter(major_time_formatter)
ax.xaxis.set_major_locator(major_time_locator)

ax.set_ylabel("Prediction Performance")
ax.set_xlabel("Time run (s)")

fig.savefig("images/pareto_aggregation_time_single.png", bbox_inches="tight")
fig.savefig("images/pareto_aggregation_time_single.pdf", bbox_inches="tight")

# %%
