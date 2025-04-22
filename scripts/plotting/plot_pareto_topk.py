"""This script is used to prepare the Pareto plots that compare the performance
by value of top-k, considering both the run time and the peak RAM.
"""

# %%
import os

os.chdir("../..")
# %%
import matplotlib.pyplot as plt
import polars as pl

from src.utils import constants
from src.utils.plotting import pareto_frontier_plot

plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
df = pl.read_csv("results/results_topk.csv")
df = df.group_by(constants.GROUPING_KEYS + ["top_k"]).agg(
    pl.mean("time_run", "peak_fit", "prediction_metric")
)
# %%
df_pareto = (
    df.group_by("top_k")
    .agg(pl.mean("time_run"), pl.mean("prediction_metric"), pl.mean("peak_fit"))
    .sort("top_k")
)
hue_order = df_pareto["top_k"].to_numpy()


def prepare_sem_df(df, variable):
    _df_sem = (
        df.group_by("top_k")
        .agg(
            pl.count("prediction_metric").alias("count"),
            pl.std("prediction_metric").alias("std_pred"),
            pl.std(variable).alias(f"std_{variable}"),
        )
        .with_columns(
            (pl.col(f"std_{variable}") / (pl.col("count") ** 0.5)).alias(
                f"sem_{variable}"
            ),
            sem_pred=pl.col("std_pred") / (pl.col("count") ** 0.5),
        )
        .sort("top_k")
    )

    return _df_sem


# %%
df_sem = prepare_sem_df(df, "time_run")

xerr = df_sem["sem_time_run"].to_numpy()
yerr = df_sem["sem_pred"].to_numpy()

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(5, 3.5), layout="constrained")
(h, l), _ = pareto_frontier_plot(
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
ax.set_ylim([0.43, 0.55])
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

ax.legend(
    h,
    l,
    title="Value of k",
    loc="upper right",
    bbox_to_anchor=(1.35, 1.05),
    frameon=False,
    title_fontsize="large",
)
_x, _y = df_pareto.filter(top_k=30).select("time_run", "prediction_metric")

x_text = _x.item()
y_text = _y.item()

ax.annotate(
    "k used in experiments",  # Annotation text
    xy=(x_text, y_text),  # Point to annotate
    xytext=(x_text - 3, y_text - 0.03),  # Position of the text
    arrowprops=dict(facecolor="black", arrowstyle="->"),  # Arrow style
    fontsize=14,  # Text font size
    color="black",  # Text color
)

fig.savefig("images/pareto_topk_time.png", bbox_inches="tight")
fig.savefig("images/pareto_topk_time.pdf", bbox_inches="tight")

# %%
df_sem = prepare_sem_df(df, "peak_fit")

xerr = df_sem["sem_peak_fit"].to_numpy()
yerr = df_sem["sem_pred"].to_numpy()

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(5, 3.5), layout="constrained")
(h, l), _ = pareto_frontier_plot(
    df_pareto.to_pandas(),
    x_var="peak_fit",
    y_var="prediction_metric",
    hue_var="top_k",
    palette="tab10",
    hue_order=hue_order,
    ax=ax,
    ax_title="",
    ax_xlabel="",
)
ax.set_ylim([0.43, 0.55])
ax.set_ylabel("Prediction performance")
ax.set_xlabel("Peak RAM fit (MB)")

scatter = ax.collections[0]  # Get the first collection (scatter points)
colors = (
    scatter.get_facecolor()
)  # Get the facecolors (which represent the colors for each point)

for idx, c in enumerate(colors):
    ax.errorbar(
        x=df_pareto["peak_fit"][idx],
        y=df_pareto["prediction_metric"][idx],
        xerr=xerr[idx],
        yerr=yerr[idx],
        fmt="none",
        ecolor=c,
    )

ax.legend(
    h,
    l,
    title="Value of k",
    loc="upper right",
    bbox_to_anchor=(1.35, 1.05),
    frameon=False,
    title_fontsize="large",
)
fig.savefig("images/pareto_topk_ram.png", bbox_inches="tight")
fig.savefig("images/pareto_topk_ram.pdf", bbox_inches="tight")

# %%
