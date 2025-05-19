"""This script is used to prepare the Pareto plot that relates the prediction 
performance with the run time as the value of k (the number of candidates) increases. 

This version is animated for social media posting
"""

# %%
import os

os.chdir("../..")
# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import polars as pl

from src.utils import constants
from src.utils.plotting import pareto_frontier_plot

plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

def major_time_formatter(x, pos):
    # return f"{x/60:.0f}min"
    if x > 60:
        return f"{x/60:.0f}m{x%60:.0f}s"
    else:
        return f"{x:.1f}s"

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
df = pl.read_csv("results/results_topk.csv")
df = df.group_by(constants.GROUPING_KEYS + ["top_k"]).agg(
    pl.mean("time_run", "peak_fit", "prediction_metric")
)
df_pareto = (
    df.group_by("top_k")
    .agg(pl.mean("time_run"), pl.mean("prediction_metric"), pl.mean("peak_fit"))
    .sort("top_k")
)
hue_order = df_pareto["top_k"].to_numpy()

_x, _y = df_pareto.filter(top_k=30).select("time_run", "prediction_metric")

frame_annotation = df_pareto.with_row_index().filter(pl.col("top_k") >=30)["index"][0]
x_text = _x.item()
y_text = _y.item()

df_sem = prepare_sem_df(df, "time_run")

xerr = df_sem["sem_time_run"].to_numpy()
yerr = df_sem["sem_pred"].to_numpy()
df_pareto = df_pareto.to_pandas()
n_frames = len(df_pareto)

#%%
fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(7, 3.5), layout="constrained")
def update(frame): 
    ax.clear()
    df_frame = df_pareto[:frame]
    (h, l), _ = pareto_frontier_plot(
        df_frame,
        x_var="time_run",
        y_var="prediction_metric",
        hue_var="top_k",
        palette="tab10",
        hue_order=hue_order,
        ax=ax,
        ax_title="",
        ax_xlabel="",
        alpha=1
    )
    ax.set_ylim([0.43, 0.55])
    ax.set_ylabel("Prediction performance")
    ax.set_xlabel("Time run (s)")
    ax.set_title("Performance against runtime")

    if frame >= frame_annotation:
        ax.annotate(
            "Candidates\nused in\nexperiments",  # Annotation text
            xy=(x_text, y_text),  # Point to annotate
            xytext=(0.4, 0.3),  # Position of the text
            textcoords="axes fraction",
            arrowprops=dict(facecolor="black", arrowstyle="->"),  # Arrow style
            fontsize=14,  # Text font size
            color="black",  # Text color,
            ha="center"
        )
        
    if frame == n_frames - 1:
        ax.annotate(
            "Performance\nplateaus\nwith many\ncandidates",  # Annotation text
            xy=(x_text, y_text),  # Point to annotate
            xytext=(0.8, 0.2),  # Position of the text
            textcoords="axes fraction",
            fontsize=14,  # Text font size
            color="red",  # Text color,
            ha="center"
        )
    
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
        title="# of candidates",
        loc="upper right",
        bbox_to_anchor=(1.35, 1.05),
        frameon=False,
        title_fontsize="large",
    )
    ax.xaxis.set_major_formatter(major_time_formatter)
    return ax

ani = FuncAnimation(
    fig=fig,
    func=update,
    frames=range(1, len(df_pareto)),
    interval=5000,# in ms
    repeat=False
)
ani.save('images/pareto.gif',writer=FFMpegWriter(fps=2, extra_args=["-loop", "-1"]),)
# ani.save('pareto.gif', fps=2, extra_args=["-loop", "1"])

# %%

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(7, 3.5), layout="constrained")
ax.clear()
df_frame = df_pareto
(h, l), _ = pareto_frontier_plot(
    df_frame,
    x_var="time_run",
    y_var="prediction_metric",
    hue_var="top_k",
    palette="tab10",
    hue_order=hue_order,
    ax=ax,
    ax_title="",
    ax_xlabel="",
    alpha=1
)
ax.set_ylim([0.43, 0.55])
ax.set_ylabel("Prediction performance")
ax.set_xlabel("Time run (s)")
ax.set_title("Performance against runtime")

ax.annotate(
    "Candidates\nused in\nexperiments",  # Annotation text
    xy=(x_text, y_text),  # Point to annotate
    xytext=(0.4, 0.3),  # Position of the text
    textcoords="axes fraction",
    arrowprops=dict(facecolor="black", arrowstyle="->"),  # Arrow style
    fontsize=14,  # Text font size
    color="black",  # Text color,
    ha="center"
)
    
ax.annotate(
    "Performance\nplateaus\nwith many\ncandidates",  # Annotation text
    xy=(x_text, y_text),  # Point to annotate
    xytext=(0.8, 0.2),  # Position of the text
    textcoords="axes fraction",
    fontsize=14,  # Text font size
    color="red",  # Text color,
    ha="center"
)

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
    title="# of candidates",
    loc="upper right",
    bbox_to_anchor=(1.35, 1.05),
    frameon=False,
    title_fontsize="large",
)
ax.xaxis.set_major_formatter(major_time_formatter)

fig.savefig("images/pareto_social.png", bbox_inches="tight")
# %%
