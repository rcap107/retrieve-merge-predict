# %% This script is used to prepare two plots that report all combinations of variables with each other
# This allows readers to see how different configurations are more or less resilient
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.constants import LABEL_MAPPING, COMPACT_LABEL_MAPPING

# %%
sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
# %%
df_aggregation = pl.read_parquet("results/results_aggregation.parquet")
df_general = pl.read_parquet("results/results_general.parquet")
df_retrieval = pl.read_parquet("results/results_retrieval.parquet")
df_master = pl.read_parquet("results/master_list.parquet")

# %%
df = df_retrieval.to_pandas()
df["case"] = df["base_table"] + "-" + df["target_dl"]


# %%
def get_label(variable, value):
    return (
        f"{LABEL_MAPPING[variable][value]} ({COMPACT_LABEL_MAPPING[variable][value]})"
    )


# %%
fig, axs = plt.subplots(
    5,
    5,
    figsize=(15, 15),
    sharex=True,
    sharey="row",
    layout="constrained",
    # gridspec_kw={"hspace": 0.15},
)
ncols = 2


variables = ["chosen_model", "jd_method", "estimator", "target_dl", "base_table"]

for idx_row, var_1 in enumerate(variables):
    for idx_col, var_2 in enumerate(variables):
        ax = axs[idx_row, idx_col]
        ax.axvspan(0, -0.5, zorder=0, alpha=0.05, color="red")
        ax.set_xlim([-0.2, 1.1])

        if var_1 == var_2:
            sns.boxplot(data=df, x="prediction_metric", y=var_1, ax=ax, legend="full")
        else:
            sns.boxplot(data=df, x="prediction_metric", y=var_1, hue=var_2, ax=ax)

        h, l = ax.get_legend_handles_labels()
        l = [COMPACT_LABEL_MAPPING[var_2][_] for _ in l]

        # Needed to plot the final legend
        if idx_row == 3 and idx_col == 4:
            fallback_h, l = ax.get_legend_handles_labels()
            fallback_l = [COMPACT_LABEL_MAPPING[var_2][_] for _ in l]

        if idx_row == 4:
            if idx_col == 4:
                h, l = fallback_h, fallback_l
            ax.set_xlabel("")
            leg = ax.legend(
                h,
                l,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                title=LABEL_MAPPING["variables"][var_2],
                # title=var_2,
                ncols=ncols,
                mode="expand",
                edgecolor="white",
                # columnspacing=20
            )
            leg.set_in_layout(True)

        else:
            ax.legend().remove()

        if idx_col == 0:
            yticks = ax.get_yticks()
            yticklabels = [_.get_text() for _ in ax.get_yticklabels()]
            ax.set_yticks(yticks, [get_label(var_1, _) for _ in yticklabels])
            ax.set_ylabel(LABEL_MAPPING["variables"][var_1])
    print(var_1)

fig.align_ylabels()

# %%
fig.savefig(
    "images/all_combinations_retrieval.png", bbox_inches="tight", pad_inches=1.2
)
fig.savefig(
    "images/all_combinations_retrieval.pdf", bbox_inches="tight", pad_inches=1.2
)

# %%
