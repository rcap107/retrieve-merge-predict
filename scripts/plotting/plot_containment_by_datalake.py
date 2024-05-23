"""
This script is used to prepare figure 5(a) in the paper.
"""

# %%
# %cd ~/bench

# #%%
# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.utils.constants import LABEL_MAPPING

# %%
sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

STATS_DIR = Path("results/stats")

# %%
df_opendata = pl.read_csv(
    STATS_DIR / "analysis_query_results_open_data_us_stats_all.csv"
)
df_wordnet = pl.read_csv(
    STATS_DIR / "analysis_query_results_wordnet_full_stats_all.csv"
)
df_vldb_10 = pl.read_csv(
    STATS_DIR / "analysis_query_results_wordnet_vldb_10_stats_all.csv"
)
df = pl.concat(
    [
        df_wordnet,
        df_opendata,
        df_vldb_10,
    ]
).filter(pl.col("top_k") == 30)
df = df.with_columns(
    containment=(
        pl.when(pl.col("containment") > 1)
        .then(pl.col("containment") / pl.col("src_nrows"))
        .otherwise(pl.col("containment"))
    )
)
# %%
order = ["minhash", "minhash_hybrid", "exact_matching", "starmie"]
fig, ax = plt.subplots(squeeze=True, figsize=(4, 3), layout="constrained")
sns.boxplot(
    data=df.to_pandas(),
    x="containment",
    y="retrieval_method",
    hue="data_lake_version",
    ax=ax,
    order=order,
)

mapping = {
    "wordnet_full": "YADL Comb.",
    "open_data_us": "Open Data US",
    "wordnet_vldb_10": "YADL 10k",
}

h, l = ax.get_legend_handles_labels()
labels = [mapping[_] for _ in l]
ax.get_legend().remove()

fig.legend(
    h,
    labels,
    loc="upper left",
    fontsize=10,
    ncols=3,
    bbox_to_anchor=(0, 1.0, 1, 0.1),
    mode="expand",
)
ax.set_yticklabels(
    [LABEL_MAPPING["jd_method"][x.get_text()] for x in ax.get_yticklabels()]
)
ax.set_xlabel("Containment")
ax.set_ylabel("")
fig.savefig("images/containment-barplot-datalake.pdf", bbox_inches="tight")
fig.savefig("images/containment-barplot-datalake.png", bbox_inches="tight")

# %%
