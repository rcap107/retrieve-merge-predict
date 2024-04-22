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

#%%
sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")

DEFAULT_QUERY_RESULT_DIR = Path("results/query_results")


# %%
df_od = pl.read_csv("analysis_query_results_open_data_us-fixed.csv")
df_wn = pl.read_csv("analysis_query_results.csv")
df_od_hybrid = pl.read_csv("analysis_query_results_open_data_us_hybrid.csv")
df = pl.concat([df_wn, df_od, df_od_hybrid]).filter(pl.col("top_k") == 200)

# %%
order = ["minhash", "minhash_hybrid", "exact_matching"]
fig, ax = plt.subplots(squeeze=True, figsize=(4, 3), layout="constrained")
sns.boxplot(
    data=df.to_pandas(),
    x="containment",
    y="retrieval_method",
    hue="data_lake_version",
    ax=ax,
    order=order,
)

mapping = {"wordnet_full": "YADL Wordnet", "open_data_us": "Open Data US"}

h, l = ax.get_legend_handles_labels()
labels = [mapping[_] for _ in l]
ax.get_legend().remove()

fig.legend(
    h,
    labels,
    loc="upper left",
    fontsize=10,
    ncols=2,
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
