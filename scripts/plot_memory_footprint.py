# #%%
# %cd ~/bench
#%%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib import ticker

from src.utils.constants import LABEL_MAPPING

sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

#%%
df_raw = pl.read_parquet("results/overall/wordnet_general_first.parquet")

df_mem = df_raw.select(
    pl.col("estimator"),
    pl.col("chosen_model"),
    pl.col("peak_fit"),
    # pl.col("peak_predict"),
    # pl.col("peak_test"),
).melt(id_vars=["estimator", "chosen_model"])

#%%
mapping = {"peak_fit": "Fit", "peak_predict": "Predict"}
fig, ax = plt.subplots(figsize=(6, 1.5), layout="constrained")
sns.boxplot(
    data=df_mem.to_pandas(),
    x="value",
    y="chosen_model",
    hue="variable",
    palette="Set1",
    ax=ax,
    dodge=0.15,
)
ax.set_xlabel("Peak Memory (MiB)")
ax.set_ylabel(None)
# ax.set_ylabel("Estimator")
# h, l = ax.get_legend_handles_labels()
ax.get_legend().remove()
ax.set_yticklabels(
    [LABEL_MAPPING["chosen_model"][x.get_text()] for x in ax.get_yticklabels()]
)
# fig.legend(h, [mapping[x] for x in l], title="Operation", loc="upper right")
fig.savefig("images/memory_usage.pdf")

# %%
