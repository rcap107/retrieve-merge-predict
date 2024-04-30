"""
Figure 4: prediction performance by data lake.
"""
#%%
# %cd ~/bench

import matplotlib.pyplot as plt

#%%
import polars as pl
import seaborn as sns

sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
df_raw = pl.read_parquet("results/overall/overall_first.parquet")
df_raw = df_raw.filter(
    ~pl.col("target_dl").is_in(
        [
            "wordnet_vldb_50",
            "wordnet_vldb_3",
        ]
    )
)
# %%
fig, ax = plt.subplots(squeeze=True, figsize=(6, 2), layout="constrained")
sns.boxplot(data=df_raw.to_pandas(), x="r2score", y="target_dl", ax=ax)
ax.set_ylabel("")
ax.set_xlabel("Prediction performance")

mapping = {
    "wordnet_full": "YADL Wordnet",
    "binary_update": "YADL Binary",
    "open_data_us": "Open Data US",
    "wordnet_vldb_10": "YADL 10k",
}

ax.set_yticklabels([mapping[x.get_text()] for x in ax.get_yticklabels()])

# fig.savefig("images/performance_data_lakes.png")
# fig.savefig("images/performance_data_lakes.pdf")

# %%
