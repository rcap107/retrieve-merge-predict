#%%
# %cd ~/bench

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#%%
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

import src.utils.plotting as plotting

sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
df_raw = pl.read_parquet("results/overall/overall_first.parquet")
# %%
fig, ax = plt.subplots(squeeze=True, figsize=(6, 2), layout="constrained")
sns.boxplot(data=df_raw.to_pandas(), x="r2score", y="target_dl", ax=ax)
ax.set_ylabel("")
ax.set_xlabel("Prediction performance")

mapping = {
    "wordnet_full": "YADL Wordnet",
    "binary_update": "YADL Binary",
    "open_data_us": "Open Data US",
}

ax.set_yticklabels([mapping[x.get_text()] for x in ax.get_yticklabels()])

fig.savefig("images/performance_data_lakes.png")
fig.savefig("images/performance_data_lakes.pdf")

# %%
