"""
Figure 7: distribution of the containment of the top-100 candidates by JC.
"""
# %%
# cd ~/bench

import pickle

# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# %%
cfg = pl.Config()
cfg.set_fmt_str_lengths(150)
sns.set_context("talk")

# %%
def prepare_results(root_dir):
    list_res = []
    for _path in root_dir.iterdir():
        if not _path.stem.startswith("minhash_index") and _path.is_file():
            with open(_path, "rb") as fp:
                res = joblib.load(fp)
                d = res["counts"].with_columns(pl.lit(_path.stem).alias("base_table"))
                d = d.filter(~pl.col("base_table").str.contains("depleted"))
            list_res.append(d)
    df = pl.concat(list_res)
    top_res = []
    for gidx, group in df.filter(pl.col("containment") > 0).group_by(["base_table"]):
        new_g = group.with_columns(
            pl.col("base_table")
            .str.replace("-yadl_col_to_embed", "")
            .str.replace("em_index_", "")
            .alias("base_table")
        )
        top_res.append(
            new_g.top_k(by="containment", k=100).with_row_count("rank", offset=1)
        )
    df_top = pl.concat(top_res)
    return df_top


def plot_results(df, tag, ax):
    sns.lineplot(data=df, x="rank", y="containment", hue="base_table", ax=ax)

    ax.axvline(30, alpha=0.5)
    ax.annotate("30", xy=(30, 0), xytext=(30, -0.18), color="blue")
    ax.set_ylabel("Containment")
    ax.set_xlabel("")
    ax.set_title(tag)
    ax.get_legend().remove()
    return ax


# %%
root_dirs = list(
    map(
        Path,
        [
            "/home/soda/rcappuzz/work/benchmark-join-suggestions/data/metadata/_indices/wordnet_full",
            "data/metadata/_indices/binary_update",
            "/home/soda/rcappuzz/work/benchmark-join-suggestions/data/metadata/_indices/open_data_us",
        ],
    )
)

labels = ["Wordnet Full", "Binary", "Open Data US"]

fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, layout="constrained")

for idx, (r, l) in enumerate(zip(root_dirs, labels)):
    df_top = prepare_results(r)
    plot_results(df_top, l, ax=axs[idx])

# fig.supxlabel("")
fig.savefig("images/containment_datalake.pdf")
fig.savefig("images/containment_datalake.png")


# %%
