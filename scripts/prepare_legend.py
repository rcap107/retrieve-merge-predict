# %%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from matplotlib.table import Table

import src.utils.constants as constants
from src.utils.logging import read_and_process, read_logs
from src.utils.plotting import prepare_scatterplot_mapping_case

# %%
result_path = "results/results_general.parquet"

df = pl.read_parquet(result_path)

# df = read_and_process(df_results)

# %%
df = df.with_columns(
    (
        pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
    ).alias("case")
)
# %%
cases = df.select(pl.col("case").unique()).sort("case")["case"]
# %%
d = (
    df.with_columns(pl.col("base_table").str.split("-").list.first())
    .group_by("target_dl")
    .agg(pl.n_unique("base_table"))
    .to_dict()
)
# number of colors by data lake
n_colors = dict(zip(*d.values()))
# %%
tabs = (
    df.select(pl.col("target_dl"), pl.col("case"))
    .unique()
    .sort("target_dl", "case")["case"]
    .to_numpy()
)

# %%
scatterplot_mapping = prepare_scatterplot_mapping_case(df)

# %%
tabs = (
    df.select(pl.col("base_table").str.split("-").list.first().unique())
    .sort("base_table")["base_table"]
    .to_numpy()
)
data_lakes = (
    df.select(pl.col("target_dl").unique()).sort("target_dl")["target_dl"].to_numpy()
)

# %%

# Data for the table
data = [["•"] * len(tabs) for _ in data_lakes]
# %%
# Create a figure and axis
fig, ax = plt.subplots(figsize=(16, 2), layout="constrained")

# Hide axes
ax.axis("off")

# Create the table
table = ax.table(
    cellText=data,
    cellLoc="center",
    bbox=[0, 0, 1, 1],  # Force the table bounding box to cover the entire figure
    rowLabels=[constants.LABEL_MAPPING["target_dl"][_] for _ in data_lakes],
    colLabels=[constants.LEGEND_LABELS[_] for _ in tabs],
    colWidths=[0.75] * len(tabs),
    colLoc="center",
    rowLoc="center",
    # row=[0.75] * len(tabs),
)
# Set the table properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 0.2)
# table.set_text_props(horizontalalignment="center")

used = []
for rid, this_data_lake in enumerate(data_lakes, start=1):
    for cid, this_table in enumerate(tabs, start=0):
        cell = table[rid, cid]
        t = this_table.split("-")[0]
        this_case = f"{t}-{this_data_lake}"
        print(this_case)
        if this_table == "us_accidents_large" and this_data_lake == "wordnet_vldb_50":
            print("aa")
        # color = scatterplot_mapping.get(this_case, "white")
        if this_case == "us_county_population-open_data_us" or this_case in [
            "schools-wordnet_full",
            "schools-binary_update",
            "schools-wordnet_vldb_10",
            "schools-wordnet_vldb_50",
        ]:
            cell.set(hatch="/")
            cell.set_text_props(text="")
        else:
            color = scatterplot_mapping[this_case]
            # color = scatterplot_mapping.get(this_case, "white")

        # cell.set_facecolor(color=color)
        cell.set_text_props(color=color)  # Set dot color
        cell.set_text_props(color=color)  # Set dot color
        cell.set_fontsize(40)
plt.show()

fig.savefig("images/legend.png")
fig.savefig("images/legend.pdf")

# %%
