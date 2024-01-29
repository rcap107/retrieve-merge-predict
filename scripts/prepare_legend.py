# %%
# %cd /Users/rcap/Projects/benchmark-join-suggestions/
# %%
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import src.utils.constants as constants
from src.utils.plotting import prepare_scatterplot_mapping_case
# %%
df = pl.read_parquet("results/overall_first.parquet")
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
tabs = df.select(pl.col("base_table").str.split("-").list.first().unique()).sort("base_table")["base_table"].to_numpy()
data_lakes = df.select(pl.col("target_dl").unique())["target_dl"].to_numpy()

# %%

# Data for the table
data = [[constants.LEGEND_LABELS[_], "•", "•", "•"] for _ in tabs]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(5,2), layout="constrained")

# Hide axes
ax.axis("off")

# Create the table
table = ax.table(
    cellText=data,
    cellLoc="center",
    bbox=[0,0,1,1], # Force the table bounding box to cover the entire figure
    colLabels=["Table"] + [constants.LABEL_MAPPING["target_dl"][_] for _ in data_lakes],
    colWidths=[0.5,0.25,.25,.25]
)
# Set the table properties
table.auto_set_font_size(False)
table.set_fontsize(12)

used = []
for rid, this_table in enumerate(tabs, start=1):
    for cid, this_data_lake in enumerate(data_lakes, start=1):
        cell = table[rid, cid]
        t = this_table.split("-")[0]
        this_case  = f"{t}-{this_data_lake}"
        color = scatterplot_mapping.get(this_case, "white")
        if this_case == "us_county_population-open_data_us":
            cell.set(hatch="/")
        cell.set_text_props(color=color)  # Set dot color
        cell.set_fontsize(20)
plt.show()

fig.savefig("images/legend.png")
fig.savefig("images/legend.pdf")

# %%
