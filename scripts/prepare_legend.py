#%%
import matplotlib.pyplot as plt
import polars as pl

# %%
df = pl.read_parquet("results/overall/overall_first.parquet")
# %%
df = df.with_columns(
    (
        pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
    ).alias("case")
)
# %%
cases = df.select(pl.col("case").unique()).to_numpy()
# %%
colors = plt.colormaps[colormap_name].resampled(len(scatterplot_labels)).colors
