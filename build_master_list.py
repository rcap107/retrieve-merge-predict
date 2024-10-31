# Build the master list with all the experiments
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.utils.logging import read_logs

# Putting together all relevant runs
run_ids = [
    "0428",
    "0429",
    "0430",
    "0453",
    "0454",
    "0459",
    "0457",
    "0467",
    "0468",
    "0476",
    "0477",
    "0478",
    "0481",
    "0482",
    "0483",
    "0485",
    "0471",
    "0484",
    "0486",
    "0487",
    "0494",
    "0495",
    "0496",
    "0497",
    "0501",
    "0500",
    "0502",
    "0503",
    "0635",
    "0636",
    "0637",
    "0638",
    "0665",
    "0671",
    "0672",
    "0673",
    "0674",
    "0680",
    "0682",
    "0683",
    "0686",
    "0688",
]
run_ids = sorted(list(set(run_ids)))

base_path = "results/logs/"
dest_path = Path("results/overall")
overall_list = []

for r_path in tqdm(
    Path(base_path).iterdir(), total=sum(1 for _ in Path(base_path).iterdir())
):
    r_id = str(r_path.stem).split("-")[0]
    if r_id in run_ids:
        try:
            df_raw = read_logs(exp_name=None, exp_path=r_path)
            if r_id == "0673":  # This run was made before fixing the model label
                df_raw = df_raw.with_columns(chosen_model=pl.lit("ridgecv"))
            overall_list.append(df_raw)
        except pl.exceptions.SchemaError:
            print("Failed ", r_path)

df_overall = pl.concat(overall_list).with_columns(
    source_table=pl.col("base_table").str.split("-").list.first()
)

df_overall.write_csv("results/master_list.csv")
