# %%
from pathlib import Path
import shutil

source_folder = Path("results/query_results/binary_update")
data_lakes = ["binary_update", "wordnet_full", "wordnet_vldb_10", "wordnet_vldb_50"]

for dl in data_lakes:
    source_folder = Path(f"results/query_results/{dl}")

    base_tables = [
        "company_employees-yadl-depleted",
        "housing_prices-yadl-depleted",
        "us_accidents_2021-yadl-depleted",
        "us_elections-yadl-depleted",
    ]

    for retrieval_method in ["exact_matching", "minhash", "starmie", "minhash_hybrid"]:
        files_to_check = [
            f"{source_folder.stem}__{retrieval_method}__{_}__col_to_embed.pickle"
            for _ in base_tables
        ]

        for f in Path(source_folder).iterdir():
            if f.name in files_to_check:
                # print(f)
                fixed_name = Path(source_folder, f.name.replace("-depleted", ""))
                if not fixed_name.exists():
                    shutil.copy2(f, fixed_name)
