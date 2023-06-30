"""Time and prepare minhash index with different parameters"""


import datetime
import logging
from pathlib import Path
from src.data_structures.indices import MinHashIndex

log_format = "%(message)s"

logger = logging.getLogger("metadata_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt=log_format)

fh = logging.FileHandler(filename="results/logging_minhash.log")
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)


def save_indices(index_dict, index_dir):
    """Save all the indices found in `index_dict` in separate pickle files, in the
    directory provided in `index_dir`.

    Args:
        index_dict (dict): Dictionary containing the indices.
        index_dir (str): Path where the dictionaries will be saved.
    """
    if Path(index_dir).exists():
        for index_name, index in index_dict.items():
            print(f"Saving index {index_name}")
            filename = f"{index_name}_index.pickle"
            fpath = Path(index_dir, filename)
            index.save_index(fpath)
    else:
        raise ValueError(f"Invalid `index_dir` {index_dir}")


indices = {}
index_dir = Path("data/metadata/_indices/testing_minhash/partitions")

n_perm = 64
for case in ["binary", "wordnet"]:
    # for case in ["binary"]:
    for n_part in [16, 32, 64]:
        metadata_dir = Path(f"data/metadata/{case}")
        mh_config = {
            "data_dir": metadata_dir,
            "thresholds": [10, 20, 80],
            "oneshot": True,
            "num_perm": n_perm,
            "num_part": n_part,
        }
        start_time = datetime.datetime.now()
        index = MinHashIndex(**mh_config)
        end_time = datetime.datetime.now()

        index_name = f"minhash_{case}_part_{n_part}"

        indices[index_name] = index

        log_str = f"{case},{mh_config['thresholds']},{mh_config['num_perm']},{n_part},{start_time},{end_time},{(end_time-start_time).total_seconds()}"

        logger.info("%s", log_str)


from src.pipeline import save_indices

save_indices(indices, index_dir)
