import pickle
from pathlib import Path

from src.data_structures.indices import ManualIndex, MinHashIndex


def get_candidates(query_table, query_column, indices):
    """Given query table and column, query the required indices and produce the
    candidates. Used for debugging.

    Args:
        query_table (_type_): _description_
        query_column (_type_): _description_
        indices (_type_): _description_
    """
    pass


def write_candidates_on_file(candidates, output_file_path, separator=","):
    with open(output_file_path, "w") as fp:
        fp.write("tbl_pth1,tbl1,col1,tbl_pth2,tbl2,col2\n")

        for key, cand in candidates.items():
            rstr = cand.get_joinpath_str(sep=separator)
            fp.write(rstr + "\n")

    # open output file

    # write the candidates

    # metam format is left_table;left_on_column;right_table;right_on_column


def fix_path(path):
    part_list = []
    for part in Path(path).parts:

        if part == "store":
            part_list.append("store3")
        elif part == "yago3-dl":
            part_list.append("yadl")
        elif part == "work":
            continue
        else:
            part_list.append(part)

    return Path(*part_list)
