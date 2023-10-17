"""This script is used to batch convert tables in a dir from csv to parquet or viceversa.
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import polars as pl
import polars.selectors as cs
from clevercsv import Sniffer
from joblib import Parallel, delayed
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="w",
    filename="batch_convert_logger.txt",
    level=logging.DEBUG,
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        type=str,
        action="store",
        help="Input dir to browse.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        type=str,
        action="store",
        help="Dir where the converted files will be stored.",
    )

    parser.add_argument(
        "destination_format",
        type=str,
        action="store",
        choices=["csv", "parquet"],
        help="Destination format to use when converting.",
        default="parquet",
    )

    parser.add_argument(
        "--input_format",
        type=str,
        action="store",
        default="csv",
        choices=["csv", "parquet"],
        help="Input format to use when converting.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        action="store",
        default="1",
        help="Number of parallel jobs to run.",
    )

    args = parser.parse_args()
    return args


def convert_csv_to_parquet(table_path, dir_in, dir_out):
    dest_path = Path(*[p for p in table_path.parts if p not in dir_in.parts])
    # If destination path exists, skip table
    if dest_path.exists():
        return dest_path
    # Try to infer the dialect
    with open(table_path) as csvfile:
        try:
            dialect = Sniffer().sniff(csvfile.read(1024))
            has_header = Sniffer().has_header(sample=csvfile.read(1024))
            delimiter = dialect.delimiter
            # if delimiter not in [",", ";"]:
            #     print(Path(table_path).stem, delimiter)
        except:
            # logging.error("Parsing failed: Table %s" % table_path)
            return dest_path

    # Read the file
    # Try to save using the given output format, log errors
    try:
        df = pl.read_csv(
            table_path,
            infer_schema_length=10000,
            ignore_errors=False,
            has_header=has_header,
            separator=delimiter,
        )
        if len(df) == 0:
            raise ValueError
        df.columns = [f"c_{c_idx}_" + _.strip() for c_idx, _ in enumerate(df.columns)]
        os.makedirs(dest_path.parent, exist_ok=True)
        destination_path = Path(dir_out, dest_path).with_suffix(".parquet")
        df.with_columns(cs.string().str.strip()).write_parquet(destination_path)
        return None
    except (pl.ComputeError, ValueError):
        # logging.error("Conversion failed: Table %s" % table_path)
        return dest_path


def convert_in_parallel(dir_in: Path, dir_out: Path, input_format="csv", n_jobs=1):
    total_files = sum((1 for _ in dir_in.glob(f"*.{input_format}")))
    if total_files == 0:
        raise RuntimeError(
            f"No files with format {input_format} were found in {dir_in}."
        )

    r = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(convert_csv_to_parquet)(table_path, dir_in, dir_out)
        for idx, table_path in tqdm(
            enumerate(dir_in.glob(f"*.{input_format}")), total=total_files
        )
    )
    failed = [_ for _ in r if _ is not None]
    print(f"Total converted: {total_files - len(failed)}/{total_files}")
    with open("failed_tables.txt", "w") as fp:
        for dest_path in r:
            if dest_path is not None:
                fp.write(f"{dest_path}\n")


def convert_in_batch(
    dir_in: Path, dir_out: Path, input_format="csv", output_format="parquet"
):
    """Convert all files found in `dir_in` that have the given format `input_format`
    to the given `output_format` in `dir_out`.

    Args:
        dir_in (Path): Dir to search files to convert in.
        dir_out (Path): Dir where the converted files will be saved.
        input_format (str, optional): Format of the input files to convert. Defaults to "csv".
        output_format (str, optional): Format to be used when saving the files. Defaults to "parquet".

    Raises:
        RuntimeError: Raise RuntimeError if no files with the given input format are found.
    """
    total_files = sum((1 for _ in dir_in.glob(f"*.{input_format}")))
    if total_files == 0:
        raise RuntimeError(
            f"No files with format {input_format} were found in {dir_in}."
        )

    for ff in tqdm(dir_in.glob(f"*.{input_format}"), total=total_files):
        # Read the file
        if input_format == "csv":
            df = pl.read_csv(ff, infer_schema_length=None, ignore_errors=True)
        elif input_format == "parquet":
            df = pl.read_parquet(ff)
        else:
            raise NotImplementedError

        # Try to save using the given output format, log errors
        try:
            dest_path = Path(*[p for p in ff.parts if p not in dir_in.parts])
            os.makedirs(dest_path.parent, exist_ok=True)
            if output_format == "parquet":
                destination_path = Path(dir_out, dest_path).with_suffix(".parquet")
                df.write_parquet(destination_path)
            elif output_format == "csv":
                destination_path = Path(dir_out, dest_path).with_suffix(".csv")
                df.write_csv(destination_path)
            else:
                raise NotImplementedError
        except pl.ComputeError:
            logging.error("Failed: {}".format(ff))


def convert(args):
    dir_in = Path(args.input_folder)
    dir_out = Path(args.output_folder)
    if not dir_in.exists():
        raise FileNotFoundError(f"Input folder {args.input_folder} does not exist.")
    os.makedirs(args.output_folder, exist_ok=True)
    convert_in_parallel(dir_in, dir_out, n_jobs=args.n_jobs)

    # convert_in_batch(
    #     dir_in,
    #     dir_out,
    #     input_format=args.input_format,
    #     output_format=args.destination_format,
    # )


# %%
if __name__ == "__main__":
    args = parse_args()
    convert(args)
