"""
Main entrypoint for the benchmarking code.
"""

import argparse
import json
import os
import pickle
import pprint
from collections import deque
from datetime import datetime as dt
from pathlib import Path
from pprint import pformat

import submitit
import toml
from joblib import Memory
from tqdm import tqdm

from src.utils.notifications import get_apobj

mem = Memory(location="__cache__", verbose=0)


# Fixing the number of polars threads for better reproducibility.
os.environ["POLARS_MAX_THREADS"] = "32"
from src.pipeline import prepare_config_dict, prepare_specific_configs, single_run
from src.utils.logging import archive_experiment, get_exp_name, setup_run_logging


def parse_args():
    """Parse arguments on the command line.

    Returns:
        argparse.Namespace: Arguments parsed on the command line.
    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--input_path",
        action="store",
        default=None,
        help="Path of the config file to be used.",
        type=argparse.FileType("r"),
    )

    group.add_argument(
        "--recovery_path",
        action="store",
        default=None,
        help="Path of the experiment to recover",
        type=Path,
    )

    group.add_argument(
        "--selected_config",
        action="store",
        default=None,
        help="Path to the pickle of specific configurations to run.",
        type=Path,
    )

    parser.add_argument(
        "-a",
        "--archive",
        required=False,
        action="store_true",
        help="If specified, archive the current run.",
    )

    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        help="If specified, skip writing logging.",
    )

    parser.add_argument(
        "--gpu", action="store_true", help="Whether or not to run computation on a GPU."
    )
    parser.add_argument(
        "--n-cpus",
        "-w",
        type=int,
        default=32,
        help="Number of CPUs per run of run_one.",
    )

    parser.add_argument(
        "--n-tasks",
        "-n",
        type=int,
        default=10,
        help="Number of concurrent tasks to run. ",
    )

    return parser.parse_args()


@mem.cache
def run_one(parameters, gpu=False):
    # Setup device to use GPU or not
    if gpu and torch.cuda.is_available():
        device = "cuda"
    elif gpu:
        raise RuntimeError("requested GPU run but cuda is not available.")
    else:
        device = "cpu"
    print("Using device:", device)

    # do computation and save as file.
    result = 42

    return result


def get_executor_marg(
    job_name, timeout_hour=60, n_cpus=10, max_parallel_tasks=256, gpu=False
):
    """Return a submitit executor to launch various tasks on a SLURM cluster.

    Parameters
    ----------
    job_name: str
        Name of the tasks that will be run. It will be used to create an output
        directory and display task info in squeue.
    timeout_hour: int
        Maximal number of hours the task will run before being interupted.
    n_cpus: int
        Number of CPUs requested for each task.
    max_parallel_tasks: int
        Maximal number of tasks that will run at once. This can be used to
        limit the total amount of the cluster used by a script.
    gpu: bool
        If set to True, require one GPU per task.
    """

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f"{timeout_hour}:00:00",
        array_parallelism=max_parallel_tasks,
        slurm_additional_parameters={
            "ntasks": 1,
            "cpus-per-task": n_cpus,
            "distribution": "block:block",
        },
    )
    if gpu:
        executor.update_parameters(
            slurm_gres=f"gpu:1",
            slurm_setup=[
                "#SBATCH -p parietal,gpu",
                # "#SBATCH -p parietal,gpu,gpu-best",
            ],
        )
    return executor


if __name__ == "__main__":
    apobj = get_apobj()

    args = parse_args()
    os.makedirs("results/logs", exist_ok=True)

    start_run = dt.now()
    if args.recovery_path is not None:
        # If args.recovery_path is provided, the script will look for the
        # missing_runs.pickle file in the given path and
        # try to reboot a run from there.
        if args.recovery_path.exists():
            pth = args.recovery_path
            missing_runs_path = Path(pth, "missing_runs.pickle")
            missing_runs_config = Path(pth, pth.stem + ".cfg")
            with open(missing_runs_config, "r") as fp:
                base_config = json.load(fp)
            run_variants = pickle.load(open(missing_runs_path, "rb"))
        else:
            raise IOError(f"File {args.recovery_path} not found.")
    elif args.selected_config is not None:
        # Using a specific set of configurations
        if args.selected_config.exists():
            run_variants = prepare_specific_configs(args.selected_config)
            base_config = run_variants
    else:
        # No recovery, simply read a toml file from the given input path.
        base_config = toml.load(args.input_path)
        run_variants = prepare_config_dict(base_config, args.debug)

    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name(debug=args.debug)

    pth_missing_runs = Path("results/logs/", exp_name, "missing_runs.pickle")

    # Submit one task per set of parameters
    executor = get_executor_marg(
        "retrieve-merge-predict",
        timeout_hour=24 * 6 + 12,
        n_cpus=args.n_cpus,
        max_parallel_tasks=args.n_tasks,
        gpu=args.gpu,
    )

    print(f"Exp Name: {exp_name}")
    # Run the computation on SLURM cluster with `submitit`
    print("Submitting jobs...", end="", flush=True)
    msg = f"""
    Submitted job `{exp_name}`

    ---

    {pformat(base_config)}

    """
    apobj.notify(msg)
    with executor.batch():
        tasks = [
            executor.submit(
                single_run, **{"run_config": parameters, "run_name": exp_name}
            )
            for parameters in run_variants
        ]

    end_run = dt.now()
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
    apobj.notify(f"Completed job.\n\nExp Name: {exp_name}")
