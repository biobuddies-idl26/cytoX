"""Combine per-task scGen results.csv files into one table.

Each run folder (e.g. data/scgen_runs_L4_H800_Z100) holds one task_XXX/
subfolder per (cytokine, held-out cell type) task, each with a results.csv.
This script stacks them and adds two columns:
    run   - the run folder name, e.g. scgen_runs_L4_H800_Z100
    task  - the task folder name, e.g. task_000

Usage (from the cytoX folder):
    python combine_results.py
    python combine_results.py --pattern "data/scgen_runs_*/task_*/results.csv" --out data/results_all.csv

Or import it:
    from combine_results import combine_results
    df = combine_results()
"""

import argparse
import glob
import os

import pandas as pd

DEFAULT_PATTERN = "scgen_runs_L*_H800_Z100/task_*/results.csv"
DEFAULT_OUT = "results_all.csv"


def find_result_files(pattern=DEFAULT_PATTERN, exclude=("old", "test")):
    """Return sorted results.csv paths matching `pattern`,
    skipping any run folder whose name contains a word in `exclude`."""
    files = sorted(glob.glob(pattern))
    return [f for f in files if not any(word in run_name(f) for word in exclude)]


def run_name(path):
    """Run folder name for a results.csv path, e.g. scgen_runs_L4_H800_Z100."""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def task_name(path):
    """Task folder name for a results.csv path, e.g. task_000."""
    return os.path.basename(os.path.dirname(path))


def read_one(path):
    """Read one results.csv and add the run and task columns at the front."""
    df = pd.read_csv(path)
    df.insert(0, "run", run_name(path))
    df.insert(1, "task", task_name(path))
    return df


def combine_results(pattern=DEFAULT_PATTERN, exclude=("old", "test")):
    """Stack every matching results.csv into one DataFrame, keeping all columns."""
    files = find_result_files(pattern, exclude)
    if not files:
        raise FileNotFoundError(f"No results.csv files match {pattern!r}. "
                                "Run this from the cytoX folder or pass --pattern.")
    return pd.concat([read_one(f) for f in files], ignore_index=True)


def summarize(df, expected_tasks=84):
    """Print task counts per run and flag runs that are incomplete."""
    counts = df.groupby("run")["task"].nunique()
    for run, n in counts.items():
        flag = "" if n == expected_tasks else f"  <-- expected {expected_tasks}"
        print(f"{run}: {n} tasks{flag}")
    print(f"total rows: {len(df)}, columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"glob for results.csv files (default: {DEFAULT_PATTERN})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where to save the combined CSV (default: {DEFAULT_OUT})")
    parser.add_argument("--expected-tasks", type=int, default=84)
    args = parser.parse_args()

    df = combine_results(args.pattern)
    summarize(df, args.expected_tasks)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
