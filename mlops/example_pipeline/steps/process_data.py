"""
Example pipeline step for processing data
"""
import argparse
from os.path import join
import os
from pipelinebenchmarker.benchmark_utils import PipelineBenchmarker


def process_prepared_data(
    root_dir: str,
    prepared_dir: str,
    results_dir: str,
) -> str:
    """Placeholder function for processing data

    Parameters
    ----------
    root_dir : str
        Mounted Azure blob storage path
    prepared_dir : str
        Directory containing prepared data
    results_dir : str
        Directory to store results
    """
    # -----------------------------------------------------------
    # Initialize the benchmarker at the start of the step
    # -----------------------------------------------------------
    benchmarker = PipelineBenchmarker(mounted_blob_dir=root_dir)

    # -----------------------------------------------------------
    # Placeholder code for data preparation
    # -----------------------------------------------------------
    with open(join(root_dir, results_dir, "temp.txt"), "w") as output_file:
        output_file.write("This is a test file")

    # -----------------------------------------------------------
    # End step by saving benchmark values
    # -----------------------------------------------------------
    experiment_output_dir = "output"
    os.makedirs(experiment_output_dir, exist_ok=True)
    benchmarker.save_step_benchmark(
        step_name="process_data", experiment_output_dir=experiment_output_dir
    )


def main():
    """Process data step placeholder"""

    parser = argparse.ArgumentParser(description="Data processing step")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root Directory. The mounted directory for azure blob storage",
    )
    parser.add_argument(
        "--prepared_dir", required=True, help="Directory to store prepared data"
    )
    parser.add_argument(
        "--results_dir", required=True, help="Directory to store results"
    )
    args, _ = parser.parse_known_args()

    process_prepared_data(
        root_dir=args.root_dir,
        prepared_dir=args.prepared_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
