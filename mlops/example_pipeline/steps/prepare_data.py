"""
Example pipeline step for preparing data
"""
import argparse
from os.path import join
import os
from pipelinebenchmarker.benchmark_utils import PipelineBenchmarker


def prepare_data_for_processing(
    root_dir: str,
    raw_dir: str,
    prepared_dir: str,
) -> str:
    """Placeholder function for preparing data for processing

    Parameters
    ----------
    root_dir : str
        Mounted Azure blob storage path
    raw_dir : str
        Directory containing raw data
    prepared_dir : str
        Directory to store prepared data
    """
    # -----------------------------------------------------------
    # Initialize the benchmarker at the start of the step
    # -----------------------------------------------------------
    benchmarker = PipelineBenchmarker(mounted_blob_dir=root_dir)

    # -----------------------------------------------------------
    # Placeholder code for data preparation
    # -----------------------------------------------------------
    total_size, file_count = benchmarker.get_file_size_and_count(
        directory=raw_dir, file_extension=".png"
    )

    with open(join(root_dir, prepared_dir, "temp.txt"), "w") as output_file:
        output_file.write("This is a test file")

    # Saving additional metrics is completely optional
    additional_metrics = {
        "total_size_of_input": [total_size],
        "n_files": [file_count],
    }

    # -----------------------------------------------------------
    # End step by saving benchmark values
    # -----------------------------------------------------------
    experiment_output_dir = "output"
    os.makedirs(experiment_output_dir, exist_ok=True)
    benchmarker.save_step_benchmark(
        step_name="prepare_data",
        experiment_output_dir=experiment_output_dir,
        benchmark_dict=additional_metrics,
    )


def main():
    """Prepare data step placeholder"""

    parser = argparse.ArgumentParser(description="Initial data preparation")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root Directory. The mounted directory for azure blob storage",
    )
    parser.add_argument(
        "--raw_dir", required=True, help="Directory containing raw data"
    )
    parser.add_argument(
        "--prepared_dir", required=True, help="Directory to store prepared data"
    )
    args, _ = parser.parse_known_args()

    prepare_data_for_processing(
        root_dir=args.root_dir, raw_dir=args.raw_dir, prepared_dir=args.prepared_dir
    )


if __name__ == "__main__":
    main()
