"""
Example final benchmarking step
"""
import argparse
from os.path import join
from pipelinebenchmarker.benchmark_utils import PipelineBenchmarker


def benchmark_pipeline(
    root_dir: str,
) -> str:
    """Placeholder function for processing data

    Parameters
    ----------
    root_dir : str
        Mounted Azure blob storage path
    """
    # -----------------------------------------------------------
    # Initialize the benchmarker at the start of the step
    # -----------------------------------------------------------
    benchmarker = PipelineBenchmarker(mounted_blob_dir=root_dir)

    # -----------------------------------------------------------
    # Save entire pipeline benchmarks
    # (This will automatically pull benchmarks from previous steps)
    # -----------------------------------------------------------
    experiment_output_dir = "output"
    benchmarker.save_pipeline_benchmark(
        pipeline_steps=["prepare_data", "process_data"],
        experiment_output_dir=experiment_output_dir,
    )

    # -----------------------------------------------------------
    # Write or append to table keeping track of benchmarks per run
    # -----------------------------------------------------------
    aggregate_results_path = join(root_dir, "pipeline_benchmarks", "benchmarks.csv")
    benchmarker.update_benchmark_table(table_filepath=aggregate_results_path)


def main():
    """Benchmarking step placeholder"""

    parser = argparse.ArgumentParser(description="Pipeline benchmarking step")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root Directory. The mounted directory for azure blob storage",
    )
    args, _ = parser.parse_known_args()

    benchmark_pipeline(
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()
