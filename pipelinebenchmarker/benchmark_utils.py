"""
A modular class designed for
benchmarking pipeline steps and overall run.
"""
import os
from os.path import join
from pathlib import Path
import tempfile
from typing import Tuple
import datetime
import warnings
import json
import pandas as pd
from azureml.core import Run
from azureml.core.compute import ComputeTarget
from pipelinebenchmarker.timestamp import format_seconds_to_timestamp


class PipelineBenchmarker:
    """PipelineBenchmarker class"""

    def __init__(self, mounted_blob_dir: str = "") -> None:

        """Constructs an instance of the PipelineBenchmarker class.

        Parameters
        ----------
        mounted_blob_dir : str
            (Optional) Path to mounted Azure blob storage
        """
        self.mounted_blob_dir = mounted_blob_dir
        if len(mounted_blob_dir) > 0:
            self.blob_storage_mounted = True
        else:
            self.blob_storage_mounted = False

        self.step_duration_start = datetime.datetime.utcnow()
        self.run = Run.get_context()
        self.parent_run = Run.get_context().parent
        self.step_details = self.run.get_details()
        self.pipeline_details = self.parent_run.get_details()
        self.workspace = self.run.experiment.workspace
        self.pipeline_benchmarks = {}

    def __get_n_nodes(self) -> Tuple[int, str]:
        """Gets the number of active nodes in AML compute.

        Returns
        -------
        n_nodes : int
            Number of currently active nodes
        compute_name : str
            Name of the compute cluster
        """
        # Get workspace info
        compute_name = self.step_details["target"]

        # Retrieve relevant compute from current workspace
        aml_compute = ComputeTarget(self.workspace, compute_name)
        n_nodes = int(aml_compute.get_status().current_node_count)

        return n_nodes, compute_name

    def __get_compute_config(self) -> Tuple[str, str, str, str]:
        """Retrieves the compute target's configuration.

        Returns
        -------
        compute_name : str
            Name of the compute cluster
        vm_size : str
            Size of the compute cluster (e.g., "STANDARD_DS_v2")
        vm_priority : str
            Priority of the compute target (dedicated or low priority)
        region : str
            Location of the compute cluster
        """
        # Get workspace info
        compute_name = self.step_details["target"]
        aml_compute = ComputeTarget(self.workspace, compute_name)

        # Retrieve properties
        vm_size = aml_compute.vm_size
        vm_priority = aml_compute.vm_priority
        region = aml_compute.cluster_location

        return compute_name, vm_size, vm_priority, region

    def __get_step_benchmark(
        self, experiment_output_dir: str, filename: str
    ) -> pd.DataFrame:
        """Gets the benchmark data from a step.

        Parameters
        ----------
        experiment_output_dir : str
            Name of output folder for the experiment
        filename : str
            Name of step benchmark file

        Returns
        -------
        df_metadata : pd.DataFrame
            Dataframe containing benchmark values from given step
        """
        with tempfile.NamedTemporaryFile(suffix=".csv") as filepath_temp:
            filepath_steps_metadata = join(experiment_output_dir, filename)
            run = Run.get_context()
            run.parent.download_file(
                filepath_steps_metadata,
                filepath_temp.name,
            )
            df_metadata = pd.read_csv(filepath_temp.name)
            return df_metadata

    def get_file_size_and_count(
        self, directory: str, file_extension: str
    ) -> Tuple[float, int]:
        """Gets the total size of all files in a Azure blob storage directory and the number of files.

        Parameters
        ----------
        directory : str
            Path to directory containing files (excluding path to mounted storage base directory)
        file_extension : str
            Only files with the specified file extension will be counted (e.g., ".mp4")

        Returns
        -------
        total_size : float
            Total size of all counted files, in kilobytes
        file_count : int
            Total number of files matching the specified file extension
        """
        if self.blob_storage_mounted:
            directory = join(self.mounted_blob_dir, directory)
            total_size = 0
            file_count = 0
            for filename in os.listdir(directory):
                if file_extension in filename[-len(file_extension) :]:
                    file_size = os.path.getsize(join(directory, filename)) / 1000
                    total_size += file_size
                    file_count += 1
        else:
            warnings.warn("PipelineBenchmarker does not have access to blob storage")
            total_size = 0
            file_count = 0
        return total_size, file_count

    def save_step_benchmark(
        self,
        step_name: str,
        experiment_output_dir: str,
        benchmark_dict: dict = {},
    ) -> None:
        """Save current step's benchmarking data.

        Parameters
        ----------
        step_name : str
            Name of the pipeline step (e.g., "extract_audio")
        experiment_output_dir : str
            Directory at pipeline experiment level to store output in AML (e.g., "output")
        benchmark_dict : dict
            (Optional) Dictionary containing additional metrics to save (should be flat, not nested).
            Duration, node count, and compute cluster name are automatically included so there is no need to include them in the input.
            Dictionary values should be contained within brackets (e.g., {"n_files": [34]}).
        """
        os.makedirs(experiment_output_dir, exist_ok=True)

        # Check if benchmark_dict is nested
        is_nested = any(isinstance(i, dict) for i in benchmark_dict.values())
        if is_nested:
            warnings.warn(
                "Step benchmark values may not save correctly in the final consolidation step if benchmark_dict is nested. Please flatten the dictionary."
            )

        # Add duration and node count to dictionary
        duration_in_sec = (
            datetime.datetime.utcnow() - self.step_duration_start
        ).total_seconds()
        benchmark_dict["duration"] = [format_seconds_to_timestamp(duration_in_sec)]
        n_nodes, compute_name = self.__get_n_nodes()
        benchmark_dict["n_nodes"] = [n_nodes]
        benchmark_dict["compute_name"] = [compute_name]
        benchmark_dict["pipeline_step"] = [step_name]

        # Convert dictionary to dataframe
        df_metadata = pd.DataFrame(benchmark_dict)
        filename = f"benchmark_{step_name}.csv"
        metadata_filepath = join(experiment_output_dir, filename)
        df_metadata.to_csv(metadata_filepath, header=True)

        # Write to pipeline run output such that it is accessible in the final benchmarking step
        self.run.parent.upload_folder(experiment_output_dir, experiment_output_dir)
        print(
            f"{filename} written to AML pipeline experiment output directory: {experiment_output_dir}"
        )

    def save_pipeline_benchmark(
        self, pipeline_steps: list, experiment_output_dir: str
    ) -> None:
        """Consolidates all step benchmarks into one overall pipeline benchmark json.
        Uploads final json to AML output and to Azure blob storage if mounted.

        Parameters
        ----------
        pipeline_steps : list
            List of step names as defined in the pipeline (e.g., ["extract_audio", "process_audio"])
        experiment_output_dir : str
            Directory at pipeline experiment level to store output in AML (e.g., "output")
        """
        os.makedirs(experiment_output_dir, exist_ok=True)

        # Initialize
        run_id = self.run.parent.id
        t_pipeline_start = datetime.datetime.strptime(
            self.pipeline_details["startTimeUtc"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        pipeline_duration = (
            datetime.datetime.utcnow() - t_pipeline_start
        ).total_seconds()
        compute_name, vm_size, vm_priority, region = self.__get_compute_config()

        pipeline_benchmarks = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "pipeline_duration": format_seconds_to_timestamp(pipeline_duration),
            "compute_config": {
                "name": compute_name,
                "vm_size": vm_size,
                "vm_priority": vm_priority,
                "region": region,
            },
            "steps": {},
        }
        self.pipeline_benchmarks = pipeline_benchmarks

        # Add in benchmarks from each step
        max_nodes_used = 0
        for step in pipeline_steps:
            df_step_benchmark = self.__get_step_benchmark(
                experiment_output_dir,
                f"benchmark_{step}.csv",
            )
            # Format into dictionary
            step_dictionary = df_step_benchmark.to_dict(orient="records")[0]
            step_dictionary.pop("Unnamed: 0")  # remove index
            pipeline_benchmarks["steps"][step] = step_dictionary

            if step_dictionary["n_nodes"] > max_nodes_used:
                max_nodes_used = step_dictionary["n_nodes"]

        pipeline_benchmarks["max_nodes_used"] = max_nodes_used

        # Write to AML output
        with open(join(experiment_output_dir, "benchmarks.json"), "w") as json_file:
            json.dump(pipeline_benchmarks, json_file)
        self.run.parent.upload_folder(experiment_output_dir, experiment_output_dir)
        print(
            f"benchmarks.json written to AML pipeline experiment output directory: {experiment_output_dir}"
        )

        # Write to mounted azure blob storage
        if self.blob_storage_mounted:
            folder_path = join(
                self.mounted_blob_dir, "pipeline_benchmarks", "individual_runs"
            )
            os.makedirs(folder_path, exist_ok=True)
            filename = f"benchmark_{run_id}.json"

            with open(join(folder_path, filename), "w") as output_file:
                json.dump(pipeline_benchmarks, output_file)
            print(f"{filename} written to {folder_path}")

    def update_benchmark_table(self, table_filepath: str) -> None:
        """Writes or updates a table storing all the benchmarking info.

        Parameters
        ----------
        table_filepath : str
            Path to the table which contains or will contain benchmarks for all runs. It is recommended this is a filepath to somewhere in mounted blob storage.
        """
        new_benchmark = self.pipeline_benchmarks

        pipeline_benchmarks = {
            "datetime": [new_benchmark["datetime"]],
            "run_id": [new_benchmark["run_id"]],
            "pipeline_duration": [new_benchmark["pipeline_duration"]],
            "compute_name": [new_benchmark["compute_config"]["name"]],
            "compute_size": [new_benchmark["compute_config"]["vm_size"]],
            "compute_priority": [new_benchmark["compute_config"]["vm_priority"]],
            "compute_region": [new_benchmark["compute_config"]["region"]],
        }

        df_append = pd.DataFrame()
        for step in new_benchmark["steps"].keys():
            step_benchmarks = pipeline_benchmarks.copy()
            step_benchmarks["step_name"] = [step]
            for step_val in new_benchmark["steps"][step].keys():
                if step_val not in [
                    "compute_name",
                    "compute_size",
                    "compute_priority",
                    "compute_region",
                    "pipeline_step",
                ]:
                    if step_val in ["duration", "n_nodes"]:
                        step_benchmarks[step_val] = new_benchmark["steps"][step][
                            step_val
                        ]
                    else:
                        # Preface specific (non-default) step metrics with step name
                        step_benchmarks[f"{step}_{step_val}"] = new_benchmark["steps"][
                            step
                        ][step_val]

            df_temp = pd.DataFrame(step_benchmarks)
            df_append = df_append.append(df_temp, ignore_index=True)

        # Check if file exists
        if Path(table_filepath).is_file():
            # file exists
            df_append.to_csv(table_filepath, mode="a", header=False, index=False)
        else:
            df_append.to_csv(table_filepath, header=True, index=False)
