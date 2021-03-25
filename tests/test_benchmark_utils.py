"""
Test benchmark_utils.py
"""
import tempfile
from os.path import join
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest
import pandas as pd
from pipelinebenchmarker.benchmark_utils import PipelineBenchmarker


@pytest.fixture(scope="module")
@patch("azureml.core.Run")
@patch("azureml.core.Run.get_context")
@patch("azureml.core.compute.ComputeTarget")
def mock_benchmarker(mock_compute_target, mock_run_get_context, mock_run):
    """mock azure ml calls in PipelineBenchmarker class

    Parameters
    ----------
    mock_compute_target : object
        [mocked object]
    mock_run_get_context : object
        [mocked object]
    mock_run : object
        [mocked object]
    """
    mock_run.return_value = MagicMock()
    mock_run_get_context.return_value = MagicMock()
    mock_compute_target.return_value = MagicMock()

    benchmarker = PipelineBenchmarker()
    return benchmarker


def test_save_step_benchmark(mock_benchmarker):
    """test save_step_benchmark

    Parameters
    ----------
    mock_benchmarker : PipelineBenchmarker
        Mocked PipelineBenchmarker object
    """
    with tempfile.TemporaryDirectory() as tempdir:
        # Arrange
        step_name = "step_1"
        experiment_output_dir = tempdir
        new_benchmark = {"time_to_complete": ["00:13:44.920"], "n_nodes": [2]}

        # Act
        with patch.object(
            mock_benchmarker,
            "_PipelineBenchmarker__get_n_nodes",
            return_value=(2, "namehere"),
        ):
            mock_benchmarker.save_step_benchmark(
                step_name, experiment_output_dir, new_benchmark
            )

        # Assert
        assert Path(join(experiment_output_dir, f"benchmark_{step_name}.csv")).exists


@pytest.mark.skip(reason="cannot reasonably test outside of pipeline run")
def test_get_n_nodes():
    """test get_n_nodes"""
    pytest.skip("cannot reasonably test outside of pipeline run")


@pytest.mark.skip(reason="cannot reasonably test outside of pipeline run")
def test_get_compute_config():
    """test get_compute_config"""
    pytest.skip("cannot reasonably test outside of pipeline run")


@pytest.mark.skip(reason="cannot reasonably test outside of pipeline run")
def test_get_step_benchmark():
    """test get_step_benchmark"""
    pytest.skip("cannot reasonably test outside of pipeline run")


def test_get_file_size_and_count(mock_benchmarker):
    """test get_file_size_and_count

    Parameters
    ----------
    mock_benchmarker : PipelineBenchmarker
        Mocked PipelineBenchmarker object
    """
    with tempfile.TemporaryDirectory() as tempdir:
        # Arrange
        file_extension = ".txt"
        filenames = [
            f"temp1{file_extension}",
            f"temp2{file_extension}",
            f"temp3{file_extension}",
        ]
        file_sizes = [1073741824, 2192834101, 567333962]
        expected_total_size = sum(file_sizes) / 1000
        for filename, file_size in zip(filenames, file_sizes):
            with open(join(tempdir, filename), "wb") as test_file:
                test_file.seek(file_size - 1)
                test_file.write(b"\0")
                test_file.close()

        # Act
        mock_benchmarker.blob_storage_mounted = True
        test_size, test_count = mock_benchmarker.get_file_size_and_count(
            tempdir, file_extension
        )

        # Assert
        assert test_count == len(filenames)
        assert test_size == expected_total_size


def test_update_benchmark_table_exist_true(mock_benchmarker):
    """test update_benchmark_table

    Parameters
    -------
    mock_benchmarker : PipelineBenchmarker
        Mocked PipelineBenchmarker object
    """
    report_exists = True
    with tempfile.TemporaryDirectory() as tempdir:
        # Arrange
        test_new_benchmark = {
            "datetime": "2021-02-19 21:16:43",
            "run_id": "run-2",
            "pipeline_duration": "00:02:43.684",
            "compute_config": {
                "name": "benchmark",
                "vm_size": "STANDARD_DS_V2",
                "vm_priority": "dedicated",
                "region": "westus",
            },
            "steps": {
                "prepare_data": {
                    "total_size_of_input": 0,
                    "n_files": 0,
                    "duration": "00:00:01.560",
                    "n_nodes": 1,
                    "compute_name": "benchmark",
                    "pipeline_step": "prepare_data",
                },
                "process_data": {
                    "duration": "00:00:01.875",
                    "n_nodes": 1,
                    "compute_name": "benchmark",
                    "pipeline_step": "process_data",
                },
            },
            "max_nodes_used": 1,
        }

        table_filepath = join(tempdir, "existing_report.csv")
        if report_exists:
            # create initial report
            existing_report = test_new_benchmark.copy()
            existing_report["run_id"] = "run-1"
            mock_benchmarker.pipeline_benchmarks = existing_report
            mock_benchmarker.update_benchmark_table(table_filepath)

        # Act
        mock_benchmarker.pipeline_benchmarks = test_new_benchmark
        mock_benchmarker.update_benchmark_table(table_filepath)
        df_test = pd.read_csv(table_filepath)

        # Assert
        if report_exists:
            assert len(df_test) == 4
            assert "run-1" in df_test.run_id.values
            assert "run-2" in df_test.run_id.values
        else:
            assert len(df_test) == 2
            assert "run-1" not in df_test.run_id.values
            assert "run-2" in df_test.run_id.values


def test_update_benchmark_table_exist_false(mock_benchmarker):
    """test update_benchmark_table

    Parameters
    -------
    mock_benchmarker : PipelineBenchmarker
        Mocked PipelineBenchmarker object
    """
    report_exists = False
    with tempfile.TemporaryDirectory() as tempdir:
        # Arrange
        test_new_benchmark = {
            "datetime": "2021-02-19 21:16:43",
            "run_id": "run-2",
            "pipeline_duration": "00:02:43.684",
            "compute_config": {
                "name": "benchmark",
                "vm_size": "STANDARD_DS_V2",
                "vm_priority": "dedicated",
                "region": "westus",
            },
            "steps": {
                "prepare_data": {
                    "total_size_of_input": 0,
                    "n_files": 0,
                    "duration": "00:00:01.560",
                    "n_nodes": 1,
                    "compute_name": "benchmark",
                    "pipeline_step": "prepare_data",
                },
                "process_data": {
                    "duration": "00:00:01.875",
                    "n_nodes": 1,
                    "compute_name": "benchmark",
                    "pipeline_step": "process_data",
                },
            },
            "max_nodes_used": 1,
        }

        table_filepath = join(tempdir, "existing_report.csv")
        if report_exists:
            # create initial report
            existing_report = test_new_benchmark.copy()
            existing_report["run_id"] = "run-1"
            mock_benchmarker.pipeline_benchmarks = existing_report
            mock_benchmarker.update_benchmark_table(table_filepath)

        # Act
        mock_benchmarker.pipeline_benchmarks = test_new_benchmark
        mock_benchmarker.update_benchmark_table(table_filepath)
        df_test = pd.read_csv(table_filepath)

        # Assert
        if report_exists:
            assert len(df_test) == 4
            assert "run-1" in df_test.run_id.values
            assert "run-2" in df_test.run_id.values
        else:
            assert len(df_test) == 2
            assert "run-1" not in df_test.run_id.values
            assert "run-2" in df_test.run_id.values
