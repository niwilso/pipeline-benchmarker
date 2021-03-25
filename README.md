# Pipeline Benchmarking

The `PipelineBenchmarker` module assists in automatically tracking benchmarking values of interest in any [Azure Machine Learning (AML) pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines). This includes compute metadata, step duration time, maximum number of nodes used, and any custom parameter values designated by the user.

This module is helpful in improving observability at the per-pipeline run basis and can be used for evaluating compute configuration trade-offs and for obtaining values for cost estimation (e.g., [Azure Cognitive Services](https://azure.microsoft.com/services/cognitive) consumption logging).

A corresponding blog post article covering how this module has been used in the real world will be available soon.

## Repo Overview

This directory contains code (`mlops` and `pipelinebenchmarker`) and documentation (`docs`) for benchmarking AML pipelines.

For an example of how to implement the `PipelineBenchmarker` class in an AML pipeline, please refer to the `mlops/example_pipeline` and `docs` directories.

## Testing

To run unit tests, execute the following line in this top-level directory.

```cmd
pytest tests
```
