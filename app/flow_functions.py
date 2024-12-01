from argparse import ArgumentParser, Namespace
from typing import List, Optional, Union

import yaml
from numpy import ndarray
from pydantic import ValidationError

from app.clustering import CDBSCAN, CAgglomerativeClustering, CKMeans
from app.utils.io_handler import JSONHandler, NumpyHandler
from app.utils.validations import Config


def parse_arguments() -> "Namespace":
    """
    Parse command-line arguments for configuration, input, and output paths.

    Returns:
        Namespace: Parsed arguments including paths for
        config, input, output, and sample weights.

    Example:
        ```python
        args = parse_arguments()
        print(args.config)  # Path to config.yaml
        ```
    """
    parser = ArgumentParser(description="Data Clustering Tool")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--input", required=True, help="Path to input data file"
    )
    parser.add_argument("--output", required=True, help="Path to save results")
    parser.add_argument("--sample-weight", help="Optional sample weight file")
    return parser.parse_args()


def load_and_validate_config(config_path: str) -> Config:
    """
    Load and validate YAML configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Config: Validated configuration object.

    Raises:
        ValidationError: If the configuration is invalid.

    Example:
        ```python
        config = load_and_validate_config("config.yaml")
        print(config.clustering.algorithm)
        ```
    """
    try:
        with open(config_path, "r") as file:
            raw_config = yaml.safe_load(file)
        return Config(**raw_config)
    except ValidationError as e:
        raise ValidationError(f"Invalid configuration: {e}")


def load_data(
    input_handler: Union[JSONHandler, NumpyHandler],
    input_path: str,
    sample_weight_path: Optional[str],
) -> tuple[Optional[ndarray], Optional[ndarray]]:
    """
    Load input data and optional sample weights using the given handler.

    Args:
        input_handler (Union[JSONHandler, NumpyHandler]): Handler to load data.
        input_path (str): Path to input data file.
        sample_weight_path (Optional[str]): Path to sample weight file.

    Returns:
        tuple: Loaded training data and sample weights.

    Example:
        ```python
        train_data, weights = load_data(handler, "data.json", "weights.json")
        ```
    """
    try:
        train_data = input_handler.load_data(input_path)
        train_sample_weight = (
            input_handler.load_data(sample_weight_path)
            if sample_weight_path
            else None
        )
        return train_data, train_sample_weight
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def validate_sample_weights(
    train_data: ndarray, train_sample_weight: Optional[List[float]]
) -> None:
    """
    Ensure sample weights match the number of data points.

    Args:
        train_data (ndarray): Training data.
        train_sample_weight (Optional[List[float]]): Sample weights.

    Raises:
        ValueError: If weights and data points mismatch.

    Example:
        ```python
        validate_sample_weights(train_data, sample_weights)
        ```
    """
    if (
        train_sample_weight is not None
        and len(train_sample_weight) != train_data.shape[0]
    ):
        raise ValueError(
            f"""
            Sample weights ({len(train_sample_weight)})
            do not match data points ({train_data.shape[0]}).
            """
        )


def execute_clustering(
    clustering_algorithm: Union[CKMeans, CDBSCAN, CAgglomerativeClustering],
    train_data: ndarray,
    train_sample_weight: Optional[ndarray],
) -> ndarray:
    """
    Apply a clustering algorithm to the training data.

    Args:
        clustering_algorithm
        (Union[CKMeans, CDBSCAN, CAgglomerativeClustering]):
        Clustering algorithm instance.
        train_data (ndarray): Data to cluster.
        train_sample_weight (Optional[ndarray]): Sample weights.

    Returns:
        ndarray: Clustering results.

    Raises:
        Exception: If clustering fails.

    Example:
        ```python
        result = execute_clustering(algorithm, train_data, sample_weights)
        print(result)
        ```
    """
    try:
        tsw_converted: Optional[List[float]] = (
            train_sample_weight.tolist()
            if train_sample_weight is not None
            else None
        )
        validate_sample_weights(train_data, tsw_converted)
        return clustering_algorithm.cluster(
            train_data, sample_weight=tsw_converted
        )
    except Exception as e:
        raise Exception(f"Clustering error: {e}")


def save_results(
    output_handler: Union[JSONHandler, NumpyHandler],
    result: Optional[ndarray],
    output_path: str,
) -> None:
    """
    Save clustering results to a specified file.

    Args:
        output_handler (Union[JSONHandler, NumpyHandler]):
        Handler to save data.
        result (Optional[ndarray]): Clustering results.
        output_path (str): Path to save the results.

    Example:
        ```python
        save_results(handler, result, "output.json")
        ```
    """
    try:
        if result is not None:
            output_handler.save_data(result, output_path)
    except Exception as e:
        raise Exception(f"Error saving results: {e}")
