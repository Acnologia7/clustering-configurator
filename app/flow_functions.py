from argparse import ArgumentParser, Namespace
from typing import Optional

import yaml
from numpy import ndarray

from app.dependencies import ClusteringContainer
from app.utils.io_handler import IOHandler
from app.utils.validations import Config


def parse_arguments() -> "Namespace":
    """
    Parse command-line arguments for the clustering tool.

    The function defines the required command-line arguments
    for running the clustering workflow,
    including paths for configuration, input data,
    output file, and an optional sample weight file.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Example Usage:
        ```python
        args = parse_arguments()
        print(args.config)  # Path to the config file
        print(args.input)   # Path to the input data file
        print(args.output)  # Path to save output results
        print(args.sample_weight)  # Optional sample weight file path
        ```
    """
    parser = ArgumentParser(description="Data Clustering Tool")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--input", required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save clustered output"
    )
    parser.add_argument(
        "--sample-weight",
        help="Path to sample weight file for training (optional)",
    )
    return parser.parse_args()


def load_and_validate_config(config_path: str) -> Config:
    """
    Load and validate a configuration from a YAML file.

    This function reads the YAML configuration file,
    parses it into a dictionary,
    and validates it against the `Config` schema.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: Validated configuration object.

    Raises:
        Exception: If the configuration file is missing or invalid.

    Example Usage:
        ```python
        config = load_and_validate_config("config.yaml")
        print(config.clustering.algorithm)
        # Access clustering algorithm details
        ```
    """
    try:
        with open(config_path, "r") as file:
            raw_config = yaml.safe_load(file)
        validated_config = Config(**raw_config)
        return validated_config
    except Exception as e:
        raise Exception(f"Configuration Validation Error: {e}")


def load_data(
    io_handler: IOHandler,
    input_path: str,
    sample_weight_path: Optional[str],
) -> tuple[Optional[ndarray], Optional[ndarray]]:
    """
    Load input data and optional sample weights.

    The function uses the provided IO handler to
    load training data from the specified input path.
    Optionally, it loads sample weight data if a weight file path is provided.

    Args:
        io_handler (IOHandler): Instance of an IO handler for data loading.
        input_path (str): Path to the input data file.
        sample_weight_path (Optional[str]):
        Path to the sample weight file (optional).

    Returns:
        tuple: A tuple containing:
            - Optional[ndarray]:
            Training data.
            - Optional[ndarray]:
            Sample weights, or None if no weight file is provided.

    Raises:
        Exception:
        If there is an error loading the input or sample weight data.

    Example Usage:
        ```python
        io_handler = IOHandler()
        train_data, train_sample_weight = load_data(
            io_handler, "input.json", "weights.json"
        )
        print(train_data)  # Loaded training data
        print(train_sample_weight)  # Loaded sample weights
        ```
    """
    try:
        train_data = io_handler.load_data(input_path)
        train_sample_weight = (
            io_handler.load_data(sample_weight_path)
            if sample_weight_path
            else None
        )
        return train_data, train_sample_weight
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def validate_sample_weights(
    train_data: ndarray, train_sample_weight: ndarray
) -> None:
    """
    Validate the length of
    sample weights against the number of training data points.

    Ensures that the number of
    sample weights matches the number of training data points.
    If there is a mismatch, raises a ValueError.

    Args:
        train_data (ndarray): Training data array.
        train_sample_weight (ndarray): Sample weight array.

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If the number of rows in `train_sample_weight`
        does not match the number of rows in `train_data`.

    Example Usage:
        ```python
        train_data = np.array([[1, 2], [3, 4]])
        sample_weights = [0.5, 0.5]
        validate_sample_weights(train_data, sample_weights)
        # Passes validation
        ```
    """
    if (
        train_sample_weight is not None
        and train_sample_weight.shape[0] != train_data.shape[0]
    ):
        raise ValueError(
            f"""
            Sample weight array size ({train_sample_weight.shape[0]})
            does not match the number
            of training data points ({train_data.shape[0]})
            """
        )


def execute_clustering(
    container: ClusteringContainer,
    train_data: ndarray,
    train_sample_weight: ndarray,
) -> ndarray:
    """
    Perform clustering using the specified algorithm.

    This function validates the sample weights and
    applies the clustering algorithm from
    the provided container to the training data.
    It returns the clustering results.

    Args:
        container (ClusteringContainer):
        Container holding the clustering algorithm.
        train_data (ndarray): Training data array.
        train_sample_weight (ndarray): Sample weight array.

    Returns:
        ndarray: Clustering results.

    Raises:
        Exception: If an error occurs during clustering.

    Example Usage:
        ```python
        container = ClusteringContainer()
        train_data = np.array([[1, 2], [3, 4]])
        sample_weights = [0.5, 0.5]
        result = execute_clustering(container, train_data, sample_weights)
        print(result)  # Clustering result
        ```
    """
    try:
        validate_sample_weights(train_data, train_sample_weight)

        clustering_tool = container.clustering_algorithm()
        result = clustering_tool.cluster(
            train_data,
            sample_weight=train_sample_weight,
        )

        return result

    except Exception as e:
        raise Exception(f"Error during clustering: {e}")


def save_results(
    io_handler: IOHandler, result: Optional[ndarray], output_path: str
) -> None:
    """
    Save clustering results to a specified file.

    This function saves the clustering result to
    the specified output file using the provided
    IO handler. If no result is provided, the function does nothing.

    Args:
        io_handler (IOHandler): Instance of an IO handler for data saving.
        result (Optional[ndarray]): Clustering results to save.
        output_path (str): Path to the output file.

    Returns:
        None: This function does not return any value.

    Raises:
        Exception: If there is an error saving the results.

    Example Usage:
        ```python
        io_handler = IOHandler()
        result = np.array([0, 1, 1, 0])
        save_results(io_handler, result, "output.json")
        ```
    """
    try:
        if result is not None:
            io_handler.save_data(result, output_path)
    except Exception as e:
        raise Exception(f"Error saving results: {e}")
