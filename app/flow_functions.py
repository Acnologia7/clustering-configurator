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
        ValueError:
        If the YAML file cannot be parsed or if the configuration is invalid.
        FileNotFoundError:
        If the configuration file does not exist.
        Exception:
        For unexpected errors during the configuration loading process.


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
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise Exception(f"Unexpected error occurs: {e}")


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

    Raises:
        FileNotFoundError:
        If the input file or optional sample weight file does not exist.
        ValueError:
        If the input data or sample weights are
        invalid or improperly formatted.
        Exception:
        For any other unexpected errors during data loading.

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
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Input file: {input_path}."
            f"{
                f"Sample weight file: {sample_weight_path}."
                if sample_weight_path else ""
            } "
            f"Ensure the path is correct."
        ) from e
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Error parsing data from file: {e}. Verify the file format."
        ) from e
    except Exception as e:
        raise Exception(f"Unexpected error loading data: {e}") from e


def validate_sample_weights(
    train_data: ndarray, train_sample_weight: Optional[List[float]]
) -> None:
    """
    Ensure sample weights match the number of data points.

    Args:
        train_data (ndarray): Training data.
        train_sample_weight (Optional[List[float]]): Sample weights.

    Raises:
        ValueError:
        If the number of sample weights
        does not match the number of data points.


    Example:
        ```python
        validate_sample_weights(train_data, sample_weights)
        ```
    """
    if train_sample_weight is not None:
        num_data_points = train_data.shape[0]
        num_weights = len(train_sample_weight)

        if num_weights != num_data_points:
            raise ValueError(
                f"Sample weights mismatch: "
                f"Expected {num_data_points} data points, "
                f"but found {num_weights} sample weights. "
                f"Please ensure the number of "
                f"weights matches the number of data points."
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
        TypeError:
        If the sample weights cannot be converted to a list.
        ValueError:
        If the sample weights fail validation or mismatch the data shape.
        RuntimeError:
        If the clustering algorithm encounters an internal error.
        Exception:
        For any other unexpected errors during clustering.


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
    except TypeError as e:
        raise TypeError(
            f"Failed to convert sample weights to list: {e}. "
            f"Ensure sample weights are a valid ndarray."
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Sample weights validation error: {e}. "
            f"Check if weights match the data shape."
        ) from e
    except RuntimeError as e:
        raise RuntimeError(
            f"Clustering algorithm error: {e}. "
            f"Ensure the algorithm is configured correctly."
        ) from e
    except Exception as e:
        raise Exception(f"Unexpected clustering error: {e}") from e


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

    Raises:
        FileNotFoundError:
        If the output directory does not exist.
        IOError:
        If there is an I/O error while saving the results.
        ValueError:
        If the result is None or invalid for saving.
        Exception:
        For any other unexpected errors during result saving.

    Example:
        ```python
        save_results(handler, result, "output.json")
        ```
    """
    try:
        if result is None:
            raise ValueError("Result is None. Cannot save empty results.")

        output_handler.save_data(result, output_path)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Output path '{output_path}' not found: {e}. "
            f"Check the directory and permissions."
        ) from e
    except IOError as e:
        raise IOError(
            f"I/O error while saving results to '{output_path}': {e}. "
            f"Ensure the path is writable."
        ) from e
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Unexpected error while saving results: {e}") from e
