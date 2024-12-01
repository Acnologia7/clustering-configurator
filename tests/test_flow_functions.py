from unittest.mock import MagicMock, mock_open

import numpy as np
import pytest

from app.flow_functions import (
    execute_clustering,
    load_and_validate_config,
    load_data,
    parse_arguments,
    save_results,
    validate_sample_weights,
)
from app.utils.validations import Config


def test_parse_arguments(mocker):
    """Test argument parsing."""
    mocker.patch(
        "sys.argv",
        [
            "program_name",
            "--config",
            "config.yaml",
            "--input",
            "input.json",
            "--output",
            "output.json",
        ],
    )

    args = parse_arguments()
    assert args.config == "config.yaml"
    assert args.input == "input.json"
    assert args.output == "output.json"
    assert args.sample_weight is None


def test_load_and_validate_config(mocker):
    """Test loading and validating config."""
    mocker.patch(
        "builtins.open",
        mock_open(read_data="clustering:\n  algorithm: KMeans"),
    )
    mocker.patch(
        "yaml.safe_load",
        return_value={"clustering": {"algorithm": "KMeans"}},
    )
    mock_config = Config(clustering={"algorithm": "KMeans"})

    result = load_and_validate_config("mock_config.yaml")

    assert result.clustering == mock_config.clustering
    assert result.clustering.algorithm == mock_config.clustering.algorithm


def test_load_data(mocker):
    """Test loading data and sample weights."""
    mock_io_handler = MagicMock()

    # Define a mock function for load_data
    def mock_load_data(path):
        if path == "input.json":
            return np.array([[1, 2], [3, 4]])
        elif path == "weights.json":
            return [0.5, 0.5]
        else:
            raise FileNotFoundError(f"File {path} not found!")

    mock_io_handler.load_data.side_effect = mock_load_data

    # Test case with sample weights
    train_data, train_sample_weight = load_data(
        mock_io_handler, "input.json", "weights.json"
    )
    assert np.array_equal(train_data, np.array([[1, 2], [3, 4]]))
    assert train_sample_weight == [0.5, 0.5]

    # Test case without sample weights
    train_data, train_sample_weight = load_data(
        mock_io_handler, "input.json", None
    )
    assert np.array_equal(train_data, np.array([[1, 2], [3, 4]]))
    assert train_sample_weight is None


def test_validate_sample_weights():
    """Test sample weight validation."""
    train_data = np.array([[1, 2], [3, 4], [5, 6]])
    train_sample_weight = np.array([0.1, 0.2, 0.3])

    # Valid weights
    try:
        validate_sample_weights(train_data, train_sample_weight)
    except ValueError:
        pytest.fail("Validation raised ValueError unexpectedly!")

    # Invalid weights
    train_sample_weight = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        validate_sample_weights(train_data, train_sample_weight)


def test_execute_clustering(mocker):
    """Test clustering execution."""
    clustering_algorithm = MagicMock()
    clustering_algorithm.cluster.return_value = "mock_result"
    clustering_algorithm.return_value = clustering_algorithm

    train_data = np.array([[1, 2], [3, 4]])
    train_sample_weight = np.array([0.5, 0.5])

    result = execute_clustering(
        clustering_algorithm, train_data, train_sample_weight
    )
    print(result)
    assert result == "mock_result"
    clustering_algorithm.cluster.assert_called_once_with(
        train_data, sample_weight=train_sample_weight.tolist()
    )


def test_save_results(mocker):
    """Test saving results."""
    mock_io_handler = MagicMock()

    # Test successful save
    save_results(mock_io_handler, "mock_result", "output.json")
    mock_io_handler.save_data.assert_called_once_with(
        "mock_result", "output.json"
    )
