import os

import numpy as np
import pytest

from app.utils.io_handler import IOHandler, JSONHandler, NumpyHandler


@pytest.fixture
def sample_data():
    """Fixture for sample NumPy array data."""
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def json_file(tmp_path, sample_data):
    """Fixture for a temporary JSON file containing sample data."""
    filepath = tmp_path / "test.json"
    JSONHandler.save(sample_data, str(filepath).replace(".json", ""))
    return filepath


@pytest.fixture
def numpy_file(tmp_path, sample_data):
    """Fixture for a temporary NumPy file containing sample data."""
    filepath = tmp_path / "test.npy"
    NumpyHandler.save(sample_data, str(filepath).replace(".npy", ""))
    return filepath


def test_json_handler_save_and_load(tmp_path, sample_data):
    """Test JSONHandler's save and load functionality."""
    filepath = tmp_path / "test.json"
    # Test saving data
    JSONHandler.save(sample_data, str(filepath).replace(".json", ""))
    assert os.path.exists(filepath)

    # Test loading data
    loaded_data = JSONHandler.load(str(filepath))
    assert np.array_equal(loaded_data, sample_data)


def test_numpy_handler_save_and_load(tmp_path, sample_data):
    """Test NumpyHandler's save and load functionality."""
    filepath = tmp_path / "test.npy"
    # Test saving data
    NumpyHandler.save(sample_data, str(filepath).replace(".npy", ""))
    assert os.path.exists(filepath)

    # Test loading data
    loaded_data = NumpyHandler.load(str(filepath))
    assert np.array_equal(loaded_data, sample_data)


def test_io_handler_json_to_numpy(json_file, tmp_path):
    """Test IOHandler for loading from JSON and saving to NumPy."""
    output_path = tmp_path / "output.npy"
    io_handler = IOHandler(JSONHandler, NumpyHandler)

    # Load from JSON
    data = io_handler.load_data(str(json_file))
    assert data is not None

    # Save to NumPy
    io_handler.save_data(data, str(output_path).replace(".npy", ""))
    assert os.path.exists(output_path)

    # Verify the saved file
    loaded_data = NumpyHandler.load(str(output_path))
    assert np.array_equal(loaded_data, data)


def test_io_handler_numpy_to_json(numpy_file, tmp_path):
    """Test IOHandler for loading from NumPy and saving to JSON."""
    output_path = tmp_path / "output.json"
    io_handler = IOHandler(NumpyHandler, JSONHandler)

    # Load from NumPy
    data = io_handler.load_data(str(numpy_file))
    assert data is not None

    # Save to JSON
    io_handler.save_data(data, str(output_path).replace(".json", ""))
    assert os.path.exists(output_path)

    # Verify the saved file
    loaded_data = JSONHandler.load(str(output_path))
    assert np.array_equal(loaded_data, data)
