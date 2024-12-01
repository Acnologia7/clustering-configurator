import os

import numpy as np
import pytest

from app.utils.io_handler import JSONHandler, NumpyHandler


@pytest.fixture
def sample_data():
    """Fixture for sample NumPy array data."""
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def json_file(tmp_path, sample_data):
    """Fixture for a temporary JSON file containing sample data."""
    filepath = tmp_path / "test.json"
    json_io_handler = JSONHandler()
    json_io_handler.save_data(sample_data, str(filepath))
    return filepath


@pytest.fixture
def numpy_file(tmp_path, sample_data):
    """Fixture for a temporary NumPy file containing sample data."""
    filepath = tmp_path / "test.npy"
    numpy_io_handler = NumpyHandler()
    numpy_io_handler.save_data(sample_data, str(filepath))
    return filepath


def test_json_handler_save_and_load(tmp_path, sample_data):
    """Test JSONHandler's save and load functionality."""
    filepath = tmp_path / "test.json"
    json_io_handler = JSONHandler()

    # Test saving data
    json_io_handler.save_data(sample_data, str(filepath))
    assert os.path.exists(filepath)

    # Test loading data
    loaded_data = json_io_handler.load_data(str(filepath))
    assert np.array_equal(loaded_data, sample_data)


def test_numpy_handler_save_and_load(tmp_path, sample_data):
    """Test NumpyHandler's save and load functionality."""
    filepath = tmp_path / "test.npy"
    numpy_io_handler = NumpyHandler()

    # Test saving data
    numpy_io_handler.save_data(sample_data, str(filepath))
    assert os.path.exists(filepath)

    # Test loading data
    loaded_data = numpy_io_handler.load_data(str(filepath))
    assert np.array_equal(loaded_data, sample_data)


def test_io_handler_json_to_numpy(json_file, tmp_path):
    """Test IOHandler for loading from JSON and saving to NumPy."""
    output_path = tmp_path / "output.npy"
    json_io_handler = JSONHandler()
    numpy_io_handler = NumpyHandler()

    # Load from JSON
    data = json_io_handler.load_data(str(json_file))
    assert data is not None

    # Save to NumPy
    numpy_io_handler.save_data(data, str(output_path))
    assert os.path.exists(output_path)

    # Verify the saved file
    loaded_data = numpy_io_handler.load_data(str(output_path))
    assert np.array_equal(loaded_data, data)


def test_io_handler_numpy_to_json(numpy_file, tmp_path):
    """Test IOHandler for loading from NumPy and saving to JSON."""
    output_path = tmp_path / "output.json"
    numpy_io_handler = NumpyHandler()
    json_io_handler = JSONHandler()

    # Load from NumPy
    data = numpy_io_handler.load_data(str(numpy_file))
    assert data is not None

    # Save to JSON
    json_io_handler.save_data(data, str(output_path))
    assert os.path.exists(output_path)

    # Verify the saved file
    loaded_data = json_io_handler.load_data(str(output_path))
    assert np.array_equal(loaded_data, data)
