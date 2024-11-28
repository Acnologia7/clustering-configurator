import json
from typing import Optional, Union

import numpy as np
from numpy import ndarray


class JSONHandler:
    """
    A handler for reading and writing data in JSON format.

    This class provides static methods to load data
    from a JSON file and save data to a JSON file.
    Data is converted between JSON format and NumPy arrays
    for compatibility with numerical operations.

    Methods:
        load(filepath: str) -> Optional[ndarray]:
            Load JSON data from a file and convert it to a NumPy array.

        save(data: ndarray, filepath: str) -> None:
            Save a NumPy array to a JSON file.

    Example usage:
        >>> json_handler = JSONHandler()
        >>> data = json_handler.load('data.json')
        >>> print(data)
        [[1, 2, 3], [4, 5, 6]]
        >>> json_handler.save(data, 'output')
    """

    @staticmethod
    def load(filepath: str) -> Optional[ndarray]:
        """
        Load data from a JSON file and return it as a NumPy array.

        Args:
            filepath (str): Path to the JSON file to load.

        Returns:
            Optional[ndarray]: A NumPy array containing
            the loaded data, or None if the file does not exist.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For other issues while reading the file.

        Example:
            >>> data = JSONHandler.load('data.json')
            >>> print(data)
            [[1, 2, 3], [4, 5, 6]]
        """
        try:
            with open(filepath, "r") as file:
                loaded_data = json.load(file)
                data_array = np.array(loaded_data)
        except FileNotFoundError:
            print(f"File: {filepath} does not exist.")
        except Exception as e:
            raise e

        return data_array

    @staticmethod
    def save(data: ndarray, filepath: str) -> None:
        """
        Save a NumPy array to a JSON file.

        Args:
            data (ndarray): The NumPy array to save.
            filepath (str): The path where the JSON file
            will be saved with extention '.json'.

        Raises:
            Exception: If there is an error while writing the file.

        Example:
            >>> data = np.array([[1, 2, 3], [4, 5, 6]])
            >>> JSONHandler.save(data, 'output')
        """
        try:
            with open(f"{filepath}.json", "w") as file:
                json.dump(data.tolist(), file)
        except Exception as e:
            raise e


class NumpyHandler:
    """
    A handler for reading and writing data in NumPy format (.npy).

    This class provides static methods to load data
    from a NumPy file and save data to a NumPy file.
    It allows efficient storage and
    retrieval of NumPy arrays in their native format.

    Methods:
        load(filepath: str) -> Optional[ndarray]:
            Load data from a NumPy file and return it as a NumPy array.

        save(data: ndarray, filepath: str) -> None:
            Save a NumPy array to a file in NumPy format.

    Example usage:
        >>> numpy_handler = NumpyHandler()
        >>> data = numpy_handler.load('data.npy')
        >>> print(data)
        [[1, 2, 3], [4, 5, 6]]
        >>> numpy_handler.save(data, 'output')
    """

    @staticmethod
    def load(filepath: str) -> Optional[ndarray]:
        """
        Load data from a NumPy file and return it as a NumPy array.

        Args:
            filepath (str): Path to the NumPy file to load.

        Returns:
            Optional[ndarray]: A NumPy array containing the loaded data,
            or None if the file does not exist.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For other issues while reading the file.

        Example:
            >>> data = NumpyHandler.load('data.npy')
            >>> print(data)
            [[1, 2, 3], [4, 5, 6]]
        """
        try:
            data_array = np.load(filepath)
        except FileNotFoundError:
            print(f"File: {filepath} does not exist.")
        except Exception as e:
            raise e

        return data_array

    @staticmethod
    def save(data: ndarray, filepath: str) -> None:
        """
        Save a NumPy array to a file in NumPy format.

        Args:
            data (ndarray): The NumPy array to save.
            filepath (str): The path where
            the NumPy file will be saved (without extension).

        Raises:
            Exception: If there is an error while writing the file.

        Example:
            >>> data = np.array([[1, 2, 3], [4, 5, 6]])
            >>> NumpyHandler.save(data, 'output')
        """
        try:
            np.save(f"{filepath}.npy", data)
        except Exception as e:
            raise e


class IOHandler:
    """
    A utility class for loading and saving data using injected handlers.

    This class serves to abstract
    the details of file format handling by injecting
    specific handlers (e.g., JSONHandler or NumpyHandler)
    for input and output operations.

    Attributes:
        input_handler (Union[JSONHandler, NumpyHandler]):
        The handler used for loading data.
        output_handler (Union[JSONHandler, NumpyHandler]):
        The handler used for saving data.

    Methods:
        load_data(filepath: str) -> Optional[ndarray]:
            Load data using the input handler.

        save_data(data: ndarray, filepath: str) -> None:
            Save data using the output handler.

    Example usage:
        >>> input_handler = JSONHandler()
        >>> output_handler = NumpyHandler()
        >>> io_handler = IOHandler(input_handler, output_handler)
        >>> data = io_handler.load_data('data.json')
        >>> print(data)
        [[1, 2, 3], [4, 5, 6]]
        >>> io_handler.save_data(data, 'output')
    """

    def __init__(
        self,
        input_handler: Union[JSONHandler, NumpyHandler],
        output_handler: Union[JSONHandler, NumpyHandler],
    ) -> None:
        """
        Initialize the IOHandler with specified input and output handlers.

        Args:
            input_handler (Union[JSONHandler, NumpyHandler]):
            The handler used for loading data.
            output_handler (Union[JSONHandler, NumpyHandler]):
            The handler used for saving data.
        """
        self.input_handler = input_handler
        self.output_handler = output_handler

    def load_data(self, filepath: str) -> Optional[ndarray]:
        """
        Load data using the injected input handler.

        Args:
            filepath (str): Path to the file to load.

        Returns:
            Optional[ndarray]: A NumPy array containing the loaded data,
            or None if the file does not exist.

        Raises:
            Exception: If there is an error during data loading.

        Example:
            >>> io_handler = IOHandler(JSONHandler(), NumpyHandler())
            >>> data = io_handler.load_data('data.json')
            >>> print(data)
            [[1, 2, 3], [4, 5, 6]]
        """
        try:
            return self.input_handler.load(filepath)
        except Exception as e:
            raise e

    def save_data(self, data: ndarray, filepath: str) -> None:
        """
        Save data using the injected output handler.

        Args:
            data (ndarray): The NumPy array to save.
            filepath (str): The path where the file
            will be saved (without extension).

        Raises:
            Exception: If there is an error during data saving.

        Example:
            >>> io_handler = IOHandler(JSONHandler(), NumpyHandler())
            >>> data = np.array([[1, 2, 3], [4, 5, 6]])
            >>> io_handler.save_data(data, 'output')
        """
        try:
            self.output_handler.save(data, filepath)
        except Exception as e:
            raise e
