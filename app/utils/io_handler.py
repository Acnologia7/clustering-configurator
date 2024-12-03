import json
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy import ndarray


class BaseHandler(ABC):
    """
    Abstract base class for data handlers.

    Subclasses must implement `load_data` and `save_data`.

    Methods:
        load_data(filepath: str) -> Optional[ndarray]: Load data from a file.
        save_data(data: ndarray, filepath: str) -> None: Save data to a file.
    """

    @abstractmethod
    def load_data(self, filepath: str) -> Optional[ndarray]:
        """
        Load data from a file.

        Args:
            filepath (str): Path to the file.

        Returns:
            Optional[ndarray]: Loaded data as a NumPy array or None.

        Example:
            >>> handler.load_data("data.json")
        """
        pass

    @abstractmethod
    def save_data(self, data: ndarray, filepath: str) -> None:
        """
        Save data to a file.

        Args:
            data (ndarray): Data to save.
            filepath (str): Path to save the file.

        Example:
            >>> handler.save_data(np.array([1, 2, 3]), "output.npy")
        """
        pass


class JSONHandler(BaseHandler):
    """
    Handler for JSON files.

    Methods:
        load_data(filepath: str) -> Optional[ndarray]: Load JSON data.
        save_data(data: ndarray, filepath: str) -> None: Save data as JSON.
    """

    def load_data(self, filepath: str) -> Optional[ndarray]:
        """
        Load data from a JSON file.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            Optional[ndarray]: Loaded data as a NumPy array or None.

        Example:
            >>> handler = JSONHandler()
            >>> data = handler.load_data("data.json")
        """
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
                return np.array(data)
        except FileNotFoundError:
            print(f"File: {filepath} does not exist.")
            return None
        except Exception as e:
            raise e

    def save_data(self, data: ndarray, filepath: str) -> None:
        """
        Save data to a JSON file.

        Args:
            data (ndarray): Data to save.
            filepath (str): Path to save the JSON file.

        Example:
            >>> handler = JSONHandler()
            >>> handler.save_data(np.array([1, 2, 3]), "output.json")
        """
        try:
            with open(filepath, "w") as file:
                json.dump(data.tolist(), file)
        except Exception as e:
            raise e


class NumpyHandler(BaseHandler):
    """
    Handler for NumPy files (.npy).

    Methods:
        load_data(filepath: str) -> Optional[ndarray]: Load .npy data.
        save_data(data: ndarray, filepath: str) -> None: Save data as .npy.
    """

    def load_data(self, filepath: str) -> Optional[ndarray]:
        """
        Load data from a NumPy file.

        Args:
            filepath (str): Path to the .npy file.

        Returns:
            Optional[ndarray]: Loaded data as a NumPy array or None.

        Example:
            >>> handler = NumpyHandler()
            >>> data = handler.load_data("data.npy")
        """
        try:
            return np.load(filepath)
        except FileNotFoundError:
            print(f"File: {filepath} does not exist.")
            return None
        except Exception as e:
            raise e

    def save_data(self, data: ndarray, filepath: str) -> None:
        """
        Save data to a NumPy file.

        Args:
            data (ndarray): Data to save.
            filepath (str): Path to save the .npy file.

        Example:
            >>> handler = NumpyHandler()
            >>> handler.save_data(np.array([1, 2, 3]), "output.npy")
        """
        try:
            np.save(filepath, data)
        except Exception as e:
            raise e
