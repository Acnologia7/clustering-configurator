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

        Raises:
            FileNotFoundError:
            If the file does not exist.
            ValueError:
            If the file is not a valid JSON or is corrupted.
            PermissionError:
            If there are insufficient permissions to read the file.
            Exception:
            For any other unexpected errors.

        Example:
            >>> handler = JSONHandler()
            >>> data = handler.load_data("data.json")
        """
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
                return np.array(data)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File not found: {filepath}. Please check the file path."
            ) from e

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON data from file '{filepath}': {str(e)}. "
                f"The file may be corrupted or improperly formatted."
            ) from e

        except PermissionError as e:
            raise PermissionError(
                f"Permission denied while reading the file: {filepath}."
                f"Check your file permissions."
            ) from e

        except Exception as e:
            raise Exception(
                f"An unexpected error occurred"
                f"while loading data from {filepath}: {e}"
            ) from e

    def save_data(self, data: ndarray, filepath: str) -> None:
        """
        Save data to a JSON file.

        Args:
            data (ndarray): Data to save.
            filepath (str): Path to save the JSON file.

        Raises:
            FileNotFoundError:
            If the directory to save the file does not exist.
            PermissionError:
            If there are insufficient permissions to write to the file.
            ValueError:
            If the file is not a valid JSON or is corrupted.
            Exception:
            For any other unexpected errors.

        Example:
            >>> handler = JSONHandler()
            >>> handler.save_data(np.array([1, 2, 3]), "output.json")
        """
        try:
            with open(filepath, "w") as file:
                json.dump(data.tolist(), file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Directory for {filepath}"
                f"does not exist or is not accessible."
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied when attempting to"
                f"save data to {filepath}."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON data from file '{filepath}': {str(e)}. "
                f"The file may be corrupted or improperly formatted."
            ) from e
        except Exception as e:
            raise Exception(
                f"An error occurred while saving data to {filepath}: {e}"
            ) from e


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

        Raises:
            FileNotFoundError:
            If the file does not exist.
            ValueError:
            If there is an error converting data to a NumPy array.
            Exception:
            For any other unexpected errors.

        Example:
            >>> handler = NumpyHandler()
            >>> data = handler.load_data("data.npy")
        """
        try:
            return np.load(filepath)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Error: The file '{filepath}'"
                f"does not exist. Please check the file path."
            ) from e
        except TypeError as e:
            raise ValueError(
                f"Error converting data to a NumPy array from"
                f"file '{filepath}': {e}."
            ) from e
        except Exception as e:
            raise Exception(
                f"Unexpected error while loading file '{filepath}': {e}"
            ) from e

    def save_data(self, data: ndarray, filepath: str) -> None:
        """
        Save data to a NumPy file.

        Args:
            data (ndarray): Data to save.
            filepath (str): Path to save the .npy file.

        Raises:
            FileNotFoundError:
            If the directory to save the file does not exist.
            PermissionError:
            If there are insufficient permissions to write to the file.
            Exception:
            For any other unexpected errors.

        Example:
            >>> handler = NumpyHandler()
            >>> handler.save_data(np.array([1, 2, 3]), "output.npy")
        """
        try:
            np.save(filepath, data)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Failed to save data."
                f"The directory for {filepath} does not exist."
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied when"
                f"attempting to save data to {filepath}."
            ) from e
        except Exception as e:
            raise Exception(
                f"An error occurred while saving data to {filepath}: {e}"
            ) from e
