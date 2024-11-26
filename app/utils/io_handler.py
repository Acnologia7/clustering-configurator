import json
from typing import Optional, Union

import numpy as np
from numpy import ndarray


class IOHandler:
    """
    A utility class for loading and saving data in either JSON or NumPy format.

    Methods:
        load_data(filepath: str, file_format: str) -> Optional[ndarray]:
            Loads data from a file in either JSON or NumPy format.

        save_data(data: Union[ndarray, float],
        filepath: str, file_format: str) -> None:
        Saves data to a file in either JSON or NumPy format.
    """

    @staticmethod
    def load_data(filepath: str, file_format: str) -> Optional[ndarray]:
        """
        Loads data from a file in either JSON or NumPy format.

        Args:
            filepath (str): The path to the input file.
            file_format (str): The format of the file to load.
                Must be either "json" or "numpy".

        Returns:
            Optional[ndarray]: The loaded data as a NumPy array,
            or None if an error occurred.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If there are errors while reading the file.
        """
        try:
            if file_format == "json":
                with open(filepath, "r") as file:
                    return np.array(json.load(file))
            elif file_format == "numpy":
                return np.load(filepath)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {e}")

    @staticmethod
    def save_data(
        data: Union[ndarray, float], filepath: str, file_format: str
    ) -> None:
        """
        Saves data to a file in either JSON or NumPy format.

        Args:
            data (Union[ndarray, float]): The data to be saved.
            filepath (str): The path to the output file.
            file_format (str): The format of the file to save.
                Must be either "json" or "numpy".

        Raises:
            FileNotFoundError: If the specified file cannot be accessed.
            Exception: If there are errors while writing the file.
        """
        try:
            if file_format == "json":
                data_to_write = (
                    data.tolist() if not isinstance(data, float) else data
                )
                with open(filepath, "w") as file:
                    json.dump(data_to_write, file)
            elif file_format == "numpy":
                np.save(filepath, data)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise Exception(f"Error saving data to {filepath}: {e}")
