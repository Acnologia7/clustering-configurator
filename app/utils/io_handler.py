import json
from typing import Optional

import numpy as np
from numpy import ndarray


class IOHandler:
    """
    A utility class for loading and saving data in either JSON or NumPy format.

    Methods:
        load_data(config: IOHandlerConfig) -> Optional[ndarray]:
            Loads data from a file in either JSON or NumPy format.

        save_data(data: np.ndarray, config: IOHandlerConfig) -> None:
            Saves data to a file in either JSON or NumPy format.
    """

    class IOHandler:
        @staticmethod
        def load_data(filepath: str, format: str) -> Optional[np.ndarray]:
            """
            Loads data from a file in either JSON or NumPy format.

            Args:
                filepath (str): The path to the input file.
                format (str): The format of the file to load.
                Must be either "json" or "numpy".

            Returns:
                Optional[np.ndarray]: The loaded data as a NumPy array,
                or None if an error occurred.

            Raises:
                FileNotFoundError: If the specified file does not exist.
                Exception: If there are errors while reading the file.
            """
            try:
                if format == "json":
                    with open(filepath, "r") as file:
                        return np.array(json.load(file))
                elif format == "numpy":
                    return np.load(filepath)
            except FileNotFoundError:
                raise
            except Exception:
                raise
            return None

    @staticmethod
    def save_data(data: ndarray, filepath: str, format: str) -> None:
        """
        Saves data to a file in either JSON or NumPy format.

        Args:
            data (np.ndarray): The data to be saved.
            filepath (str): The path to the output file.
            format (str): The format of the file to save.
            Must be either "json" or "numpy".

        Raises:
            FileNotFoundError: If the specified file cannot be accessed.
            Exception: If there are errors while writing the file.
        """
        try:
            if format == "json":
                with open(filepath, "w") as file:
                    json.dump(data.tolist(), file)
            elif format == "numpy":
                np.save(filepath, data)
        except FileNotFoundError:
            raise
        except Exception:
            raise
