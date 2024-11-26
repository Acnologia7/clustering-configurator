from typing import Optional, Type, Union

from numpy import ndarray
from sklearn.base import BaseEstimator


class GeneralClusteringWrapper(BaseEstimator):
    def __init__(self, base_class: Type[BaseEstimator], config: dict):
        """
        A generalized wrapper for clustering algorithms.

        Args:
            base_class (Type[BaseEstimator]):
            The clustering algorithm class (e.g., KMeans, DBSCAN).
            config (dict): The configuration dictionary for the algorithm.
        """
        self.config: dict = config
        self.base_class: Type[BaseEstimator] = base_class
        self.selected_method: str = self.config["clustering"]["method"]
        self.selected_attribute: Optional[str] = self.config["clustering"][
            "attribute"
        ]

        self.hyperparameters: dict = self.config["clustering"][
            "hyperparameters"
        ]
        self.model: BaseEstimator = base_class(**self.hyperparameters)

    def __call__(self, *args, **kwargs) -> Optional[ndarray]:
        """
        Call the selected method with provided arguments and
        handle data processing based on user configuration.

        Args:
            *args: Positional arguments, where the first is the training data.
            **kwargs: Keyword arguments, such as test data and sample weights.

        Returns:
            Optional[ndarray]: The result of the selected method,
            or None if the method does not return anything.
        """
        if len(args) == 0:
            raise ValueError("Training data (train_data) is required.")

        train_data: ndarray = args[0]
        test_data: Optional[ndarray] = kwargs.get("test_data")
        train_sample_weight: Optional[ndarray] = kwargs.get(
            "train_sample_weight"
        )
        test_sample_weight: Optional[ndarray] = kwargs.get(
            "test_sample_weight"
        )

        if not test_data:
            test_data = train_data

        if self.selected_method in {"predict", "transform", "score", "fit"}:
            self.model.fit(train_data, sample_weight=train_sample_weight)

        method = getattr(self.model, self.selected_method)

        if self.selected_method == "score":
            return method(test_data, sample_weight=test_sample_weight)
        elif self.selected_method in {"predict", "transform"}:
            return method(test_data)
        elif self.selected_method == "fit":
            return None

        return method(test_data, sample_weight=train_sample_weight)

    def get_selected_attribute(self) -> Optional[Union[ndarray, int, float]]:
        """
        Access the selected attribute of the model.

        Returns:
            Optional[Union[ndarray, int, float]]:
            The value of the selected attribute or None if not set.
        """
        if self.selected_attribute:
            return getattr(self.model, self.selected_attribute)
        return None
