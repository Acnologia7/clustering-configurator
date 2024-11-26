from argparse import ArgumentParser

import yaml
from dependencies import ClusteringContainer
from utils.validations import Config


def main():
    parser = ArgumentParser(description="Data Clustering Tool")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--train-input", required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--test-input", help="Path to test data file (optional)"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save clustered output"
    )
    parser.add_argument(
        "--train-sample-weight",
        help="Path to sample weight file for training (optional)",
    )
    parser.add_argument(
        "--test-sample-weight",
        help="Path to sample weight file for testing (optional)",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as file:
            raw_config = yaml.safe_load(file)
        validated_config = Config(**raw_config)
    except Exception as e:
        print(f"Configuration Validation Error: {e}")
        return

    container = ClusteringContainer()
    container.config.from_dict(validated_config.model_dump(mode="json"))

    io_handler = container.io_handler()

    try:
        train_data = io_handler.load_data(
            args.train_input, validated_config.clustering.input_format
        )
        test_data = (
            io_handler.load_data(
                args.test_input, validated_config.clustering.input_format
            )
            if args.test_input
            else None
        )
        train_sample_weight = (
            io_handler.load_data(
                args.train_sample_weight,
                validated_config.clustering.input_format,
            )
            if args.train_sample_weight
            else None
        )
        test_sample_weight = (
            io_handler.load_data(
                args.test_sample_weight,
                validated_config.clustering.input_format,
            )
            if args.test_sample_weight
            else None
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        clustering_tool = container.clustering_algorithm()
        result = clustering_tool(
            train_data,
            test_data=test_data,
            train_sample_weight=train_sample_weight,
            test_sample_weight=test_sample_weight,
        )
        attribute = clustering_tool.get_selected_attribute()

        if result is not None:
            io_handler.save_data(
                result, args.output, validated_config.clustering.output_format
            )

        if attribute is not None:
            print(f"{validated_config.clustering.attribute}: {attribute}")

    except Exception as e:
        print(f"Error during clustering: {e}")
        return


if __name__ == "__main__":
    main()
