from app.dependencies import ClusteringContainer
from app.flow_functions import (execute_clustering, load_and_validate_config,
                                load_data, parse_arguments, save_results)


def main():
    try:
        args = parse_arguments()
        validated_config = load_and_validate_config(args.config)

        container = ClusteringContainer()
        container.config.from_dict(validated_config.model_dump(mode="json"))

        input_handler = container.input_handler()
        output_handler = container.output_handler()
        clustering_algorithm = container.clustering_algorithm()

        train_data, train_sample_weight = load_data(
            input_handler, args.input, args.sample_weight
        )

        result = execute_clustering(
            clustering_algorithm, train_data, train_sample_weight
        )

        save_results(output_handler, result, args.output)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
