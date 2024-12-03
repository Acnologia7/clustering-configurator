import logging
from traceback import format_exc

from pydantic import ValidationError

from app.dependencies import ClusteringContainer
from app.flow_functions import (execute_clustering, load_and_validate_config,
                                load_data, parse_arguments, save_results)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def main():
    try:
        logging.info("Parsing arguments...")
        args = parse_arguments()

        logging.info("Loading and validating configuration...")
        validated_config = load_and_validate_config(args.config)

        logging.info("Initializing clustering container...")
        container = ClusteringContainer()
        container.config.from_dict(validated_config.model_dump(mode="json"))

        logging.info("Setting up handlers and algorithm...")
        input_handler = container.input_handler()
        output_handler = container.output_handler()
        clustering_algorithm = container.clustering_algorithm()

        logging.info("Loading data...")
        train_data, train_sample_weight = load_data(
            input_handler, args.input, args.sample_weight
        )

        logging.info("Executing clustering...")
        result = execute_clustering(
            clustering_algorithm, train_data, train_sample_weight
        )

        logging.info("Saving results...")
        save_results(output_handler, result, args.output)

        logging.info("Clustering workflow completed successfully.")

    except ValidationError as e:
        logging.error("Configuration validation error: %s", e)
    except TypeError as e:
        logging.error("Type error: %s", e)
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
    except ValueError as e:
        logging.error("Value error: %s", e)
    except Exception as e:
        logging.critical("An unexpected error occurred: %s", e)
        logging.debug("Stack trace:\n%s", format_exc())
        raise


if __name__ == "__main__":
    main()
