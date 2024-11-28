# Clustering configurator

This is a Python-based data clustering tool designed to handle clustering tasks, including loading data, validating configurations, performing clustering, and saving the results. It integrates various machine learning and data processing utilities into a cohesive workflow.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pre-Commit Hook](#pre-commit-hook)

## Installation

To install the project and its dependencies, follow these steps:

1. **Clone the Repository**

   Clone the repository to your local machine.

2. **Create a Virtual Environment**

   It’s recommended to use a virtual environment to manage dependencies.

   - To create a virtual environment, run:
     ```
     python -m venv venv
     ```
   - To activate the virtual environment:
     - On Windows:
       ```
       .\venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```
       source venv/bin/activate
       ```

3. **Install Dependencies**

   Install the required dependencies using:
   `pip install -r requirements.txt`

## Configuration

Before using the tool, you need to provide a configuration file (`config.yaml`) that outlines the clustering algorithm and other relevant parameters. Below is an example of how the `config.yaml` should look:

```yaml
clustering:
  algorithm: "KMeans" # Clustering algorithm (KMeans, DBSCAN, AgglomerativeClustering)
  hyperparameters: # Parameters for corresponding algorithm
    n_clusters: 8
    init: "k-means++"
    n_init: "auto"
    max_iter: 300
    tol: 1e-4
    verbose: 0
    random_state: None
    copy_x: True
    algorithm: "lloyd"
  input_format: "json" # Format of input data (numpy or json)
  output_format: "json" # Format of output data (numpy or json)

# Hyperparameters can be ommited completly
# if you wish to use default parameters defined by scikit-learn docummentation
```

For complete view I add list of typed hyperparameters for every supported algorithm how what this app supports with their default values:

```yaml
#KMeans
    n_clusters: int = 8
    init: Union[Literal["k-means++", "random"], list] = "k-means++"
    n_init: Union[Literal["auto"], int] = "auto"
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: Optional[int] = None
    copy_x: bool = True
    algorithm: Literal["lloyd", "elkan"] = "lloyd"

#DBSCAN
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    metric_params: Optional[dict[str, Any]] = None
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: Optional[float] = None
    n_jobs: Optional[int] = None

#AgglomerativeClustering
    n_clusters: Optional[int] = 2
    metric: str = "euclidean"
    memory: Optional[str] = None
    connectivity: Optional[list[Any]] = None
    compute_full_tree: Union[Literal["auto"], bool] = "auto"
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    distance_threshold: Optional[float] = None
    compute_distances: bool = False
```

You can modify the algorithm and hyperparameters section to match your clustering setup. Ensure that the file paths in the configuration point to the correct locations.

## Usage

Once the dependencies are installed and configuration is set up, you can use the clustering tool via the command line.

The tool accepts the following command-line arguments:

    --config: Path to the configuration file (required).
    --input: Path to the input data file (required).
    --output: Path to save the output clustered data (required).
    --sample-weight: Path to the sample weight file (optional).

Running the Tool

Here’s an example of how to run the tool from the command line:

```bash
python main.py --config config.yaml --input input.json --output output.json --sample-weight weights.json

```

This command will:

- Parse the arguments.
- Load and validate the configuration.
- Load the input data and sample weights (optinal).
- Perform the clustering operation.
- Save the clustered result to output.

## Pre-Commit Hook

To ensure code quality and consistency, I used a pre-commit hooks that automatically check for Python code formatting and other checks before committing changes to the repository. Those hooks helps maintain clean and consistent code, reducing the chance of errors and style violations.

### Setting Up Pre-Commit Hook

Follow these steps to set up the pre-commit hook:

1. Install pre-commit

   If you don't have pre-commit installed, you can install it with:

   ```bash
   pip install pre-commit
   ```

2. Innstall Git Hooks

   In the root directory of your project, create a .pre-commit-config.yaml file with the following content:

   ```yaml
   repos:
   # Black: Auto-formatting
   - repo: https://github.com/psf/black
       rev: 23.9.1
       hooks:
       - id: black

   # Flake8: Linting
   - repo: https://github.com/PyCQA/flake8
       rev: 6.1.0
       hooks:
       - id: flake8

   # Mypy: Type checking
   - repo: https://github.com/pre-commit/mirrors-mypy
       rev: v1.5.1
       hooks:
       - id: mypy
           additional_dependencies:
           [
               types-PyYAML==6.0.2,
               dependency-injector==4.43.0,
               scikit-learn==1.5.2,
           ]

   # End-of-file-fixer: Ensure consistent EOF
   - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
       - id: end-of-file-fixer

   # Check YAML files
   - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
       - id: check-yaml

   # Sort imports with isort
   - repo: https://github.com/pre-commit/mirrors-isort
       rev: v5.10.1
       hooks:
       - id: isort

   # Run tests with pytest
   - repo: local
       hooks:
       - id: pytest
           name: pytest
           entry: python -m pytest tests
           language: system
           types: [python]
           pass_filenames: false
           always_run: true
   ```

3. Install the Pre-Commit Hook

   After creating the .pre-commit-config.yaml file, install the hooks using the following command:

   ```bash
   pre-commit install
   ```

4. Running Pre-Commit Hook Manually (Optional)

   If you want to manually run the pre-commit hook on all files, use the following command:

   ```bash
   pre-commit run --all-files
   ```
