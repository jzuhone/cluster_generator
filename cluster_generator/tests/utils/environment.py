"""
Utility Functions for Configuration Loading and File Downloading.

This module provides utility functions to facilitate loading configuration settings from
a YAML file and downloading files from a specified URL. These utilities are designed to
streamline configuration management and data acquisition processes within various
applications and scripts.

Functions
---------
load_config(file_path: str) -> dict
    Load configuration settings from a specified YAML file.

download_file(url: str, output_path: str) -> None
    Download a file from a specified URL to a local path.

Examples
--------
To load configuration settings from a YAML file:

    >>> config = load_config("config.yaml")
    >>> print(config)

To download a file from a URL:

    >>> download_file("https://example.com/data.csv", "local_data.csv")
"""
from typing import Dict

import yaml


def load_config(file_path: str) -> Dict:
    """
    Load the configuration settings from a YAML file.

    This function reads a YAML file from the specified path and returns its contents as a dictionary.
    It uses `yaml.safe_load` to parse the YAML, ensuring that the file content is loaded safely.

    Parameters
    ----------
    file_path : str
        The path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration settings.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.

    Examples
    --------
    >>> config = load_config("config.yaml")
    >>> print(config)
    """
    try:
        with open(file_path, "r") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}"
        ) from fnf_error
    except yaml.YAMLError as yaml_error:
        raise yaml.YAMLError(f"Error parsing YAML file: {file_path}") from yaml_error
