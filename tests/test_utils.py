from typing import List, Dict
from colorama import Fore
from colorama import init as colorama_init
import test_consts as consts
import pandas as pd
from pandas import DataFrame
import json
import os

colorama_init(autoreset=True)


def save_dataset_to_resources(dataset: DataFrame, dataset_name: str):
    """
    Save the dataset to resources/datasets.
    :param dataset: The dataset to save.
    :param dataset_name: The name of the dataset.
    """
    # Check if resources/datasets exists. If not, create it.
    if not os.path.exists("resources/datasets"):
        os.makedirs("resources/datasets")
    dataset.to_csv(f"resources/datasets/{dataset_name}.csv", index=False)


def load_dataset(path: str | None, dataset_name: str, default_datasets: List[Dict]) -> DataFrame | None:
    """
    Load a dataset from the given path.\n
    If the dataset can not be loaded from the given path, we try loading it by name from resources/datasets, the
    default library.
    If this too fails, we try loading it from the default datasets configuration by downloading it from the internet.
    :param path: Path to the dataset.
    :param dataset_name: Name of the dataset.
    :param default_datasets: List of dictionaries, each containing the name and path of a default dataset.
    :return: The loaded dataset.
    """
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        try:
            return pd.read_csv(f"resources/datasets/{dataset_name}.csv")
        except FileNotFoundError:
            for default_dataset in default_datasets:
                if default_dataset['name'] == dataset_name:
                    dataset = pd.read_csv(default_dataset['link'])
                    # Save the dataset to resources/datasets for future use.
                    save_dataset_to_resources(dataset, dataset_name)
                    return dataset
        raise FileNotFoundError(f"Dataset not found at path {path} and {dataset_name} not found in default datasets.")


def load_datasets(results: List[dict], default_configuration_path: str = "resources/default_datasets.json") -> Dict[
    str, DataFrame]:
    """
    Load all datasets from the given results.
    If a dataset is present in multiple results, only the first occurrence is considered.
    If a dataset can not be loaded but its name is present in the default configuration, the dataset will be loaded
    using the default configuration.
    :param results: List of dictionaries, each containing the content of a result file, loaded by load_all_results.
    :return: A dictionary with dataset names as keys and DataFrames as values.
    """
    default_datasets = list(json.load(open(default_configuration_path, "r"))['all'])
    datasets = {}

    for result in results:
        # Get the dataset name and path from the result.
        first_dataset_path = result['first_dataset']
        first_dataset_name = result['dataset_name']
        second_dataset_path = result['second_dataset']
        second_dataset_name = result['second_dataset_name']

        # Load the first dataset.
        if first_dataset_name not in datasets:
            datasets[first_dataset_name] = load_dataset(first_dataset_path, first_dataset_name, default_datasets)
        # Load the second dataset.
        if second_dataset_name not in datasets:
            datasets[second_dataset_name] = load_dataset(second_dataset_path, second_dataset_name, default_datasets)

    return datasets


def fix_duplicate_col_names_and_bin_names(dict_list: List[Dict]):
    """
    This function takes a list of influence values dictionaries or significance values dictionaries,
    and fixes the duplicate column names and bin names by appending a number to the column name and bin name.
    :param dict_list: List of dictionaries, in the format of influence values or significance values.
    :return: List of dictionaries, with fixed duplicate column names and bin names.
    """
    keys = set()
    idx_dict = {}
    for d in dict_list:
        col_name = d['column']
        bin_name = d['bin']
        if col_name in keys:
            idx_dict[col_name] += 1
            d['column'] = f'{col_name}_{idx_dict[col_name]}'
            d['bin'] = f'{bin_name}_{idx_dict[col_name]}'
        else:
            keys.add(col_name)
            idx_dict[col_name] = 1
    return dict_list


def print_test_messages(messages: List[str], test_name: str, passed: bool) -> None:
    """
    This function prints the test messages, the test name and whether the test passed or not.
    :param messages: List of messages to print.
    :param test_name: Name of the test.
    :param passed: Boolean value, whether the test passed or not.
    :return: None
    """
    if passed:
        color = Fore.GREEN
    else:
        color = Fore.RED

    print(f"\n{color} --------------------------------------------------------------")
    print(f"\t{color} {test_name}:\n")

    for message in messages:
        print(f"\t\t{color} {message}")

    if passed:
        print(f"{'\n' if len(messages) > 0 else ''}\t{color} TEST PASSED")
    else:
        print(f"{'\n' if len(messages) > 0 else ''}\t{color} TEST FAILED")
    print(f"{color} --------------------------------------------------------------")


def print_result_summary(test_outcomes: List[Dict], dataset_name: str) -> None:
    """
    This function prints the summary of the test outcomes.
    If any test failed, the function will print the failed tests.
    :param test_outcomes: List of dictionaries, each containing the test name, whether the test passed or not, and the failed tests.
    :param dataset_name: Name of the dataset.
    :return: None
    """
    passed_tests = [outcome for outcome in test_outcomes if outcome['passed']]
    failed_tests = [outcome for outcome in test_outcomes if not outcome['passed']]

    print(f"\n\n{Fore.CYAN} --------------------------------------------------------------")
    print(f"\t{Fore.CYAN} SUMMARY OF TESTS ON DATASET {dataset_name}:\n")
    print(f"\t{Fore.CYAN} QUERIES THAT PASSED ALL TESTS: {len(passed_tests)}")
    print(f"\t{Fore.CYAN} QUERIES WITH FAILED TESTS: {len(failed_tests)}")

    if len(passed_tests) > 0:
        print(f"\n\t{Fore.GREEN} PASSED ALL TESTS ON QUERIES:")
        for test in passed_tests:
            print(f"\t\t{Fore.GREEN}- {test['test_name']}")

    if len(failed_tests) > 0:
        print(f"\n\t{Fore.RED} FAILED TESTS ON QUERIES:")
        for test in failed_tests:
            print(f"\t\t{Fore.RED}- {test['test_name']}")
            print(f"\t\t{Fore.RED} TESTS FAILED ON THIS QUERY:")
            for failed_test in test['failed_tests']:
                print(f"\t\t\t{Fore.RED}- {consts.test_fail_explanations[failed_test]}")
        print(f"\n\t{Fore.RED} See above test results for more details on the failed tests.")
    else:
        print(f"\n\t{Fore.GREEN} ALL TESTS PASSED ON DATASET!")

    print(f"{Fore.CYAN} --------------------------------------------------------------")


def print_execution_summary(failed_on_datasets: List[str], passed_on_datasets: List[str]) -> None:
    """
    This function prints the summary of the test execution.
    It prints which datasets had every test pass, and which datasets had failed tests.
    :param failed_on_datasets: List of datasets that had failed tests.
    :param passed_on_datasets: List of datasets that had every test pass.
    :return: None
    """
    print(f"\n\n{Fore.CYAN} --------------------------------------------------------------")
    print(f"\t{Fore.CYAN} SUMMARY OF TEST EXECUTION:")
    print(f"\t{Fore.CYAN} DATASETS THAT PASSED ALL TESTS: {len(passed_on_datasets)}")
    print(f"\t{Fore.CYAN} DATASETS WITH FAILED TESTS: {len(failed_on_datasets)}")

    if len(passed_on_datasets) > 0:
        print(f"\n\t{Fore.GREEN} PASSED ALL TESTS ON DATASETS:")
        for dataset in passed_on_datasets:
            print(f"\t\t{Fore.GREEN}- {dataset}")

    if len(failed_on_datasets) > 0:
        print(f"\n\t{Fore.RED} FAILED TESTS ON DATASETS:")
        for dataset in failed_on_datasets:
            print(f"\t\t{Fore.RED}- {dataset}")
        print(f"\n\t{Fore.RED} See above test results for more details on the failed tests.")
    else:
        print(f"\n\t{Fore.GREEN} ALL TESTS PASSED ON ALL DATASETS!")
    print(f"{Fore.CYAN} --------------------------------------------------------------")
