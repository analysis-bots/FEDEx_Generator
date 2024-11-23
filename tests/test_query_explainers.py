import json
import pandas as pd
from pandas import DataFrame
from extract_intermediate_results import run_on_all_queries, Query
import sys
import os
from typing import List, Dict
import io
from concurrent.futures import ThreadPoolExecutor
import warnings
import random
random.seed(42)
import test_utils as tu
import test_consts as consts

warnings.filterwarnings("ignore")




default_results_path = "resources/results_files"

def load_all_results(paths: List[str] | None = None) -> List[dict]:
    """
    Load all result files from the given paths.
    If no paths are given, all files from the default results directory are loaded.
    If paths are provided, it must be a list of paths to the result files (and not to directories).
    :param paths: List of paths to result files. Leave None to load all files from the default results directory.
    :return: List of dictionaries, each containing the content of a result file.
    """
    if paths is None:
        paths = [f"{default_results_path}/{file}" for file in os.listdir(default_results_path)]
    results = []
    for path in paths:
        with open(path, "r") as file:
            results.append(json.load(file))
        file.close()


    # Convert the dataframes that were converted to strings back to dataframes.
    for result in results:
        for k, v in result.items():
            # Try converting the key to a query. If this works, the value is a dict.
            # Otherwise, it's one of the metadata fields, like "dataset_name" or "dataset_path".
            try:
                query = Query.from_string(k)
                v['query_object'] = query
            # Index error occurs when the key is not a query.
            except IndexError:
                continue
            if "saved_results" in v:
                # Wrapping the string in a StringIO object to be able to read it as a file.
                result[k]["saved_results"] = pd.read_json(io.StringIO(v["saved_results"]))


    return results


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


def load_datasets(results: List[dict], default_configuration_path: str = "resources/default_datasets.json") -> Dict[str, DataFrame]:
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





def compare_results(result_files: List[dict], re_produced_results: List[dict]):
    failed_on_datasets = []
    passed_on_datasets = []
    for result in result_files:
        print(f"\n \n -------------------------------------------------- \n"
              f"\033[1m Comparing results on dataset {result['dataset_name']} \033[0;0m \n --------------------------------------------------")
        matching_result = [r for r in re_produced_results if r['idx'] == result['idx']][0]
        test_outcomes = []
        for k, v in result.items():
            if not isinstance(v, dict):
                continue
            test_outcomes.append({
                'test_name': k,
                'passed': True,
                'failed_tests': []
            })
            print(f"\n  \n -------------------------------------------------- \n Comparing for query {k} \n --------------------------------------------------")
            matching_result_query = matching_result[k]

            for test_name, test_args in consts.test_funcs.items():
                test_attribute = test_args['attribute_name']
                if test_attribute not in v:
                    continue
                require_duplicate_fix = test_args['require_duplicate_fix']
                if require_duplicate_fix:
                    matching_result_query[test_attribute] = tu.fix_duplicate_col_names_and_bin_names(matching_result_query[test_attribute])
                    v[test_attribute] = tu.fix_duplicate_col_names_and_bin_names(v[test_attribute])

                test_passed, errors = test_args['func'](v[test_attribute], matching_result_query[test_attribute])
                tu.print_test_messages(errors, test_name, test_passed)
                if not test_passed:
                    test_outcomes[-1]['passed'] = False
                    test_outcomes[-1]['failed_tests'].append(test_name)

        tu.print_result_summary(test_outcomes, result['dataset_name'])
        if not all([outcome['passed'] for outcome in test_outcomes]):
            failed_on_datasets.append(result['dataset_name'])
        else:
            passed_on_datasets.append(result['dataset_name'])

    tu.print_execution_summary(failed_on_datasets, passed_on_datasets)





def main():
    result_files = load_all_results(sys.argv[1:] if len(sys.argv) > 1 else None)

    for i in range(len(result_files)):
        result_files[i]['idx'] = i

    datasets = load_datasets(result_files)
    result_queries = [
        [v['query_object'] for k, v in result.items() if isinstance(v, dict) and 'query_object' in v]
        for result in result_files
    ]

    # Run all the queries on the datasets using the current implementation.
    re_produced_results = []

    # Use a thread pool to run the queries in parallel.
    with ThreadPoolExecutor() as executor:
        re_produced_results = list(executor.map(
            run_on_all_queries,
            result_queries,
            [datasets[result['dataset_name']] for result in result_files],
            [datasets[result['second_dataset_name']] for result in result_files],
            range(len(result_files)),
            [False] * len(result_files)
        ))


    compare_results(result_files, re_produced_results)



if __name__ == '__main__':
    main()