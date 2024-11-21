import json
import pandas as pd
from pandas import DataFrame
from extract_intermediate_results import run_on_all_queries, Query
import sys
import os
from typing import List, Dict
import io
from concurrent.futures import ThreadPoolExecutor
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import warnings
import result_comparison_functions as rcf
import random
random.seed(42)
import test_utils as tu

warnings.filterwarnings("ignore")
colorama_init()

test_fail_explanations = {
    "correlated attributes test": "The list of correlated attributes in the re-produced results does not match the list of correlated attributes in the saved results. There is likely an error in the method get_correlated_attributes of the Filter operation.",
    "measure scores test": "The measure scores in the re-produced results do not match the measure scores in the saved results. There is likely an error in the method calc_measure_internal of the tested measure.",
    "score dicts test": "The score dictionaries in the re-produced results do not match the score dictionaries in the saved results. There is likely an error in the method calc_measure_internal of the tested measure, or in calc_measure in BaseMeasure.",
    "influence values test": "The influence values in the re-produced results do not match the influence values in the saved results. There is likely an error in the method calc_influence_col of the tested measure."
}


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
    failed_results = []
    for result in result_files:
        print(f"\n \n -------------------------------------------------- \n"
              f"\033[1m Comparing results on dataset {result['dataset_name']} \033[0;0m \n --------------------------------------------------")
        matching_result = [r for r in re_produced_results if r['idx'] == result['idx']][0]
        failed_tests = []
        for k, v in result.items():
            if not isinstance(v, dict):
                continue
            failed_tests.append({
                'test_name': k,
                'passed': True,
                'failed_tests': []
            })
            print(f"\n Comparing query {k} \n --------------------------------------------------")
            matching_result_query = matching_result[k]

            if 'correlated_attributes' in v:
                print("Comparing correlated attributes: \n")
                passed_test = rcf.compare_correlated_attributes(v['correlated_attributes'], matching_result_query['correlated_attributes'])
                if not passed_test:
                    failed_tests[-1]['passed'] = False
                    failed_tests[-1]['failed_tests'].append('correlated attributes test')

            if 'measure_scores' in v:
                print("Comparing measure scores: \n")
                passed_test = rcf.compare_measure_scores(v['measure_scores'], matching_result_query['measure_scores'])
                if not passed_test:
                    failed_tests[-1]['passed'] = False
                    failed_tests[-1]['failed_tests'].append('measure scores test')

            if 'score_dict' in v:
                print("Comparing score dicts: \n")
                passed_test = rcf.compare_score_dicts(v['score_dict'], matching_result_query['score_dict'])
                if not passed_test:
                    failed_tests[-1]['passed'] = False
                    failed_tests[-1]['failed_tests'].append('score dicts test')

            if 'influence_vals' in v:
                print("Comparing influence values: \n")
                v['influence_vals'] = tu.fix_duplicate_col_names_and_bin_names(v['influence_vals'])
                matching_result_query['influence_vals'] = tu.fix_duplicate_col_names_and_bin_names(matching_result_query['influence_vals'])
                passed_test = rcf.compare_influence_vals(v['influence_vals'], matching_result_query['influence_vals'])
                if not passed_test:
                    failed_tests[-1]['passed'] = False
                    failed_tests[-1]['failed_tests'].append('influence values test')

            if 'significance_vals' in v:
                print("Comparing significance values: \n")
                v['significance_vals'] = tu.fix_duplicate_col_names_and_bin_names(v['significance_vals'])
                matching_result_query['significance_vals'] = tu.fix_duplicate_col_names_and_bin_names(matching_result_query['significance_vals'])
                passed_test = rcf.compare_significance_vals(v['significance_vals'], matching_result_query['significance_vals'])
                if not passed_test:
                    failed_tests[-1]['passed'] = False
                    failed_tests[-1]['failed_tests'].append('significance values test')



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