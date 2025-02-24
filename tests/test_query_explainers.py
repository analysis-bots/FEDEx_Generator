import json

import numpy.random
import pandas as pd
from extract_intermediate_results import run_on_all_queries, Query, set_working_directory
import sys
import os
from typing import List
import io
from concurrent.futures import ThreadPoolExecutor
import warnings
import random

random.seed(42)
numpy.random.seed(42)
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
            if "final_df" in v and isinstance(v["final_df"], str):
                result[k]["final_df"] = pd.read_json(io.StringIO(v["final_df"]))

    return results


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
            print(
                f"\n  \n -------------------------------------------------- \n Comparing for query {k} \n --------------------------------------------------")
            matching_result_query = matching_result[k]

            for test_name, test_args in consts.test_funcs.items():
                # Get the attribute name to test, if it exists in the result.
                test_attribute = test_args['attribute_name']
                if test_attribute not in v:
                    continue
                # It may be required to rename duplicate columns or bins, if they exist (otherwise the test may
                # fail because of a comparison between two different columns with the same name).
                require_duplicate_fix = test_args['require_duplicate_fix']
                if require_duplicate_fix:
                    matching_result_query[test_attribute] = tu.fix_duplicate_col_names_and_bin_names(
                        matching_result_query[test_attribute])
                    v[test_attribute] = tu.fix_duplicate_col_names_and_bin_names(v[test_attribute])

                # Run the test and print the results.
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


def main(argv):
    set_working_directory()

    result_files = load_all_results(argv[1:] if len(argv) > 1 else None)

    for i in range(len(result_files)):
        result_files[i]['idx'] = i

    datasets = tu.load_datasets(result_files)

    # If a global column select is present in any result file, apply it to the relevant dataset, and save
    # a separate copy of the dataset with the global select applied and an identifier of the specific result.
    for result in result_files:
        if 'global_select' in result:
            dataset = datasets[result['dataset_name']]
            global_select = result['global_select']
            dataset_with_global_select = dataset[global_select]
            datasets[f"{result['dataset_name']}_{result['idx']}"] = dataset_with_global_select
            result['dataset_name'] = f"{result['dataset_name']}_{result['idx']}"
        if 'global_select_second' in result:
            dataset = datasets[result['second_dataset_name']]
            global_select = result['global_select_second']
            dataset_with_global_select = dataset[global_select]
            datasets[f"{result['second_dataset_name']}_{result['idx']}"] = dataset_with_global_select
            result['second_dataset_name'] = f"{result['second_dataset_name']}_{result['idx']}"

    result_queries = [
        [v['query_object'] for k, v in result.items() if isinstance(v, dict) and 'query_object' in v]
        for result in result_files
    ]

    print(
        "Reproducing results using the current implementation. Please be patient, this may take a while depending on the number of queries and datasets.\n")
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
    main(sys.argv)
