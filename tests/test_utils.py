from typing import List, Dict
from colorama import Fore
from colorama import init as colorama_init
import test_consts as consts

colorama_init(autoreset=True)

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

