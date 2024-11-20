import json
import pandas as pd
from extract_intermediate_results import run_on_all_queries, Query
import sys
import os
from typing import List


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
            try:
                query = Query.from_string(k)
            except Exception as e:
                print(e)
                continue
            if "saved_results" in v:
                result[k]["saved_results"] = pd.read_json(v["saved_results"])


    return results


def main():
    result_files = load_all_results()
    print(result_files)


if __name__ == '__main__':
    main()