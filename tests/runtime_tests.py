"""
A script for testing the runtime of running queries.
This script expects 2 arguments: an array of paths to files containing the queries, and an array of paths to
files containing the datasets. Both must be in the same order.
This script only measures the overall execution time it takes to explain the queries, and not the time it takes
to run each part of the explanation generation process.
As such, it is recommended to run this script with a profiler to get a more detailed view of the runtime.
"""

from tests.extract_intermediate_results import get_queries, create_operation_object
from tests.test_utils import load_dataset
import time
import argparse


def main():
    # We expect 2 arguments: an array of paths to files containing the queries, and an array of paths to
    # files containing the datasets. Both must be in the same order.
    parser = argparse.ArgumentParser(description='Test the runtime of the queries.')
    parser.add_argument('--queries', type=str, nargs='+',
                        help='Paths to files containing the queries')
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='Paths to files containing the datasets')
    args = parser.parse_args()
    if len(args.queries) != len(args.datasets):
        raise ValueError("The number of queries and datasets must be the same.")
    if len(args.queries) == 0:
        raise ValueError("No queries were provided.")
    if len(args.datasets) == 0:
        raise ValueError("No datasets were provided.")
    runtimes = []
    for i in range(len(args.queries)):
        queries, global_select, _ = get_queries(args.queries[i])
        dataset = load_dataset(args.datasets[i], "", [{}])
        if global_select:
            dataset = dataset[global_select]
        for query in queries:
            # We skip join operations, since allowing them here would make this way more complicated.
            # They use mostly the same processes as the other operations, so we can safely assume their runtime
            # is similar to the other operations.
            # Outlier explainer is excluded because it used to be part of FEDEx_Generator, but was separated into
            # its own module. This is simply an artifact to prevent errors in case anything is left over.
            if query.operation == "join" or query.explainer == "outlier":
                continue
            operation = create_operation_object(query, dataset)
            start_time = time.time()
            operation.explain()
            end_time = time.time()
            print(f"Query {query} took {end_time - start_time} seconds to explain.")
            runtimes.append(end_time - start_time)

    print(f"Total time: {sum(runtimes)} seconds.")
    print(f"Average time: {sum(runtimes) / len(runtimes)} seconds.")
    print(f"Max time: {max(runtimes)} seconds.")
    print(f"Min time: {min(runtimes)} seconds.")
    print(f"Number of queries: {len(runtimes)}")





if __name__ == '__main__':
    main()