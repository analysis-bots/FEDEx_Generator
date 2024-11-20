"""
This script is used to extract intermediate results when creating query explanations.
The goal of it is to create a 'screenshot' of results at each step of the query explanation, which can be given to the unit tests,
for the sake of making sure that the results are consistent even after making changes to the code.

The script is meant to be used in the following way:
It accepts as arguments: <first_dataset> <query_file> <output_file> <second_dataset>
- The first dataset is the path to the dataset file. This one is required.
- The query file is a file containing lines of comma separated tuples of the form (column, operation, explainer, ...arguments),
    where column is the column name, operation, such as '<=', 'join', 'groupby', etc., explainer is the explainer to use such as 'fedex', 'outlier',
    and arguments is a dict of arguments for the operation and explainer. Note that if an argument is a list, it should use ';' as a separator instead of ','.
    Example query for filter operations: (column, <=, fedex, {value: 50, top_k: 2}).
    Example query for join operations: (column1, join, fedex, {column2: 'column2', top_k: 5}).
    Example query for groupby operations: (column, groupby, outlier, {agg_function: 'mean', top_k: 5, dir: 'high'}).
- The output file is the file where the intermediate results will be written. This is a dictionary of the form:
    {query: [function_name: intermediate_result]}. The name of the input file will be added as a "dataset" key.
- The second dataset is the path to the second dataset file. This is optional, and is used for join operations.
The output file will be a JSON file, and is meant for usage in the unit tests.

This script can be used with any dataset and query file, as long as the queries are formatted correctly.
"""
import numpy as np
from numpy import ndarray, dtype

from fedex_generator.Measures.Bins import Bins
from fedex_generator.Operations.Filter import operators, Filter
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Join import Join
from fedex_generator.Measures.DiversityMeasure import DiversityMeasure
from fedex_generator.Measures.ExceptionalityMeasure import ExceptionalityMeasure
from fedex_generator.Measures.OutlierMeasure import OutlierMeasure
from fedex_generator.commons import utils
from fedex_generator.Operations.Operation import Operation
from fedex_generator.Measures.BaseMeasure import BaseMeasure

import sys
import json
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame
from paretoset import paretoset

from fedex_generator.commons.consts import SIGNIFICANCE_THRESHOLD


@dataclass
class Query:
    column: str
    operation: str
    explainer: str
    arguments: Dict[str, str | int | float]

    @staticmethod
    def from_string(s: str) -> 'Query':
        """
        Create a Query object from a string representation.
        :param s: The string to convert.
        :return: A Query object.
        """
        s = (s.replace('(', '').replace(')', '')
             .replace("Querycolumn=",'').replace("explainer=", '')
             .replace("operation=",'').replace("arguments=",'')
             .replace("'",'').split(',', maxsplit=3))
        for i in range(3):
            s[i] = s[i].strip()
        s[3] = dict_string_to_dict(s[3])
        return Query(*s)

    def __str__(self):
        # Give a string representation of the query, but if any of the dict values are lists, replace the commas in the list with ';'.
        arguments = self.arguments
        if isinstance(arguments, dict):
            arguments = {k: v if type(v) != list else str(v).replace(',', ';') for k, v in arguments.items()}
        return f"Query(column={self.column}, operation={self.operation}, explainer={self.explainer}, arguments={arguments})"


def dict_string_to_dict(d: str) -> Dict[str, str | int | float]:
    """
    Convert a string representation of a dictionary to a dictionary.
    :param d: A string representation of a dictionary.
    :return: A dictionary.
    """
    return_dict = {}
    d = d.replace('{', '').replace('}', '').replace("'", '').split(',')
    if len(d) == 1 and d[0].strip() == '':
        return return_dict

    for item in d:
        key, value = item.split(':')
        value = value.split(';')

        if len(value) > 1:
            # In the case of a list of values, replace the brackets and strip the values.
            value = [v.replace('[', '').replace(']', '').replace('"','').strip() for v in value]
        else:
            value = value[0].strip()

        key = key.strip()
        # Try to convert the value to an int or float. If it fails, keep it as a string.
        if type(value) == list:
            for i in range(len(value)):
                try:
                    value[i] = int(value[i])
                except ValueError:
                    try:
                        value[i] = float(value[i])
                    except ValueError:
                        pass
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        return_dict[key] = value
    return return_dict


def get_queries(query_file: str) -> List[Query]:
    """
    Extract queries from the query file.
    :param query_file: A path to a file containing queries, as described above.
    :return: A list of queries
    """
    queries = []
    with open(query_file, 'r') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue
            query = line.strip().replace('(', '').replace(')', '').split(',', maxsplit=3)
            for i in range(3):
                query[i] = query[i].strip()
            # Convert the 3rd element to a dictionary
            query[3] = dict_string_to_dict(query[3])
            q = Query(*query)
            queries.append(q)
    f.close()
    return queries


def create_operation_object(query: Query, dataset: DataFrame, second_dataset: DataFrame = None):
    """
    Create an operation object based on the query.
    :param query: a Query object with all the necessary information.
    :param dataset: The first dataset. Mandatory.
    :param second_dataset: The second dataset. Optional, and only needed for join operations.
    :return: an operation object, with the necessary parameters and the operation performed.
    """
    if query.operation in operators:
        operation = Filter(
            source_df=dataset, source_scheme={},
            attribute=query.column, operation_str=query.operation,
            value=query.arguments['value'])
    elif query.operation == 'groupby':
        if query.arguments['select_columns'] is None:
            after_op = dataset.groupby(query.column).aggregate(query.arguments['agg_function'])
        else:
            after_op = dataset.groupby(query.column)[query.arguments['select_columns']].aggregate(
                query.arguments['agg_function'])
        operation = GroupBy(
            source_df=dataset, source_scheme={}, agg_dict={},
            group_attributes=[query.column], result_df=after_op
        )
    elif query.operation == 'join':
        if second_dataset is None:
            raise ValueError("Second dataset is required for join operations.")
        # Due to runtime constraints, we will never use more than 1000 rows.
        # This is fine because this script is only used for testing.
        if len(dataset) > 1000:
            dataset = dataset.head(1000)
        if len(second_dataset) > 1000:
            second_dataset = second_dataset.head(1000)
        operation = Join(
            left_df=dataset, right_df=second_dataset, source_scheme={},
            attribute=query.column
        )
    else:
        raise ValueError(f"Operation {query.operation} not supported.")
    return operation


def replicate_calc_measure(operation: Operation, measure: BaseMeasure) -> tuple[List[List[float]], Dict[str, tuple]]:
    """
    Replicates the calc_measure method from the BaseMeasure class, but with saving computation results along
    the way.\n
    For detailed documentation and information, see the BaseMeasure class in the fedex_generator package.
    :param operation: The operation object to use.
    :param measure: The measure object to use.
    :return: A list of measure scores, and the score dict.
    """
    measure_scores = []
    score_dict = {}
    for attr, dataset_relation in operation.iterate_attributes():
        measure_scores.append([])

        # Replicating the calc_measure method
        source_col, res_col = measure.get_source_and_res_cols(dataset_relation, attr)
        size = operation.get_bins_count()
        bin_candidates = Bins(source_col, res_col, size)

        measure_score = -np.inf
        for bin_ in bin_candidates.bins:
            measure_scores[-1].append(measure.calc_measure_internal(bin_))
            measure_score = max(measure_score, measure_scores[-1][-1])

        score_dict[attr] = (
            dataset_relation.get_source_name(), bin_candidates, measure_score, (source_col, res_col)
        )

        # Save the score dict without the bins, as it can't be serialized to JSON at the time of writing this.

        measure.score_dict = score_dict

        max_val = max([kl_val for _, _, kl_val, _ in score_dict.values()])
        measure.max_val = max_val

    return measure_scores, score_dict


def replicate_calc_influence(measure: BaseMeasure, score_dict: Dict[str, tuple], k: int) -> tuple[
    list[ndarray[Any, dtype[Any]]], list[Any], DataFrame]:
    """
    Replicates the calc_influence method from the BaseMeasure class, but with saving computation results along
    the way.\n
    For detailed documentation and information, see the BaseMeasure class in the fedex_generator package.
    :param measure: The measure object to use.
    :param score_dict: The score dict computed by the calc_measure method.
    :param k: The number of top explanations to return.
    :return: The influence values, the significance values, and the results.
    """

    saved_influence_vals = []
    significance_vals = []


    score_and_col = [(score_dict[col][2], col, score_dict[col][1], score_dict[col][3])
                     for col in score_dict]

    list_scores_sorted = score_and_col
    list_scores_sorted.sort()

    results_columns = ["score", "significance", "influence", "explanation", "bin", "influence_vals"]
    results = pd.DataFrame([], columns=results_columns)

    for score, max_col_name, bins, _ in list_scores_sorted[:-k - 1:-1]:
        source_name, bins, score, _ = score_dict[max_col_name]
        for current_bin in bins.bins:
            influence_vals = measure.get_influence_col(max_col_name, current_bin, False)
            influence_vals_list = np.array(list(influence_vals.values()))

            saved_influence_vals.append(influence_vals_list)

            if np.all(np.isnan(influence_vals_list)):
                continue

            max_values, max_influences = measure.get_max_k(influence_vals, 1)

            for max_value, influence_val in zip(max_values, max_influences):
                significance = measure.get_significance(influence_val, influence_vals_list)

                significance_vals.append(significance)

                if significance < SIGNIFICANCE_THRESHOLD:
                    continue

                explanation = measure.build_explanation(current_bin, max_col_name, max_value, source_name)

                new_result = dict(zip(results_columns,
                                      [score, significance, influence_val, explanation, current_bin,
                                       influence_vals,
                                       current_bin.get_bin_name(), max_col_name]))
                results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)

    # There's no choice but to drop the bin column, as it can't be serialized to JSON at the time of
    # writing this.
    results = results.drop(columns=['bin'])

    return saved_influence_vals, significance_vals, results


def fedex_explain_runner(query: Query, first_dataset: DataFrame, second_dataset: DataFrame = None) -> dict:
    """
    Run the steps of a query explanation using the fedex explainer.
    :param query: The query object.
    :param first_dataset: The first dataset.
    :param second_dataset: The second dataset.
    :return: A dictionary with the results.
    """
    operation = create_operation_object(query, first_dataset, second_dataset)

    if query.operation != 'groupby':
        measure = ExceptionalityMeasure()
    else:
        measure = DiversityMeasure()

    results_dict = {}

    # If the operation is a filter operation, we can get the correlated attributes.
    if query.operation != 'groupby' and query.operation != 'join':
        correlated_attributes = operation.get_correlated_attributes()
        results_dict['correlated_attributes'] = correlated_attributes
    # If the operation is a group-by operation, we can get the one-to-many attributes and the column names.
    elif query.operation == 'groupby':
        one_to_many_attributes = GroupBy.get_one_to_many_attributes(first_dataset, [query.column])
        column_names = operation._get_columns_names()
        results_dict['one_to_many_attributes'] = one_to_many_attributes
        results_dict['column_names'] = column_names
    # Join operations have no special attributes to extract.

    measure_scores, score_dict = replicate_calc_measure(operation, measure)
    saved_influence_vals, significance_vals, saved_results = replicate_calc_influence(measure, score_dict,
                                                                                      query.arguments['top_k'] if 'top_k' in query.arguments else 1)

    # Drop bins and columns from the score dicts, as they can not be serialized.
    score_dict = {k: (v[0], v[2]) for k, v in score_dict.items()}

    # Convert the np arrays to lists, as they can't be serialized to JSON.
    saved_influence_vals = [list(influence_vals) for influence_vals in saved_influence_vals]

    # Save all of the intermediate results in a dictionary
    results_dict['measure_scores'] = measure_scores
    results_dict['score_dict'] = score_dict
    results_dict['influence_vals'] = saved_influence_vals
    results_dict['significance_vals'] = significance_vals
    results_dict['saved_results'] = saved_results.to_json(orient='records')

    return results_dict

def dir_to_int(dir: str | int) -> int:
    """
    Convert a direction string to an integer.
    :param dir: The direction string.
    :return: The direction as an integer.
    """
    if dir == 'high':
        return 1
    elif dir == 'low':
        return -1
    else:
        return dir

def outlier_explain_runner(query: Query, first_dataset: DataFrame, second_dataset: DataFrame = None) -> dict:
    """
    Run the steps of a query explanation using the outlier explainer.
    :param query: The query object.
    :param first_dataset: The first dataset. Mandatory.
    :param second_dataset: The second dataset. Only needed if a join operation is performed. Can be None if not needed.
    Should not be needed since at the time of writing this, the outlier explainer only supports groupby operations.
    :return: A dictionary with the results.
    """
    operation = create_operation_object(query, first_dataset, second_dataset)

    results_dict = {}

    if query.operation != 'groupby':
        results_dict['error'] = 'Outlier explanation is currently only supported for groupby operations.'
        return results_dict

    measure = OutlierMeasure()

    preds = []

    attrs = first_dataset.columns
    attrs = [a for a in attrs if a not in [query.column, query.arguments['select_columns']]]

    # Do the predicate calculation for each attribute, like in the explain function.
    for attr in attrs:
        predicates = measure.compute_predicates_per_attribute(
            attr=attr,
            df_in=first_dataset,
            g_att=query.column,
            g_agg=query.arguments['select_columns'],
            agg_method=query.arguments['agg_function'],
            target=query.arguments['target'],
            dir=dir_to_int(query.arguments['dir']),
            df_in_consider=first_dataset,
            df_agg_consider=operation.result_df
        )

        preds += predicates

    preds.sort(key=lambda x: -x[2])

    results_dict['preds'] = preds

    # Use the merge_preds function to get the final results.
    final_pred, final_inf, final_df = measure.merge_preds(
        df_agg=operation.result_df,
        df_in=first_dataset,
        df_in_consider=first_dataset,
        preds=preds,
        g_att=query.column,
        g_agg=query.arguments['select_columns'],
        agg_method=query.arguments['agg_function'],
        target=query.arguments['target'],
        dir=dir_to_int(query.arguments['dir']),
    )

    results_dict['final_pred'] = final_pred
    results_dict['final_inf'] = final_inf
    results_dict['final_df'] = final_df.to_json(orient='records') if final_df is not None else None

    return results_dict



def run_on_all_queries(queries: List[Query], first_dataset: DataFrame, second_dataset: DataFrame = None) -> dict:
    """
    Loop over the queries, running the appropriate operations and explainers on each one and storing the results.
    :param queries: A list of Query objects.
    :param first_dataset: The first dataset. Mandatory.
    :param second_dataset: The second dataset. Only needed for join operations. Can be None if not needed.
    :return: A dictionary with the results of all the queries, where the keys are the queries.
    """
    all_results_dict = {}
    for query in queries:

        if query.explainer == 'fedex':
            all_results_dict[str(query)] = fedex_explain_runner(query, first_dataset, second_dataset)
        elif query.explainer == 'outlier':
            all_results_dict[str(query)] = outlier_explain_runner(query, first_dataset, second_dataset)
        else:
            all_results_dict[str(query)] = {'error': f"Explainer {query.explainer} not supported."}

    return all_results_dict


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python extract_intermediate_results.py <first_dataset> <query_file> <output_file> <optional: second_dataset>")
        sys.exit(1)

    input_file = sys.argv[1]
    query_file = sys.argv[2]
    output_file = sys.argv[3]
    second_dataset_path = sys.argv[4] if len(sys.argv) == 5 else None

    queries = get_queries(query_file)
    first_dataset = pd.read_csv(input_file)
    if second_dataset_path:
        second_dataset = pd.read_csv(second_dataset_path)
    else:
        second_dataset = None

    all_results_dict = run_on_all_queries(queries, first_dataset, second_dataset)



    # Save the paths to the datasets. This serves as a backup - in case the user did not provide file paths,
    # as well as for convenience - so the test can automatically run on the datasets without the need for manual input.
    all_results_dict['first_dataset'] = input_file
    all_results_dict['second_dataset'] = second_dataset_path
    # Find the dataset name from the input file path, depending on the OS.
    if input_file.find('/') != -1:
        all_results_dict['dataset_name'] = input_file.split('/')[-1].split('.')[0]
    else:
        all_results_dict['dataset_name'] = input_file.split('\\')[-1].split('.')[0]

    # Save the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(all_results_dict, f, indent=4)
    f.close()


if __name__ == '__main__':
    main()
