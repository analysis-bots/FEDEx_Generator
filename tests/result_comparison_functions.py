"""
A collection of functions to compare the results of the query explainers.\n
These functions are used in the test_query_explainers.py file to compare the results of the saved query explainers to the re-produced query explainers.\n
These functions are mostly simple, and almost always compare both sides: first the saved results to the re-produced results, then the re-produced results to the saved results.\n
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from pandas import DataFrame, Series


def compare_correlated_attributes(saved_correlated_attributes: List[str],
                                  re_produced_correlated_attributes: List[str]) -> Tuple[bool, List[str]]:
    """
    Compare the list of correlated attributes in the saved results to the list of correlated attributes in the re-produced results.\n
    Order within the list does not matter.
    :param saved_correlated_attributes: The list of correlated attributes saved in the results.
    :param re_produced_correlated_attributes: The list of correlated attributes re-produced with the same query as the saved results.
    :return: True if the correlated attributes are the same, False otherwise, as well as a list of error messages.
    """
    error_messages = []
    passed = True
    for attr in saved_correlated_attributes:
        if attr not in re_produced_correlated_attributes:
            error_messages.append(
                f"Attribute '{attr}' not found in re-produced correlated attributes, but found in saved results.")
            passed = False
    for attr in re_produced_correlated_attributes:
        if attr not in saved_correlated_attributes:
            error_messages.append(
                f"Attribute '{attr}' not found in saved correlated attributes, but found in re-produced results.")
            passed = False

    return passed, error_messages


def compare_measure_scores(saved_measure_scores: [Dict[str, List[float]]],
                           re_produced_measure_scores: [Dict[str, List[float]]]) -> Tuple[bool, List[str]]:
    """
    Compare the measure scores in the saved results to the measure scores in the re-produced results.\n
    Order within the lists does not matter.\n
    :param saved_measure_scores: The measure scores saved in the results.
    :param re_produced_measure_scores: The measure scores re-produced with the same query as the saved results.
    :return: True if the measure scores are the same, False otherwise. Also returns a list of error messages.
    """
    passed = True
    error_messages = []
    for measure, scores in saved_measure_scores.items():
        if measure not in re_produced_measure_scores:
            error_messages.append(
                f"Measure '{measure}' not found in re-produced measure scores, but found in saved results.")
            passed = False
        else:
            for score in scores:
                if score not in re_produced_measure_scores[measure]:
                    error_messages.append(
                        f"Score '{score}' not found in re-produced measure scores for measure '{measure}', but found in saved results.")
                    passed = False
    for measure, scores in re_produced_measure_scores.items():
        if measure not in saved_measure_scores:
            error_messages.append(
                f"Measure '{measure}' not found in saved measure scores, but found in re-produced results.")
            passed = False
        else:
            for score in scores:
                if score not in saved_measure_scores[measure]:
                    error_messages.append(
                        f"Score '{score}' not found in saved measure scores for measure '{measure}', but found in re-produced results.")
                    passed = False

    return passed, error_messages


def compare_score_dicts(saved_scores: Dict[str, Tuple[str, float]], re_produced_scores: Dict[str, Tuple[str, float]]) -> \
        Tuple[bool, List[str]]:
    """
    Compare the score dicts in the saved results to the score dicts in the re-produced results.\n
    :param saved_scores: The saved score dict.
    :param re_produced_scores: The re-produced score dict.
    :return: True if the score dicts are the same, False otherwise. Also returns a list of error messages.
    """
    error_messages = []
    passed = True
    for k, v in saved_scores.items():
        if k not in re_produced_scores:
            error_messages.append(f"Key '{k}' not found in re-produced scores, but found in saved results.")
            passed = False
        else:
            if tuple(v)[1] != re_produced_scores[k][1]:
                error_messages.append(
                    f"Value '{v[1]}' for key '{k}' does not match re-produced value '{re_produced_scores[k][1]}'.")
                passed = False
    for k, v in re_produced_scores.items():
        if k not in saved_scores:
            error_messages.append(f"Key '{k}' not found in saved scores, but found in re-produced results.")
            passed = False
        else:
            if v[1] != tuple(saved_scores[k])[1]:
                error_messages.append(
                    f"Value '{v[1]}' for key '{k}' does not match saved value '{saved_scores[k][1]}'.")
                passed = False

    return passed, error_messages


def compare_influence_vals(saved_influence_vals: List[Dict], re_produced_influence_vals: List[Dict]) -> Tuple[
    bool, List[str]]:
    """
    Compare the influence values in the saved results to the influence values in the re-produced results.\n
    Order within the lists does not matter.\n
    :param saved_influence_vals: The influence values saved in the results.
    :param re_produced_influence_vals: The influence values re-produced with the same query as the saved results.
    :return: True if the influence values are the same, False otherwise.
    """
    passed = True
    error_messages = []
    # This dict is used to avoid printing the same error multiple times.
    seen_errors = defaultdict(set)
    for vals in saved_influence_vals:
        saved_column = vals['column']
        saved_bin = vals['bin']
        values = vals['values']
        found = False
        for re_vals in re_produced_influence_vals:
            re_column = re_vals['column']
            re_bin = re_vals['bin']
            re_values = re_vals['values']
            seen_errors[(saved_column, saved_bin)] = set()
            if saved_column == re_column and saved_bin == re_bin:
                found = True
                for v in values:
                    if v not in re_values and v not in seen_errors[(saved_column, saved_bin)]:
                        error_messages.append(
                            f"Value '{v}' not found in re-produced influence values for column '{saved_column}' and bin '{saved_bin}', but found in saved results.")
                        passed = False
                        seen_errors[(saved_column, saved_bin)].add(v)
                for v in re_values:
                    if v not in values and v not in seen_errors[(saved_column, saved_bin)]:
                        error_messages.append(
                            f"Value '{v}' not found in saved influence values for column '{re_column}' and bin '{re_bin}', but found in re-produced results.")
                        passed = False
                        seen_errors[(saved_column, saved_bin)].add(v)
                break
        if not found:
            error_messages.append(
                f"Column '{saved_column}' and bin '{saved_bin}' not found in re-produced influence values, but found in saved results.")
            passed = False

    for vals in re_produced_influence_vals:
        if (vals['column'], vals['bin']) not in seen_errors:
            seen_errors[(vals['column'], vals['bin'])] = set()
        re_column = vals['column']
        re_bin = vals['bin']
        re_values = vals['values']
        found = False
        for saved_vals in saved_influence_vals:
            saved_column = saved_vals['column']
            saved_bin = saved_vals['bin']
            values = saved_vals['values']
            if re_column == saved_column and re_bin == saved_bin:
                found = True
                for v in re_values:
                    if v not in values and v not in seen_errors[(re_column, re_bin)]:
                        error_messages.append(
                            f"Value '{v}' not found in saved influence values for column '{re_column}' and bin '{re_bin}', but found in re-produced results.")
                        passed = False
                        seen_errors[(re_column, re_bin)].add(v)
                for v in values:
                    if v not in re_values and v not in seen_errors[(re_column, re_bin)]:
                        error_messages.append(
                            f"Value '{v}' not found in re-produced influence values for column '{saved_column}' and bin '{saved_bin}', but found in saved results.")
                        passed = False
                        seen_errors[(re_column, re_bin)].add(v)
                break
        if not found:
            error_messages.append(
                f"Column '{re_column}' and bin '{re_bin}' not found in saved influence values, but found in re-produced results.")
            passed = False

    return passed, error_messages


def compare_significance_vals(saved_significance_vals: List[Dict], re_produced_significance_vals: List[Dict]) -> Tuple[
    bool, List[str]]:
    """
    Compare the significance values in the saved results to the significance values in the re-produced results.\n
    Order within the lists does not matter.
    :param saved_significance_vals: The significance values saved in the results.
    :param re_produced_significance_vals: The significance values re-produced with the same query as the saved results.
    :return: True if the significance values are the same, False otherwise. Also returns a list of error messages.
    """
    passed = True
    error_messages = []
    # This dict is used to avoid printing the same error multiple times.
    seen_errors = defaultdict(set)
    for vals in saved_significance_vals:
        seen_errors[(vals['column'], vals['bin'])] = set()
        saved_column = vals['column']
        saved_bin = vals['bin']
        saved_significance = vals['significance']
        found = False
        for re_vals in re_produced_significance_vals:
            re_column = re_vals['column']
            re_bin = re_vals['bin']
            re_significance = re_vals['significance']
            if saved_column == re_column and saved_bin == re_bin:
                found = True
                if saved_significance != re_significance and (saved_significance, re_significance) not in seen_errors[
                    (saved_column, saved_bin)]:
                    error_messages.append(
                        f"Saved significance '{saved_significance}' does not match re-produced significance '{re_significance}' for column '{saved_column}' and bin '{saved_bin}'.")
                    passed = False
                    seen_errors[(saved_column, saved_bin)].add((saved_significance, re_significance))
                break
        if not found:
            error_messages.append(
                f"Column '{saved_column}' and bin '{saved_bin}' not found in re-produced significance values, but found in saved results.")
            passed = False

    for vals in re_produced_significance_vals:
        if (vals['column'], vals['bin']) not in seen_errors:
            seen_errors[(vals['column'], vals['bin'])] = set()
        re_column = vals['column']
        re_bin = vals['bin']
        re_significance = vals['significance']
        found = False
        for saved_vals in saved_significance_vals:
            saved_column = saved_vals['column']
            saved_bin = saved_vals['bin']
            saved_significance = saved_vals['significance']
            if re_column == saved_column and re_bin == saved_bin:
                found = True
                if re_significance != saved_significance and (saved_significance, re_significance) not in seen_errors[
                    (re_column, re_bin)]:
                    error_messages.append(
                        f"Re-produced significance '{re_significance}' does not match saved significance '{saved_significance}' for column '{re_column}' and bin '{re_bin}'.")
                    passed = False
                    seen_errors[(re_column, re_bin)].add((saved_significance, re_significance))
                break
        if not found:
            error_messages.append(
                f"Column '{re_column}' and bin '{re_bin}' not found in saved significance values, but found in re-produced results.")
            passed = False

    return passed, error_messages


def compare_results(saved_results: DataFrame, re_produced_results: DataFrame) -> Tuple[bool, List[str]]:
    """
    Compare the results of the saved DataFrame to the results of the re-produced DataFrame.\n
    :param saved_results: The saved results DataFrame.
    :param re_produced_results: The re-produced results DataFrame.
    :return: True if the DataFrames are the same, False otherwise. Also returns a list of error messages.
    """
    passed = True
    error_messages = []
    if saved_results.shape != re_produced_results.shape:
        error_dims = []
        for i in range(len(saved_results.shape)):
            if saved_results.shape[i] != re_produced_results.shape[i]:
                error_dims.append(i)
        error_messages.append(
            f"Saved results shape '{saved_results.shape}' does not match re-produced results shape '{re_produced_results.shape}'. Inconsistent dimensions: {error_dims}.")
        passed = False
    else:
        # Check that each row exists in the re-produced results.
        for row in range(saved_results.shape[0]):
            saved_row = saved_results.iloc[row]
            found = False
            for re_row in range(re_produced_results.shape[0]):
                re_produced_row = re_produced_results.iloc[re_row]
                # We don't check for influence vals because influence vals already have their own comparison function.
                # We use the isclose function to compare the float values, because the precision of the loaded values may differ from the computed values.
                if (np.isclose(saved_row['score'], re_produced_row['score'])
                        and np.isclose(saved_row['significance'], re_produced_row['significance'])
                        and np.isclose(saved_row['influence'], re_produced_row['influence'])
                        and saved_row['explanation'] == re_produced_row['explanation']):
                    found = True
                    break
            if not found:
                error_messages.append(f"Row {row} with values: "
                                      f"score={saved_row['score']}, "
                                      f"significance={saved_row['significance']}, "
                                      f"influence={saved_row['influence']}, "
                                      f"explanation={saved_row['explanation']}, "
                                      f"not found in re-produced results, but found in saved results.")
                passed = False

        for row in range(re_produced_results.shape[0]):
            re_produced_row = re_produced_results.iloc[row]
            found = False
            for saved_row in range(saved_results.shape[0]):
                saved_row = saved_results.iloc[saved_row]
                if (np.isclose(saved_row['score'], re_produced_row['score'])
                        and np.isclose(saved_row['significance'], re_produced_row['significance'])
                        and np.isclose(saved_row['influence'], re_produced_row['influence'])
                        and saved_row['explanation'] == re_produced_row['explanation']):
                    found = True
                    break
            if not found:
                error_messages.append(f"Row {row} with values: "
                                      f"score={re_produced_row['score']}, "
                                      f"significance={re_produced_row['significance']}, "
                                      f"influence={re_produced_row['influence']}, "
                                      f"explanation={re_produced_row['explanation']}, "
                                      f"not found in saved results, but found in re-produced results.")
                passed = False

    return passed, error_messages


def compare_column_names(saved_column_names: List[str], re_produced_column_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Compare the column names in the saved results to the column names in the re-produced results.\n
    Order within the lists does not matter.
    :param saved_column_names: The column names saved in the results.
    :param re_produced_column_names: The column names re-produced with the same query as the saved results.
    :return: True if the column names are the same, False otherwise. Also returns a list of error messages.
    """
    error_messages = []
    passed = True
    for col in saved_column_names:
        if col not in re_produced_column_names:
            error_messages.append(f"Column '{col}' not found in re-produced column names, but found in saved results.")
            passed = False
    for col in re_produced_column_names:
        if col not in saved_column_names:
            error_messages.append(f"Column '{col}' not found in saved column names, but found in re-produced results.")
            passed = False

    return passed, error_messages


def compare_one_to_many_attributes(saved_one_to_many_attributes: List[str],
                                   re_produced_one_to_many_attributes: List[str]) -> Tuple[bool, List[str]]:
    """
    Compare the one-to-many attributes in the saved results to the one-to-many attributes in the re-produced results.\n
    Order within the lists does not matter.
    :param saved_one_to_many_attributes: The one-to-many attributes saved in the results.
    :param re_produced_one_to_many_attributes: The one-to-many attributes re-produced with the same query as the saved results.
    :return: True if the one-to-many attributes are the same, False otherwise. Also returns a list of error messages.
    """
    error_messages = []
    passed = True
    for attr in saved_one_to_many_attributes:
        if attr not in re_produced_one_to_many_attributes:
            error_messages.append(
                f"Attribute '{attr}' not found in re-produced one-to-many attributes, but found in saved results.")
            passed = False
    for attr in re_produced_one_to_many_attributes:
        if attr not in saved_one_to_many_attributes:
            error_messages.append(
                f"Attribute '{attr}' not found in saved one-to-many attributes, but found in re-produced results.")
            passed = False

    return passed, error_messages


def compare_predicates(saved_predicates: List[Tuple[str, Tuple[float, float] | str, float, str, int | None]],
                       re_produced_predicates: List[Tuple[str, Tuple[float, float] | str, float, str, int | None]]) -> \
        Tuple[bool, List[str]]:
    """
    Compare the predicates in the saved results to the predicates in the re-produced results.\n
    Order within the lists matter, as the predicates are saved after sorting them.
    :param saved_predicates: The predicates saved in the results.
    :param re_produced_predicates: The predicates re-produced with the same query as the saved results.
    :return: True if the predicates are the same, False otherwise. Also returns a list of error messages.
    """
    error_messages = []
    saved_len = len(saved_predicates)
    re_produced_len = len(re_produced_predicates)
    if saved_len != re_produced_len:
        error_messages.append(
            f"Saved predicates length '{saved_len}' does not match re-produced predicates length '{re_produced_len}'.")
        passed = False
    passed = True
    for i in range(len(saved_predicates)):
        try:
            saved_pred = saved_predicates[i]
            re_produced_pred = re_produced_predicates[i]
            try:
                local_errors = []
                saved_col = saved_pred[0]
                re_produced_col = re_produced_pred[0]
                # The second item can be either a bin or a category. A bin is a tuple of floats, a category is a string.
                # Due to saving and loading, the bin may be a list instead of a tuple.
                saved_cat_or_bin = tuple(saved_pred[1]) if (
                        isinstance(saved_pred[1], tuple) or isinstance(saved_pred[1], list)) else saved_pred[1]
                re_produced_cat_or_bin = tuple(re_produced_pred[1]) if (
                        isinstance(re_produced_pred[1], tuple) or isinstance(re_produced_pred[1], list)) else \
                    re_produced_pred[1]
                saved_influence = saved_pred[2]
                re_produced_influence = re_produced_pred[2]
                saved_type = saved_pred[3]
                re_produced_type = re_produced_pred[3]
                saved_index = saved_pred[4]
                re_produced_index = re_produced_pred[4]
                num_errors = 0
                if saved_col != re_produced_col:
                    local_errors.append(
                        f"\t\t- Saved column '{saved_col}' does not match re-produced column '{re_produced_col}'.")
                    num_errors += 1
                if saved_cat_or_bin != re_produced_cat_or_bin:
                    local_errors.append(
                        f"\t\t- Saved category or bin '{saved_cat_or_bin}' does not match re-produced category or bin '{re_produced_cat_or_bin}'.")
                    num_errors += 1
                if not np.isclose(saved_influence, re_produced_influence):
                    local_errors.append(
                        f"\t\t- Saved influence '{saved_influence}' does not match re-produced influence '{re_produced_influence}'.")
                    num_errors += 1
                if saved_type != re_produced_type:
                    local_errors.append(
                        f"\t\t- Saved type '{saved_type}' does not match re-produced type '{re_produced_type}'.")
                    num_errors += 1
                if saved_index != re_produced_index:
                    local_errors.append(
                        f"\t\t- Saved index '{saved_index}' does not match re-produced index '{re_produced_index}'.")
                if num_errors != 0:
                    error_messages.append(
                        f"\t{num_errors} errors in predicate {i}. Saved predicate: {saved_pred}, re-produced predicate: {re_produced_pred}")
                    error_messages += local_errors
                    passed = False
            except IndexError:
                if len(saved_pred) != 5 and len(re_produced_pred) != 5:
                    error_messages.append(
                        f"Saved predicate and re-produced predicate both have a different length than expected. If this is due to an intentional change, please ignore this error or update the comparison function. Saved predicate: {saved_pred}, re-produced predicate: {re_produced_pred}")
                elif len(saved_pred) != 5:
                    error_messages.append(
                        f"Saved predicate {saved_pred} has a different length from expected, and may be deformed.")
                elif len(re_produced_pred) != 5:
                    error_messages.append(
                        f"Re-produced predicate {re_produced_pred} has a different length from expected, and may be deformed.")
                passed = False
        # If the index is out of range, the saved results are longer than the re-produced results.
        # We don't report this separately, as it is already reported in the length comparison.
        except IndexError:
            passed = False
            break

    if len(re_produced_predicates) > len(saved_predicates):
        error_messages.append(
            "Re-produced predicates are longer than saved predicates. Additional predicates found in re-produced results.")
    for i in range(len(saved_predicates), len(re_produced_predicates)):
        error_messages.append(f"\tRe-produced predicate {re_produced_predicates[i]} not found in saved predicates.")

    return passed, error_messages


def compare_final_inf(saved_final_inf: float, re_produced_final_inf: float) -> Tuple[bool, List[str]]:
    """
    Compare the final influence values in the saved results to the final influence values in the re-produced results.\n
    :param saved_final_inf: The final influence value saved in the results.
    :param re_produced_final_inf: The final influence value re-produced with the same query as the saved results.
    :return: True if the final influence values are the same, False otherwise. Also returns a list of error messages.
    """
    error_messages = []
    passed = True
    if saved_final_inf != re_produced_final_inf:
        error_messages.append(
            f"Saved final influence value '{saved_final_inf}' does not match re-produced final influence value '{re_produced_final_inf}'.")
        passed = False

    return passed, error_messages


def compare_final_predicate(saved_final_pred: List[Tuple[str, Tuple[float, float] | str, float, str, int | None]],
                            re_produced_final_pred: List[
                                Tuple[str, Tuple[float, float] | str, float, str, int | None]]) -> Tuple[
    bool, List[str]]:
    """
    Compare the final predicate in the saved results to the final predicate in the re-produced results.\n
    :param saved_final_pred: The final predicate saved in the results.
    :param re_produced_final_pred: The final predicate re-produced with the same query as the saved results.
    :return: True if the final predicate is the same, False otherwise. Also returns a list of error messages.
    """
    # Due to loading from a json file, the tuples may be converted into lists, which will cause the comparison to fail.
    # We convert the lists back into tuples to prevent this.
    for i in range(len(saved_final_pred)):
        saved_final_pred[i] = tuple(saved_final_pred[i])

    error_messages = []
    passed = True
    if saved_final_pred != re_produced_final_pred:
        error_messages.append(
            f"Saved final predicate '{saved_final_pred}' does not match re-produced final predicate '{re_produced_final_pred}'.")
        passed = False

    return passed, error_messages


def rows_are_close(row1: Series, row2: Series) -> bool:
    """
    Check if two rows are close to each other.\n
    :param row1: First row to compare.
    :param row2: Second row to compare.
    :return: True if the rows are close, False otherwise.
    """
    if row1.equals(row2):
        return True
    for i in range(len(row1)):
        if (isinstance(row1[i], float) or isinstance(row1[i], int)) and (
                isinstance(row2[i], float) or isinstance(row2[i], int)):
            if not np.isclose(row1[i], row2[i]):
                return False
        elif row1[i] != row2[i]:
            return False
    return True


def compare_final_df(saved_final_df: DataFrame, re_produced_final_df: DataFrame) -> Tuple[bool, List[str]]:
    """
    Compare the final DataFrame in the saved results to the final DataFrame in the re-produced results.\n
    :param saved_final_df: The final DataFrame saved in the results.
    :param re_produced_final_df: The final DataFrame re-produced with the same query as the saved results.
    :return: True if the final DataFrames are the same, False otherwise. Also returns a list of error messages.
    """

    # It is possible that the representation of the final DataFrame is a Series, not a DataFrame.
    # In this case, we convert it into a d x 1 DataFrame. This is done to prevent index errors
    # which may happen because of the way we load the saved DataFrame.
    if isinstance(re_produced_final_df, Series):
        re_produced_final_df = re_produced_final_df.to_frame()
    # Same for the saved final DataFrame.
    if isinstance(saved_final_df, Series):
        saved_final_df = saved_final_df.to_frame()

    passed = True
    error_messages = []
    if saved_final_df is None and re_produced_final_df is None:
        return True, []
    elif saved_final_df is None:
        error_messages.append("Saved final DataFrame is None, but re-produced final DataFrame is not None.")
        return False, error_messages
    elif re_produced_final_df is None:
        error_messages.append("Saved final DataFrame is not None, but re-produced final DataFrame is None.")
        return False, error_messages

    if saved_final_df.shape != re_produced_final_df.shape:
        error_dims = []
        for i in range(len(saved_final_df.shape)):
            if saved_final_df.shape[i] != re_produced_final_df.shape[i]:
                error_dims.append(i)
        error_messages.append(
            f"Saved final DataFrame shape '{saved_final_df.shape}' does not match re-produced final DataFrame shape '{re_produced_final_df.shape}'. Inconsistent dimensions: {error_dims}.")
        passed = False
    # Check that the final DataFrames are the same. There is no specific scheme to follow - these are datasets.
    # We only check that the values are the same.
    for row in range(saved_final_df.shape[0]):
        saved_row = saved_final_df.iloc[row]
        found = False
        for re_row in range(re_produced_final_df.shape[0]):
            re_produced_row = re_produced_final_df.iloc[re_row]
            # We use a close comparison function to compare the float values, because the precision of the loaded values may differ from the computed values.
            if rows_are_close(saved_row, re_produced_row):
                found = True
                break
        if not found:
            error_messages.append(
                f"Row {row} with values: {saved_row} not found in re-produced final DataFrame, but found in saved final DataFrame.")
            passed = False

    for row in range(re_produced_final_df.shape[0]):
        re_produced_row = re_produced_final_df.iloc[row]
        found = False
        for saved_row in range(saved_final_df.shape[0]):
            saved_row = saved_final_df.iloc[saved_row]
            if rows_are_close(re_produced_row, saved_row):
                found = True
                break
        if not found:
            error_messages.append(
                f"Row {row} with values: {re_produced_row} not found in saved final DataFrame, but found in re-produced final DataFrame.")
            passed = False

    return passed, error_messages
