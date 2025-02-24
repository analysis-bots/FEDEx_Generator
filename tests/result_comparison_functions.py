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
