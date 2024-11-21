import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()
from typing import List, Dict, Tuple
from collections import defaultdict


def compare_correlated_attributes(saved_correlated_attributes: List[str], re_produced_correlated_attributes: List[str]) -> bool:
    """
    Compare the list of correlated attributes in the saved results to the list of correlated attributes in the re-produced results.\n
    Order within the list does not matter.\n
    Prints out any discrepancies found.
    :param saved_correlated_attributes: The list of correlated attributes saved in the results.
    :param re_produced_correlated_attributes: The list of correlated attributes re-produced with the same query as the saved results.
    :return: True if the lists are the same, False otherwise.
    """
    passed = True
    for attr in saved_correlated_attributes:
        if attr not in re_produced_correlated_attributes:
            print(f"\t {Fore.RED}Attribute '{attr}' not found in re-produced correlated attributes, but found in saved results.{Style.RESET_ALL}")
            passed = False
    for attr in re_produced_correlated_attributes:
        if attr not in saved_correlated_attributes:
            print(f"\t {Fore.RED}Attribute '{attr}' not found in saved correlated attributes, but found in re-produced results.{Style.RESET_ALL}")
            passed = False
    if passed:
        print(f"\t {Fore.GREEN}Correlated attributes comparison passed.{Style.RESET_ALL}", )
        print(f"{Fore.GREEN} -------------------------------------------------- {Style.RESET_ALL}")
    else:
        print(f"\n \t {Fore.RED}Correlated attributes comparison failed.{Style.RESET_ALL}")
        print(f"{Fore.RED} -------------------------------------------------- {Style.RESET_ALL}")

    return passed


def compare_measure_scores(saved_measure_scores: [Dict[str, List[float]]], re_produced_measure_scores: [Dict[str, List[float]]]) -> bool:
    """
    Compare the measure scores in the saved results to the measure scores in the re-produced results.\n
    Order within the lists does not matter.\n
    Prints out any discrepancies found.
    :param saved_measure_scores: The measure scores saved in the results.
    :param re_produced_measure_scores: The measure scores re-produced with the same query as the saved results.
    :return: True if the measure scores are the same, False otherwise.
    """
    passed = True
    for measure, scores in saved_measure_scores.items():
        if measure not in re_produced_measure_scores:
            print(f"\t {Fore.RED}Measure '{measure}' not found in re-produced measure scores, but found in saved results.{Style.RESET_ALL}")
            passed = False
        else:
            for score in scores:
                if score not in re_produced_measure_scores[measure]:
                    print(f"\t {Fore.RED}Score '{score}' not found in re-produced measure scores for measure '{measure}', but found in saved results.{Style.RESET_ALL}")
                    passed = False
    for measure, scores in re_produced_measure_scores.items():
        if measure not in saved_measure_scores:
            print(f"\t {Fore.RED}Measure '{measure}' not found in saved measure scores, but found in re-produced results.{Style.RESET_ALL}")
            passed = False
        else:
            for score in scores:
                if score not in saved_measure_scores[measure]:
                    print(f"\t {Fore.RED}Score '{score}' not found in saved measure scores for measure '{measure}', but found in re-produced results.{Style.RESET_ALL}")
                    passed = False

    if passed:
        print(f"\t {Fore.GREEN}Measure scores comparison passed.{Style.RESET_ALL}")
        print(f"{Fore.GREEN} -------------------------------------------------- {Style.RESET_ALL}")
    else:
        print(f"\n \t {Fore.RED}Measure scores comparison failed.{Style.RESET_ALL}")
        print(f"{Fore.RED} -------------------------------------------------- {Style.RESET_ALL}")

    return passed

def compare_score_dicts(saved_scores: Dict[str, Tuple[str, float]], re_produced_scores: Dict[str, Tuple[str, float]]) -> bool:
    """
    Compare the score dicts in the saved results to the score dicts in the re-produced results.\n
    Prints out any discrepancies found.
    :param saved_scores: The saved score dict.
    :param re_produced_scores: The re-produced score dict.
    :return: True if the score dicts are the same, False otherwise.
    """
    passed = True
    for k, v in saved_scores.items():
        if k not in re_produced_scores:
            print(f"\t {Fore.RED}Key '{k}' not found in re-produced scores, but found in saved results.{Style.RESET_ALL}")
            passed = False
        else:
            if tuple(v)[1] != re_produced_scores[k][1]:
                print(f"\t {Fore.RED}Value '{v[1]}' for key '{k}' does not match re-produced value '{re_produced_scores[k][1]}'.{Style.RESET_ALL}")
                passed = False
    for k, v in re_produced_scores.items():
        if k not in saved_scores:
            print(f"\t {Fore.RED}Key '{k}' not found in saved scores, but found in re-produced results.{Style.RESET_ALL}")
            passed = False
        else:
            if v[1] != tuple(saved_scores[k])[1]:
                print(f"\t {Fore.RED}Value '{v[1]}' for key '{k}' does not match saved value '{saved_scores[k][1]}'.{Style.RESET_ALL}")
                passed = False

    if passed:
        print(f"\t {Fore.GREEN}Score dicts comparison passed.{Style.RESET_ALL}")
        print(f"{Fore.GREEN} -------------------------------------------------- {Style.RESET_ALL}")
    else:
        print(f"\n \t {Fore.RED}Score dicts comparison failed.{Style.RESET_ALL}")
        print(f"{Fore.RED} -------------------------------------------------- {Style.RESET_ALL}")

    return passed


def compare_influence_vals(saved_influence_vals: List[Dict], re_produced_influence_vals: List[Dict]) -> bool:
    """
    Compare the influence values in the saved results to the influence values in the re-produced results.\n
    Order within the lists does not matter.\n
    Prints out any discrepancies found.
    :param saved_influence_vals: The influence values saved in the results.
    :param re_produced_influence_vals: The influence values re-produced with the same query as the saved results.
    :return: True if the influence values are the same, False otherwise.
    """
    passed = True
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
                        print(f"\t {Fore.RED}Value '{v}' not found in re-produced influence values for column '{saved_column}' and bin '{saved_bin}', but found in saved results.{Style.RESET_ALL}")
                        passed = False
                        seen_errors[(saved_column, saved_bin)].add(v)
                for v in re_values:
                    if v not in values and v not in seen_errors[(saved_column, saved_bin)]:
                        print(f"\t {Fore.RED}Value '{v}' not found in saved influence values for column '{re_column}' and bin '{re_bin}', but found in re-produced results.{Style.RESET_ALL}")
                        passed = False
                        seen_errors[(saved_column, saved_bin)].add(v)
                break
        if not found:
            print(f"\t {Fore.RED}Column '{saved_column}' and bin '{saved_bin}' not found in re-produced influence values, but found in saved results.{Style.RESET_ALL}")
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
                        print(f"\t {Fore.RED}Value '{v}' not found in saved influence values for column '{re_column}' and bin '{re_bin}', but found in re-produced results.{Style.RESET_ALL}")
                        passed = False
                        seen_errors[(re_column, re_bin)].add(v)
                for v in values:
                    if v not in re_values and v not in seen_errors[(re_column, re_bin)]:
                        print(f"\t {Fore.RED}Value '{v}' not found in re-produced influence values for column '{saved_column}' and bin '{saved_bin}', but found in saved results.{Style.RESET_ALL}")
                        passed = False
                        seen_errors[(re_column, re_bin)].add(v)
                break
        if not found:
            print(f"\t {Fore.RED}Column '{re_column}' and bin '{re_bin}' not found in saved influence values, but found in re-produced results.{Style.RESET_ALL}")
            passed = False

    if passed:
        print(f"\t {Fore.GREEN}Influence values comparison passed.{Style.RESET_ALL}")
        print(f"{Fore.GREEN} -------------------------------------------------- {Style.RESET_ALL}")
    else:
        print(f"\n \t {Fore.RED}Influence values comparison failed.{Style.RESET_ALL}")
        print(f"{Fore.RED} -------------------------------------------------- {Style.RESET_ALL}")

    return passed

def compare_significance_vals(saved_significance_vals: List[Dict], re_produced_significance_vals: List[Dict]) -> bool:
    """
    Compare the significance values in the saved results to the significance values in the re-produced results.\n
    Order within the lists does not matter.\n
    Prints out any discrepancies found.
    :param saved_significance_vals: The significance values saved in the results.
    :param re_produced_significance_vals: The significance values re-produced with the same query as the saved results.
    :return: True if the significance values are the same, False otherwise.
    """
    passed = True
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
                if saved_significance != re_significance and (saved_significance, re_significance) not in seen_errors[(saved_column, saved_bin)]:
                    print(f"\t {Fore.RED}Saved significance '{saved_significance}' does not match re-produced significance '{re_significance}' for column '{saved_column}' and bin '{saved_bin}'.{Style.RESET_ALL}")
                    passed = False
                    seen_errors[(saved_column, saved_bin)].add((saved_significance, re_significance))
                break
        if not found:
            print(f"\t {Fore.RED}Column '{saved_column}' and bin '{saved_bin}' not found in re-produced significance values, but found in saved results.{Style.RESET_ALL}")
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
                if re_significance != saved_significance and (saved_significance, re_significance) not in seen_errors[(re_column, re_bin)]:
                    print(f"\t {Fore.RED}Re-produced significance '{re_significance}' does not match saved significance '{saved_significance}' for column '{re_column}' and bin '{re_bin}'.{Style.RESET_ALL}")
                    passed = False
                    seen_errors[(re_column, re_bin)].add((saved_significance, re_significance))
                break
        if not found:
            print(f"\t {Fore.RED}Column '{re_column}' and bin '{re_bin}' not found in saved significance values, but found in re-produced results.{Style.RESET_ALL}")
            passed = False

    if passed:
        print(f"\t {Fore.GREEN}Significance values comparison passed.{Style.RESET_ALL}")
        print(f"{Fore.GREEN} -------------------------------------------------- {Style.RESET_ALL}")

    else:
        print(f"\n \t {Fore.RED}Significance values comparison failed.{Style.RESET_ALL}")
        print(f"{Fore.RED} -------------------------------------------------- {Style.RESET_ALL}")