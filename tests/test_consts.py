import result_comparison_functions as rcf

test_fail_explanations = {
    "Correlated attributes comparison test": "The list of correlated attributes in the re-produced results does not match the list of correlated attributes in the saved results. There is likely an error in the method get_correlated_attributes of the Filter operation.",
    "Measure scores comparison test": "The measure scores in the re-produced results do not match the measure scores in the saved results. There is likely an error in the method calc_measure_internal of the tested measure.",
    "Score dicts comparison test": "The score dictionaries in the re-produced results do not match the score dictionaries in the saved results. There is likely an error in the method calc_measure_internal of the tested measure, or in calc_measure in BaseMeasure.",
    "Influence values comparison test": "The influence values in the re-produced results do not match the influence values in the saved results. If the measure scores and score dictionaries are correct, there is likely an error in the method calc_influence_col in the tested measure.",
    "Significance values comparison test": "The significance values in the re-produced results do not match the significance values in the saved results. If the influence values test passed, then there is likely an error in the method get_significance in BaseMeasure.",
    "Results dataframe comparison test": "The results dataframes in the re-produced results do not match the results dataframes in the saved results. If all other tests have passed, then there is likely an error in the build_explanation method of the tested measure",
    "Column names comparison test": "The column names in the re-produced results do not match the column names in the saved results. There is likely an error in the method _get_column_names of the GroupBy operation.",
    "One to many attributes comparison test": "The one to many attributes in the re-produced results do not match the one to many attributes in the saved results. There is likely an error in the method get_one_to_many_attributes of the GroupBy operation.",
}

test_funcs = {
    "Correlated attributes comparison test": {
        "func": rcf.compare_correlated_attributes,
        "attribute_name": "correlated_attributes",
        "require_duplicate_fix": False
    },
    "Column names comparison test": {
        "func": rcf.compare_column_names,
        "attribute_name": "column_names",
        "require_duplicate_fix": False
    },
    "One to many attributes comparison test": {
        "func": rcf.compare_one_to_many_attributes,
        "attribute_name": "one_to_many_attributes",
        "require_duplicate_fix": False
    },
    "Measure scores comparison test": {
        "func": rcf.compare_measure_scores,
        "attribute_name": "measure_scores",
        "require_duplicate_fix": False
    },
    "Score dicts comparison test": {
        "func": rcf.compare_score_dicts,
        "attribute_name": "score_dict",
        "require_duplicate_fix": False
    },
    "Influence values comparison test": {
        "func": rcf.compare_influence_vals,
        "attribute_name": "influence_vals",
        "require_duplicate_fix": True
    },
    "Significance values comparison test": {
        "func": rcf.compare_significance_vals,
        "attribute_name": "significance_vals",
        "require_duplicate_fix": True
    },
    "Results dataframe comparison test": {
        "func": rcf.compare_results,
        "attribute_name": "saved_results",
        "require_duplicate_fix": False
    },
}
