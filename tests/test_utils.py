from typing import List, Dict

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
