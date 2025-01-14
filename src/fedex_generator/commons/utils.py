import inspect
import random
import numpy as np
import pandas as pd
import scipy
import statistics
import operator
from enum import Enum
from fedex_generator.commons.kstest import *


def to_valid_latex(string, is_bold: bool = False) -> str:
    """
    Convert a string to a valid latex string.
    :param string: The input string.
    :param is_bold: Whether the string should be bold.
    :return: The latex string.
    """
    latex_string = str(string)  # unicode_to_latex(string)
    # Choose the space character based on whether the string should be bold.
    space = r'\ ' if is_bold else ' '
    # Escape special characters and replace spaces with the chosen space character.
    final_str = latex_string.replace("&", "\\&").replace("#", "\\#").replace(' ', space).replace("_", space)
    return final_str


def to_valid_latex_with_escaped_dollar_char(string, is_bold: bool = False) -> str:
    """
    Convert a string to a valid latex string.
    This function adds the $ character to the list of characters that are escaped, while the function to_valid_latex
    does not escape the $ character. Other than that, the two functions are identical.
    :param string: The input string.
    :param is_bold: Whether the string should be bold.
    :return: The latex string.
    """
    return to_valid_latex(string, is_bold).replace("$", "\\$")


def smart_round(number) -> str:
    """
    Round a number to a set, certain number of decimal places.
    :param number: The input number.
    :return: The rounded number, as a string.
    """
    return f"{number:.1f}" if 0 <= number <= 2 else f"{number:.0f}"


def format_bin_item(item) -> str:
    """
    Formats a bin item to a string.
    :param item: The input item.
    :return: The formatted item.
    """
    # If the item is categorical or a date, return it as a string directly.
    if is_categorical(item) or is_date(item):
        return str(item)

    # If the item is an integer, return it as a string after converting it to int.
    if hasattr(item, "is_integer") and item.is_integer():
        return str(int(item))

    # If the item is a float between 1 and -1, return it as a string rounded to 2 decimal places.
    if -1 < item < 1:
        return f"{item:.2f}"

    # Else, return the item as a string rounded to 1 decimal place.
    return f"{item:.1f}"


def max_key(d) -> str:
    """
    Get the key with the maximum value from a dictionary.
    :param d: The input dictionary.
    :return: The key with the maximum value.
    """
    return max(d.items(), key=operator.itemgetter(1))[0]


def is_name_informative(name) -> bool:
    """
    Check if a name is informative, i.e. it includes at least one letter.
    :param name: The input name.
    :return: Whether the name is informative.
    """
    import string
    return any([char in string.ascii_letters for char in name])


def get_calling_params_name(item) -> str:
    """
    Get the name of the variable that called the function.
    :param item: The input item.
    :return: The name of the variable that called the function.
    """
    # Get the stack frames.
    frames = inspect.stack()
    highest_var = None
    # Iterate over the frames, starting from the third frame, looking for the variable that called the function
    # by comparing the id (memory address) of the variable with the id of the variables in the previous frames.
    for frame in frames[2:]:
        prev_locals = frame[0].f_locals
        for var_name in prev_locals:
            if id(item) == id(prev_locals[var_name]) and is_name_informative(var_name):
                highest_var = var_name

    return highest_var


# Define known numeric and categorical types.
NUMERIC_TYPES = ['float64', 'int32', 'int64', 'int', 'datetime64[ns]']
CATEGORICAL_TYPES = ['category', 'object', 'bool', 'tuple']


def drop_nan(array) -> np.ndarray:
    """
    Drop NaN values from an array.
    :param array: The input array.
    :return: The array with NaN values dropped, as a flattened numpy array.
    """
    return pd.Series(array).dropna().to_numpy().flatten()


class ArrayType(Enum):
    """
    Enum class to represent the type of an array.
    Either Categorical, Numeric, or Unknown.
    """
    Categorical = 1
    Numeric = 2
    Unknown = 3


def get_array_type(array_like) -> ArrayType:
    """
    Get the type of an array, by the ArrayType enum: Categorical, Numeric, or Unknown.
    :param array_like: The input array.
    :return: The type of the array.
    """
    # If the array is of a known categorical type, return Categorical.
    if np.array(array_like).dtype.name in CATEGORICAL_TYPES or np.array(array_like).dtype.name.startswith('str'):
        return ArrayType.Categorical

    # If the array is of a known numeric type, return Numeric.
    if np.array(array_like).dtype.name in NUMERIC_TYPES:
        return ArrayType.Numeric

    # Else, return Unknown.
    return ArrayType.Unknown


def is_categorical(array_like) -> bool:
    """
    Check if an array is categorical.
    :param array_like: The input array.
    :return: Whether the array is categorical.
    """
    array_type = get_array_type(array_like)
    if array_type == ArrayType.Categorical:
        return True

    if array_type == ArrayType.Numeric:
        return False

    # Throw an error if the array type is unknown.
    raise RuntimeError(f"Bad type: {np.array(array_like).dtype.name}")


def is_numeric(array_like) -> bool:
    """
    Check if an array is numeric.
    :param array_like: The input array.
    :return: Whether the array is numeric.
    """
    return not is_categorical(array_like)


def is_date(item) -> bool:
    """
    Check if an item is a date.
    :param item: The input item.
    :return: Whether the item is a date
    """
    if "date" in str(type(item)):
        return True

    return hasattr(item, 'date') or hasattr(item, 'time')
