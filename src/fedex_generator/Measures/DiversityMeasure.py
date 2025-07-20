
import numpy as np
import pandas as pd
from numpy import number
from pandas import Series
from scipy import stats
from typing import Callable

from fedex_generator.Measures.BaseMeasure import BaseMeasure, START_BOLD, END_BOLD
from fedex_generator.Measures.Bins import Bin, MultiIndexBin
from fedex_generator.commons import utils

# A constant dictionary that maps operation names to their corresponding functions, for convenience and clarity
OP_TO_FUNC = {
    'count': np.sum,
    'sum': np.sum,
    'max': np.max,
    'min': np.min,
    'mean': np.mean,
    'prod': np.prod,
    'sem': stats.sem,
    'var': np.var,
    'std': np.std,
    'median': np.median,
    'first': lambda bin_values: bin_values[0],
    'last': lambda bin_values: bin_values[-1],
    'size': np.sum,
    'nunique': lambda bin_values: len(set(bin_values))
}


def draw_bar(x: list, y: list, avg_line=None, items_to_bold=None, head_values=None, xname=None, yname=None, alpha=1.,
             ax=None):
    """
    Draw a bar chart with optional features.

    :param x: List of x-axis values.
    :param y: List of y-axis values.
    :param avg_line: Optional; a value to draw a horizontal line representing the average.
    :param items_to_bold: Optional; list of items to highlight in the bar chart.
    :param head_values: Optional; list of values to display above each bar.
    :param xname: Optional; label for the x-axis.
    :param yname: Optional; label for the y-axis.
    :param alpha: Optional; transparency level of the bars (default is 1.0).
    :param ax: Optional; matplotlib axes object to draw the bar chart on.
    """

    width = 0.5
    ind = np.arange(len(x))

    # Convert the x and y values to valid LaTeX strings, if needed
    x = x if utils.is_numeric(x) else [utils.to_valid_latex(i) for i in x]
    y = y if utils.is_numeric(y) else [utils.to_valid_latex(i) for i in y]

    # Convert the items to bold to valid LaTeX strings, if needed
    if items_to_bold is not None:
        items_to_bold = items_to_bold if utils.is_numeric(items_to_bold) else [utils.to_valid_latex(i) for i in
                                                                               items_to_bold]

    # Create the bar chart
    bar = ax.bar(ind, y, width, alpha=alpha)
    ax.set_xticks(ind)
    ax.set_xticklabels(tuple([str(i) for i in x]), rotation='vertical')
    ax.set_ylim(min(y) - min(y) * 0.01, max(y) + max(y) * 0.001)

    # Add the average line, if provided
    if avg_line is not None:
        ax.axhline(avg_line, color='red', linewidth=1)

    # Highlight the items to bold, if provided
    if items_to_bold is not None:
        for item in items_to_bold:
            bar[x.index(item)].set_color('green')

    # Add the head values, if provided
    if head_values is not None:
        for i, col in enumerate(bar):
            yval = col.get_height()
            ax.text(col.get_x(), yval + .05, head_values[i])

    # Add the x and y axis labels, if provided
    if xname is not None:
        ax.set_xlabel(utils.to_valid_latex_with_escaped_dollar_char(xname), fontsize=24)

    if yname is not None:
        ax.set_ylabel(utils.to_valid_latex_with_escaped_dollar_char(yname), fontsize=24)


def flatten_other_indexes(series, main_index):
    """
    Flatten all indexes of a pandas Series except the specified main index.

    This function resets the specified main index of the Series, flattens the remaining indexes into a single index,
    and then sets the main index back as an additional level of the index.

    :param series: The input pandas Series with a MultiIndex.
    :param main_index: The name of the main index to retain.
    :return: A pandas Series with the main index retained and other indexes flattened.
    """
    df = pd.DataFrame(series)
    df = df.reset_index(main_index)
    index_name = "_".join([ind for ind in df.index.names])
    df.index = df.index.to_flat_index()
    df.index.names = [index_name]
    df = df.set_index(main_index, append=True)
    return df[df.columns[0]]


class DiversityMeasure(BaseMeasure):
    """
    A class implementing the diversity measure, a measure for interestingness of GroupBy operations.\n
    This measure is described in the article "FEDEX: An Explainability Framework for Data Exploration Steps" by
    Daniel Deutch, Amir Gilad, Tova Milo, Amit Mualem, Amit Somech.\n
    From the article: "intuitively, a group-by step that yields a dataframe with a highly diverse set of aggregated values,
    implies a large difference between the groups".\n
    The measure is defined as the coefficient of variation: \n
    .. math:: I_A (d_{in}, q_g, d_{out}) = CV(d_{out}[A]) = \\frac{1}{\\tilde{a}} * \\sqrt{\\frac{\\sum_i (a_i - \\tilde{a})^2}{n}}\n
    Where:\n
    - :math:`I_A` is the diversity measure for attribute A.\n
    - :math:`d_{in}` is the input dataframe.\n
    - :math:`q_g` is the query, usually a group-by operation.\n
    - :math:`d_{out}` is the output dataframe.\n
    - :math:`A` is the attribute of interest.\n
    - :math:`CV` is the coefficient of variation.\n
    - :math:`\\tilde{a}` is the mean of the values of attribute A.\n
    - :math:`a_i` is the i-th value of attribute A in the output dataframe.\n
     """

    MAX_BARS = 25

    def __init__(self):
        super().__init__()

    def get_agg_func_from_name(self, name: str) -> str | Callable:
        """
        Retrieve the aggregation function corresponding to the given name.

        This function attempts to find the aggregation function based on the provided name.
        It first checks if the name corresponds to a method of `pd.Series`. If not, it checks
        if the name is in the `OP_TO_FUNC` dictionary. If the function is still not found,
        it searches within the `agg_dict` of the `operation_object`.

        :param name: The name of the aggregation function to retrieve.
        :return: The aggregation function corresponding to the given name.
        """

        # Get the operation name from the column name.
        # Name can be an int or a float in some cases, so we need to convert it to a string.
        if isinstance(name, int) or isinstance(name, float):
            name = str(name)
        # It can also be a string, or in the case of a multi-index, a tuple.
        if isinstance(name, str):
            operation = name.split("_")[-1].lower()
        elif isinstance(name, tuple):
            name = "_".join(x for x in name)
            operation = name.lower()
        else:
            raise TypeError(
                f"The type of the column name is {type(name)}, which we the developers did not expect and do not know how to handle. If you are a developer, please fix this. If you are a user, please report this. Thank you.")

        # Check if the name corresponds to a method of `pd.Series` or is in the `OP_TO_FUNC` dictionary
        # If the operation is a method of `pd.Series`, we can quickly return the function. Unless the operation name coincides with a column name,
        # in which case we can't be sure if it's a method or a column name, so we need to check the agg_dict.
        # As an example: the spotify dataset has a column named "mode", which is in-fact the reason why this condition
        # was discovered as necessary, because reaching here with "mode" as the operation name would cause an aggregation failure
        # exception in the calling function once it tries to use the function returned here.
        if hasattr(pd.Series, operation) and not operation in self.operation_object.result_df.columns:
            # In cases like 'size', this can end up returning a property, which will cause an exception when it's used
            # in the calling function. So we need to check if it's callable. Otherwise we'll find it later in the
            # agg_dict.
            attr = getattr(pd.Series, operation)
            if callable(attr):
                return attr
        elif operation in OP_TO_FUNC:
            return OP_TO_FUNC[operation]

        res = []
        # Search within the `agg_dict` of the `operation_object`.
        for x in self.operation_object.agg_dict:
            op = self.operation_object.agg_dict[x]
            if not isinstance(op, str):
                for y in op:
                    res.append(x + '_' + (y if isinstance(y, str) else y.__name__))
            else:
                res.append(x + '_' + op)
        # Note that this can fail if the agg_dict it not set properly, or if the name is not in the agg_dict.
        # For example, if the column is "All", meaning all columns are aggregated, then the name will not be in the
        # agg_dict, and this will fail.
        try:
            aggregation_index = res.index(name)
            return list(self.operation_object.agg_dict.values())[0][aggregation_index]
        except (ValueError, IndexError) as e:
            if any([x.startswith("All_") for x in res]):
                return list(self.operation_object.agg_dict.values())[0][0]
            # It is also possible that we get names that don't include the aggregation function, but are still
            # valid, like "col_name", and res will have "col_name_mean" in it.
            # This won't be caught anywhere above, so we check for it here.
            contains_name = [x for x in res if x.startswith(name)]
            if len(contains_name) > 0:
                agg_funcs = [x.split("_")[-1] for x in contains_name]
                # We return a random function from the list of aggregation functions, since we don't know which one to use.
                return np.random.choice(agg_funcs)

            raise e
        # aggregation_index = res.index(name)
        # return list(self.operation_object.agg_dict.values())[0][aggregation_index]

    def draw_bar(self, bin_item: MultiIndexBin, influence_vals: dict = None, title=None, ax=None, score=None,
                 show_scores: bool = False, explanation_num: int | None = None):
        """
        Draw a bar chart for a given bin item with optional features.

        This function creates a bar chart for the provided `bin_item`, highlighting the most influential values
        and optionally displaying additional information such as the average line, title, and scores.

        :param bin_item: The `MultiIndexBin` object containing the bin data to be plotted.
        :param influence_vals: Optional; a dictionary of influence values for the bin items.
        :param title: Optional; the title of the bar chart.
        :param ax: Optional; the matplotlib axes object to draw the bar chart on.
        :param score: Optional; the score to be displayed in the title if `show_scores` is True.
        :param show_scores: Optional; a boolean indicating whether to display the score in the title (default is False).
        :param explanation_num: Optional; an integer representing the explanation number to be included in the title.
        """
        try:
            # Get the index of the maximum value and its influence
            max_values, max_influence = self.get_max_k(influence_vals, 1)
            max_value = max_values[0]

            # Calculate the average value of the result column
            res_col = bin_item.get_binned_result_column()
            average = float(utils.smart_round(res_col.mean()))

            # Get the aggregation function corresponding to the result column name
            agger_function = self.get_agg_func_from_name(res_col.name)
            aggregated_result = res_col.groupby(bin_item.get_bin_name()).agg(agger_function)

            # Flatten the other indexes of the result column
            bin_result = bin_item.result_column.copy()
            bin_result = flatten_other_indexes(bin_result, bin_item.get_bin_name())
            smallest_multi_bin = MultiIndexBin(bin_item.source_column, bin_result, 0)
            # Calculate the influence values for the bin items
            influences = self.get_influence_col(res_col, smallest_multi_bin, True)
            rc = res_col.reset_index([bin_item.get_bin_name()])

            # Get the relevant items and influences for the maximum value
            relevant_items = rc[rc[bin_item.get_bin_name()] == max_value]
            relevant_influences = dict([(k, influences[k]) for k in relevant_items.index])
            # Get the top 10 most influential values
            max_values, max_influence = self.get_max_k(relevant_influences, 10)
            max_values = sorted(max_values)
            labels = set(aggregated_result.keys())

            # Limit the number of bars to 25, and sort the labels
            if len(labels) > self.MAX_BARS:
                top_items, _ = self.get_max_k(influence_vals, self.MAX_BARS)
                labels = sorted(top_items)
            else:
                labels = sorted(labels)

            # Get the aggregated values for the labels, and draw the bar chart
            aggregate_column = [aggregated_result.get(item, 0) for item in labels]
            if explanation_num is not None:
                title = f"{START_BOLD}[{explanation_num}]{END_BOLD} {title}" if title else f"{START_BOLD}[{explanation_num}]{END_BOLD}"
            if show_scores:
                ax.set_title(f'score: {score}\n{utils.to_valid_latex(title)}', fontdict={'fontsize': 10})
            else:
                ax.set_title(utils.to_valid_latex(title), fontdict={'fontsize': 20})

            draw_bar(labels, aggregate_column, aggregated_result.mean(), [max_value],
                     xname=f'{bin_item.get_bin_name()} values', yname=bin_item.get_value_name(),
                     ax=ax)
            ax.set_axis_on()

            # Rotate the x-axis labels for better readability, if there are over 8 labels
            if len(labels) > 8:
                ax.tick_params(axis='x', rotation=45, tickdir='in')

        except Exception as e:
            # In the case of an exception, draw a bar chart using the draw_bar method defined outside the class
            columns = bin_item.get_binned_result_column()
            max_group_value = self._find_max_group_value(columns, max_value)

            # If there are more than 25 bars, limit the number of bars to 25
            if len(columns) > self.MAX_BARS:
                columns = self._select_top_columns(influence_vals, columns, max_value, max_group_value)

            title = self._fix_explanation(title, columns, max_value, max_group_value)
            if explanation_num is not None:
                title = f"{START_BOLD}[{explanation_num}]{END_BOLD} {title}" if title else f"{START_BOLD}[{explanation_num}]{END_BOLD}"
            ax.set_title(utils.to_valid_latex(title), fontdict={'fontsize': 20})

            draw_bar(list(columns.index),
                     list(columns),
                     average,
                     [max_group_value] if not isinstance(max_group_value, list) else max_group_value,
                     yname=bin_item.get_bin_name(),
                     ax=ax,
                     )

            # Rotate the x-axis labels for better readability, if there are over 8 labels
            if len(columns) > 8:
                ax.tick_params(axis='x', rotation=45, tickdir='in')
            ax.set_axis_on()


    def _find_max_group_value(self, columns, max_value) -> str | list:
        """
        Finds the group value corresponding to the max value in the columns.
        :param columns: The columns to search for the max value.
        :param max_value: The max value to search for in the columns.
        :return: The group value corresponding to the max value.
        """
        # Using a simple columns == max_value works, so long as the index is not a multi-index one.
        # Otherwise, the max_value may end up being from one of the index levels, which will cause an exception as
        # it is not in the columns.
        if isinstance(columns.index, pd.MultiIndex) and not isinstance(max_value, tuple):
            # Search the index for any match with the max value, even if it's from a subset of the index
            # For example, if we have a MultiIndex with levels A,B,C, and the max value is from B, we can still
            # find it in the index.
            index = columns.index.to_frame()
            # Get all locations in the index where the max_value is found
            indexes = set()
            for col in index.columns:
                col_indexes = index[index[col] == max_value].index
                if len(col_indexes) > 0:
                    indexes.update(col_indexes)
            # Get the columns corresponding to the max value
            indexes = list(indexes)
            max_group_value = list(columns.loc[indexes].to_dict().keys())
            # We set this to also be equal, since this value will get accessed again in _fix_explanation, and it
            # should match the way we fixed it here.
            max_value = max_group_value
        else:
            max_group_value = list(columns[columns == max_value].to_dict().keys())[0]

        return max_group_value


    def _select_top_columns(self, influence_vals, columns, max_value, max_group_value) -> pd.Series:
        """
        Select the top columns to display in the bar chart.
        :param influence_vals: The influence values of the columns.
        :param columns: The columns to select the top items from.
        :param max_value: The max value found.
        :param max_group_value: The name of the group corresponding to the max value.
        :return: The top columns to display in the bar chart.
        """
        top_items, _ = self.get_max_k(influence_vals, self.MAX_BARS)
        # Just like before, we need different handling for multi-index cases, in the case where there are
        # multiple top items, but they are not tuples, i.e. they are not keys for the multi-index
        if isinstance(columns.index, pd.MultiIndex) and not all([isinstance(i, tuple) for i in top_items]):
            index = columns.index.to_frame()
            # We use a list this time, because we need to maintain the same order as the top items
            indexes = []
            for item in top_items:
                for col in index.columns:
                    col_indexes = index[index[col] == item].index
                    if len(col_indexes) > 0:
                        indexes.extend(col_indexes)
            if len(indexes) > len(columns):
                indexes = indexes[:len(columns)]
        else:
            indexes = []
            # Get the indexes of the top items, in order of the top items
            for item in top_items:
                indexes.extend(columns[columns == item].index)
        columns_backup = columns.copy()
        # Using the indexes in this way will automatically sort the columns such that their order matches that
        # of the top items
        columns = columns.loc[indexes]
        columns = columns[:self.MAX_BARS]
        cols_dtype_is_numeric = utils.is_numeric(columns)

        # If the max value is not in the columns, add it to the columns
        if not isinstance(max_group_value, list):
            if max_group_value not in columns:
                columns = columns.append(pd.Series(max_value, index=[max_group_value]))
        else:
            # Not having the index sorted can raise a performance warning, so we sort it if it's not sorted
            if not columns.index._is_lexsorted():
                columns = columns.sort_index()
            for value in max_group_value:
                if value not in columns:
                    # If the max value is the same dtype as the columns, we can append it directly
                    if cols_dtype_is_numeric and type(value) in (int, float):
                        columns = columns._append(pd.Series(max_value, index=[value]))
                    # Else, retrieve the value from the columns_backup, which is guaranteed to have the value
                    else:
                        value_to_add = columns_backup.loc[value]
                        columns = columns._append(pd.Series(value_to_add, index=[value]))

        return columns

    @staticmethod
    def _fix_explanation(explanation: str, binned_column: Series, max_value, max_group_value) -> str:
        """
        Change explanation column to match group by
        :param explanation:  bin explanation
        :param binned_column: bin column
        :param max_value: max value

        :return: new explanation
        """
        # Tuple is used to represent multi-index columns
        # if isinstance(max_value, tuple):
        #     max_group_value = list(binned_column.loc[max_value].to_dict().keys())[0]
        # else:
        #     max_group_value = list(binned_column[binned_column == max_value].to_dict().keys())[0]
        binned_column_name = str(binned_column.name)
        max_value_name = binned_column_name.replace('_', '\\ ')
        try:
            max_group_value.replace('$', '\\$')
            max_value_name.replace('$', '\\$')
        except:
            pass
        group_by_name = binned_column.to_frame().axes[0].name

        return explanation.replace(f'\'{max_value_name}\'=\'{max_value}\'',
                                   f'\'{group_by_name}\' = \'{max_group_value}\'')


    def _try_fix_explanation(self, explanation: str, binned_column: MultiIndexBin, max_value) -> str:
        columns = binned_column.get_binned_result_column()
        max_group_value = self._find_max_group_value(columns, max_value)
        return self._fix_explanation(explanation, columns, max_value, max_group_value)



    def interestingness_only_explanation(self, source_col: Series, result_col: Series, col_name: str) -> str:
        return f"After employing the GroupBy operation we can see highly diverse set of values in the column '{col_name}'\n" \
               f"The variance" + \
            (f" was {self.calc_var(source_col)} and now it " if source_col is not None else "") + \
            f" is {self.calc_var(result_col)}"

    def calc_influence_col(self, current_bin: Bin):
        # Get the values of the current bin
        bin_values = current_bin.get_bin_values()
        source_col = current_bin.get_source_by_values(bin_values)
        res_col = current_bin.get_result_by_values(bin_values)
        # Calculate the diversity score for the current bin
        score_all = self.calc_diversity(source_col, res_col)

        influence = []
        # Compute the diversity score for each value in the bin, then calculate the influence of each value
        # as the difference between the score of all values and the score without the current value
        for value in bin_values:
            source_col_only_list = current_bin.get_source_by_values([b for b in bin_values if b != value])
            res_col_only_list = current_bin.get_result_by_values([b for b in bin_values if b != value])

            score_without_bin = self.calc_diversity(source_col_only_list, res_col_only_list)
            influence.append(score_all - score_without_bin)

        return influence

    def calc_var(self, pd_array) -> number:
        """
        Calculate the variance of a pandas array.

        This method computes the variance of the given pandas array. If the array contains numeric values,
        it returns the variance of those values. If the array contains non-numeric values, it calculates
        the variance based on the frequency of each unique value in the array.

        :param pd_array: The pandas array for which to calculate the variance.
        :return: The variance of the array.
        """
        if utils.is_numeric(pd_array):
            return np.var(pd_array)

        appearances = (pd_array.value_counts()).to_numpy()
        mean = np.mean(appearances)
        return np.sum(np.power(appearances - mean, 2)) / len(appearances)

    def calc_diversity(self, source_col: Series, res_col: Series) -> float:
        """
        Calculate the diversity measure for the given source and result columns.

        This method computes the diversity measure, defined as the coefficient of variation,
        for the provided source and result columns. The diversity measure is calculated as
        the ratio of the variance of the result column to the variance of the source column - see the class description
        for the formula.

        :param source_col: The source column as a pandas Series. This can be None.
        :param res_col: The result column as a pandas Series.
        :return: The diversity measure as a float.
        """
        var_res = self.calc_var(res_col)

        if source_col is not None:
            var_rel = self.calc_var(source_col)
        else:
            var_rel = 1.

        res = (var_res / var_rel) if var_rel != 0 else (0. if var_res == 0 else var_res)
        return 0 if np.isnan(res) else res

    def calc_measure_internal(self, bin: Bin):
        result_column = bin.get_binned_result_column().dropna()
        if bin.name == "MultiIndexBin":
            operation = self.get_agg_func_from_name(result_column.name)
            result_column = result_column.groupby(bin.get_bin_name()).agg(operation)
        return self.calc_diversity(None if bin.source_column is None else bin.source_column.dropna(),
                                   result_column)

    def build_operation_expression(self, source_name):
        from fedex_generator.Operations.GroupBy import GroupBy
        if isinstance(self.operation_object, GroupBy):
            return f'{source_name}.groupby({self.operation_object.group_attributes})' \
                   f'.agg({self.operation_object.agg_dict})'

    def build_explanation(self, current_bin: Bin, max_col_name, max_value, source_name):
        res_col = current_bin.get_binned_result_column()
        # If the result column is categorical, return an empty explanation
        if utils.is_categorical(res_col):
            return ""

        var = self.calc_var(res_col)
        # If the current bin is numeric or categorical, set the max value to the provided value
        if current_bin.name == "NumericBin" or current_bin.name == 'CategoricalBin':
            max_value_numeric = max_value

        # Otherwise, some additional processing is required
        elif current_bin.name == "MultiIndexBin":
            # Get the result column values for the maximum value, then calculate the aggregation function
            # on those values and set the max value to the result of the aggregation function
            bin_values = current_bin.get_result_by_values([max_value])
            operation = self.get_agg_func_from_name(res_col.name)
            # If the operation is a callable function, call it with the bin values
            if callable(operation):
                max_value_numeric = operation(bin_values)
            # Otherwise, it is a string, so we need to get the function from the OP_TO_FUNC dictionary
            else:
                operation = OP_TO_FUNC[operation]
                max_value_numeric = operation(bin_values)

            # Get the name of the column with the maximum value, and compute the variance of the result column
            max_col_name = current_bin.get_bin_name()
            res_col = res_col.groupby(max_col_name).agg(operation)
            var = self.calc_var(res_col)

        elif current_bin.name == "NoBin":
            result_column = current_bin.get_binned_result_column()
            max_value_numeric = max_value
            max_value = result_column.index[result_column == max_value].tolist()
            max_col_name = result_column.index.name

        elif type(current_bin) == Bin:
            raise Exception("Bin is not supported")
        else:
            raise Exception(f"unknown bin type {current_bin.name}")

        # Get the standard deviation of the result column and calculate the z-score of the maximum value,
        # determining whether the value is relatively high or low, and generate the explanation
        sqr = np.sqrt(var)
        x = ((max_value_numeric - np.mean(res_col)) / sqr) if sqr != 0 else 0
        group_by_text = utils.to_valid_latex_with_escaped_dollar_char(f"'{max_col_name}'='{max_value}'", True)
        proportion = 'low' if x < 0 else 'high'
        proportion_column = utils.to_valid_latex_with_escaped_dollar_char(f"{proportion} '{res_col.name}'", True)

        expl = f"Groups with {START_BOLD}{group_by_text}{END_BOLD}(in green)\n" \
               f"have a relatively {START_BOLD}{proportion_column}{END_BOLD} value:\n" \
               f"{utils.smart_round(np.abs(x))} standard deviations {proportion}er than the mean\n" \
               f"({utils.smart_round(np.mean(res_col))})"

        return expl
