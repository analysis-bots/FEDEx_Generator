import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from paretoset import paretoset

from fedex_generator.Measures.Bins import Bins, Bin
from fedex_generator.commons.consts import SIGNIFICANCE_THRESHOLD, TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW
from fedex_generator.commons.utils import is_numeric, to_valid_latex

try:
    from FEDEx_Generator.src.fedex_generator.Operations.Operation import Operation
    from FEDEx_Generator.src.fedex_generator.commons.DatasetRelation import DatasetRelation
except:
    from fedex_generator.Operations.Operation import Operation
    from fedex_generator.commons.DatasetRelation import DatasetRelation

# Sets some parameters for the plots. Usage of latex is optional, and is currently set to False by default.
usetex = False  # matplotlib.checkdep_usetex(True)
print(f"usetex-{usetex}")
rc('text', usetex=usetex)
matplotlib.rcParams.update({'font.size': 16})

# Bold text in latex, for the plots
START_BOLD = r'$\bf{'
END_BOLD = '}$'


def draw_pie(items_dict: dict, important_item=None) -> None:
    """
    Draw a pie chart of the items_dict. The important_item will be highlighted.
    :param items_dict: dictionary of items and their probabilities
    :param important_item: the item to highlight
    """
    labels = items_dict.keys()
    probabilities = [items_dict[item] for item in labels]
    explode = [0.1 if item == important_item else 0 for item in labels]

    fig1, ax1 = plt.subplots()
    ax1.pie(probabilities, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def draw_histogram(items_before: list, items_after: list, label: str, title) -> None:
    """
    Draw a histogram of the items_before and items_after.
    :param items_before: list of items before the operation
    :param items_after: list of items after the operation
    :param label: the label of the histogram
    :param title: the title of the histogram
    """
    if items_before is None:
        items_before = []
    all_vals = list(set(list(items_before) + list(items_after)))
    # If the values are not numeric, we can't use bins, and we just plot the values.
    if not is_numeric(all_vals):
        if len(items_before) > 0:
            plt.hist([sorted(list(items_before)), sorted(list(items_after))], label=["Before", "After"],
                     color=[u'#ff7f0e', u'#1f77b4'])
        else:
            plt.hist([sorted(list(items_after))], label=["After"])

    # If the values are numeric, we use bins.
    else:
        bins = np.linspace(np.min(all_vals), np.max(all_vals), min(40, len(all_vals)))
        if len(items_before) > 0:
            plt.hist([list(items_before), list(items_after)], bins=bins, label=["Before", "After"],
                     color=[u'#ff7f0e', u'#1f77b4'])
        else:
            plt.hist([list(items_after)], bins=bins, label=["After"])

    plt.ylabel('results occurrences count', fontsize=16)
    plt.xlabel(to_valid_latex(label), fontsize=16)
    plt.legend(loc='upper right')
    plt.title(to_valid_latex(title))
    plt.show()


class BaseMeasure(object):
    """
    Base class for all measures. Contains the basic methods for calculating the measure and building the explanation.
    Inherit from this class to create a new measure, and implement the abstract methods.\n
    """

    def __init__(self):
        self.source_dict, self.max_val, self.score_dict, self.operation_object, self.scheme = \
            None, None, None, None, None
        self.bins = {}
        self.bin_func = pd.qcut

    def interestingness_only_explanation(self, source_col: pd.Series, result_col: pd.Series, col_name: str) -> str:
        """
        Provide an explanation, based only on the interestingness score.\n
        Abstract method, should be implemented by the inheriting class.
        :param source_col: The column from the source DataFrame
        :param result_col: The column from the result DataFrame
        :param col_name: The name of the column
        :return: The explanation string
        """
        raise NotImplementedError()

    @staticmethod
    def get_source_and_res_cols(dataset_relation: DatasetRelation, attr: str) -> Tuple[pd.Series, pd.Series]:
        """
        Static method to get source and result columns from dataset_relation.
        :param dataset_relation: dataset relation object
        :param attr: attribute name
        :return source_col, res_col: source and result columns with the attribute name, without null values
        """
        source_col, res_col = dataset_relation.get_source(attr), dataset_relation.get_result(attr)
        res_col = res_col[~res_col.isnull()]
        if source_col is not None:
            source_col = source_col[~source_col.isnull()]

        return source_col, res_col


    def _calc_measure(self, attr: str, dataset_relation: DatasetRelation, operation_object: Operation, scheme: dict,
                      use_only_columns: list, ignore=None,
                      unsampled_source_df: pd.DataFrame = None, unsampled_res_df: pd.DataFrame = None,
                      column_mapping: dict=None) \
            -> Tuple[str, float, str, Bins, Tuple[pd.Series, pd.Series]] | Tuple[str, float, None, None, None]:
        """

        :param attr: The attribute name.
        :param dataset_relation: The dataset relation for the current attribute.
        :param operation_object: The operation object for the current operation.
        :param scheme: The scheme of the columns.
        :param use_only_columns: A list of columns to include in the explanation. If empty, all columns will be included.
        :param ignore: A list of columns to ignore in the explanation. Set to [] by default.
        :param unsampled_source_df: The source DataFrame before sampling. Optional. Only needed if sampling is used.
        :param unsampled_res_df: The result DataFrame before sampling. Optional. Only needed if sampling is used.
        :param column_mapping: A dict mapping the original column names to the current column names. Optional. Needed in case some columns were renamed as part of a groupby operation.
        :return:
        """
        # If the attribute is in the ignore list, skip it.
        if attr in ignore:
            return attr, -1, None, None, None

        # Get the column scheme from the scheme dictionary. If not found, set it to 'ni'.
        column_scheme = scheme.get(attr, "ni").lower()
        # If the column scheme is 'i', skip the attribute.
        if column_scheme == "i":
            return attr, -1, None, None, None

        # If there are columns in the use_only_columns list, and the attribute is not in the list, skip it.
        if len(use_only_columns) > 0 and attr not in use_only_columns:
            return attr, -1, None, None, None

        # Get the source and result columns for the attribute.
        source_col, res_col = self.get_source_and_res_cols(dataset_relation, attr)
        # If the result column is empty, skip the attribute.
        if len(res_col) == 0:
            return attr, -1, None, None, None


        # Create bin candidates from the source and result columns, with a bin count specified by the operation object.
        size = operation_object.get_bins_count()

        bin_candidates = Bins(source_col, res_col, size)
        unsampled_bin_candidates = None

        if unsampled_source_df is not None and unsampled_res_df is not None:
            # source_col can be none in some groupby cases. We don't want to modify it if it is, as that may cause
            # unexpected behavior.
            if source_col is not None:
                # In the case of groupby that creates tuple columns, we need to handle the case where the attribute in the
                # result dataframe is a tuple that did not exist in the source dataframe.
                if attr not in unsampled_source_df.columns and isinstance(attr, tuple):
                    source_col = unsampled_source_df[attr[0]]
                else:
                    source_col = unsampled_source_df[attr] if attr not in column_mapping else unsampled_source_df[column_mapping[attr]]
            res_col = unsampled_res_df[attr]
            unsampled_bin_candidates = Bins(source_col, res_col, size)

        # Compute the measure score for each bin candidate, and get the maximum score.
        measure_score = -np.inf
        for bin_ in bin_candidates.bins:
            measure_score = max(self.calc_measure_internal(bin_), measure_score)


        return (attr, measure_score, dataset_relation.get_source_name(),
                unsampled_bin_candidates if unsampled_bin_candidates is not None else bin_candidates,
                (source_col, res_col))

    def calc_measure(self, operation_object: Operation, scheme: dict, use_only_columns: list, ignore=None,
                     unsampled_source_df: pd.DataFrame = None, unsampled_res_df: pd.DataFrame = None,
                     column_mapping: dict=None, debug_mode: bool = False
                     ) -> \
            Dict[str, float]:
        """
        Calculate the measure for each attribute in the operation_object.

        :param operation_object: The operation object - the operation that was performed, one of the classes in the Operations package.
        :param scheme: The scheme of the columns, in the form of a dictionary, where the key is the column name and the value is the scheme of the column. Columns with the scheme 'i' will be ignored.
        :param use_only_columns: A list of columns to include in the explanation. If empty, all columns will be included.
        :param ignore: A list of columns to ignore in the explanation. Set to [] by default.
        :param unsampled_source_df: The unsampled source DataFrame. Optional. Only needed if sampling is used.
        :param unsampled_res_df: The unsampled result DataFrame. Optional. Only needed if sampling is used.
        :param column_mapping: A dict mapping the original column names to the current column names. Optional. Needed in case some columns were renamed as part of a groupby operation.
        :param debug_mode: Developer option. Disables multiprocessing for easier debugging. Default is False.
        """
        # Set the operation object, the scheme, and the score dictionary to the given values.
        # Also initialize the max_val to -1.
        if ignore is None:
            ignore = []
        if column_mapping is None:
            column_mapping = {}

        self.operation_object = operation_object
        self.score_dict = {}
        self.max_val = -1
        self.scheme = scheme

        # Iterate over the attributes in the operation object.
        # From limited testing, doing this in parallel gives a small performance boost.
        if not debug_mode:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._calc_measure, attr, dataset_relation, operation_object, scheme, use_only_columns,
                                    ignore, unsampled_source_df, unsampled_res_df, column_mapping) for attr, dataset_relation in operation_object.iterate_attributes()
                ]
                for future in as_completed(futures):
                    attr, measure_score, source_name, bins, cols = future.result()
                    if measure_score != -1:
                        self.score_dict[attr] = (source_name, bins, measure_score, cols)
        else:
            for attr, dataset_relation in operation_object.iterate_attributes():
                attr, measure_score, source_name, bins, cols = self._calc_measure(attr, dataset_relation, operation_object, scheme, use_only_columns, ignore, unsampled_source_df, unsampled_res_df, column_mapping)
                if measure_score != -1:
                    self.score_dict[attr] = (source_name, bins, measure_score, cols)

        # Get the maximum value from the score dictionary and set it to the max_val attribute.
        self.max_val = max([kl_val for _, _, kl_val, _ in self.score_dict.values()])

        # Return a dictionary with the attribute names as keys and the measure scores as values.
        return dict([(attribute, _tuple[2]) for (attribute, _tuple) in self.score_dict.items()])

    def calc_measure_internal(self, _bin: Bin) -> float:
        """
        Calculate the measure score for a bin.
        This is an abstract method, and should be implemented by the inheriting class for the specific measure.
        :param _bin: The bin object.
        :return: The measure score.
        """
        raise NotImplementedError()

    def build_operation_expression(self, source_name: str) -> str:
        """
        Get manipulation expression

        :param source_name: The source df name
        :return: string of the manipulation
        """
        raise NotImplementedError()

    def build_explanation(self, current_bin: Bin, max_col_name: str, max_value: float, source_name: str) -> str:
        """
        Build an explanation for the given bin, column name, value, and source name after calculating the measure score.
        This is an abstract method, and should be implemented by the inheriting class.
        :param current_bin: The current bin object.
        :param max_col_name: The column name with the maximum measure score.
        :param max_value: The maximum value of the column.
        :param source_name: The source name.
        """
        raise NotImplementedError()

    def get_influence_col(self, current_bin: Bin) -> Dict[Any, float]:
        """
        Compute and return the influence of the column in the current bin.
        :param current_bin: The current bin object.
        :return: A dictionary with the bin values as keys and the influence values as values.
        """
        bin_values = current_bin.get_bin_values()
        return dict(zip(bin_values, self.calc_influence_col(current_bin)))

    def calc_influence_col(self, current_bin: Bin) -> List[float]:
        """
        Calculate the influence of the column in the current bin.
        This is an abstract method, and should be implemented by the inheriting class.\n
        For FEDEX, the influence of a set of rows is defined as:\n
        .. math:: I_A (d_{in}, q, d_{out}) - I_A (d_{in} - R, q, d_{out}')\n
        Where:\n
        - :math:`I_A (d_{in}, q, d_{out})` is the measure score of the column A in the current bin.
        - :math:`d_{in}` is the input dataframe.
        - :math:`d_{out}` is the output dataframe.
        - :math:`q` is the query.
        - :math:`R` is the set of rows in the current bin.
        - :math:`d_{out}'` is the output dataframe after removing the rows in the current bin.
        :param current_bin: The current bin object.
        :return: A list of influence values. Each value corresponds to a value in the bin.
        """
        raise NotImplementedError()

    def get_max_k(self, score_dict, k) -> Tuple[List[str], List[float]]:
        """
        Get the top k attributes with the highest scores from the score dictionary.

        This method sorts the attributes based on their scores and returns the top k attributes along with their scores.

        :param score_dict: A dictionary where keys are attribute names and values are their corresponding scores.
        :param k: The number of top attributes to return.
        :return: A tuple containing two lists - the top k attribute indexes and their corresponding scores.
        """
        # Get the keys and values from the score dictionary, convert them to arrays,
        # and get the sorted indices of the values.
        score_array_keys = list(score_dict.keys())
        score_array = np.array([score_dict[i] for i in score_array_keys])
        max_indices = score_array.argsort()

        unique_max_indexes = []
        influence_vals_max_indexes = []

        # Iterate over the indices, and add the values and keys to the above 2 lists, in ascending order.
        for index in max_indices:
            current_index = score_array_keys[index]
            unique_max_indexes.append(current_index)
            influence_vals_max_indexes.append(score_dict[current_index])

        # Get the top k attributes and their scores, then return them.
        max_indices = unique_max_indexes[-k:][::-1]
        max_influences = influence_vals_max_indexes[-k:][::-1]
        return max_indices, max_influences



    def _try_fix_explanation(self, explanation: str, binned_column: pd.Series, max_value) -> str:
        """
        If the explanation is not valid, for example in the case of group-bys where we have a value instead
        of the group names, try to fix it.
        Override this method in the inheriting class if needed, otherwise leave as is (returns the explanation as is).
        :param explanation: The explanation to fix.
        :param binned_column: The binned column.
        :param max_value: The max value of the bin.
        :param max_group_value: The group with the max value.
        :return:
        """
        return explanation

    def draw_bar(self, bin_item: Bin, influence_vals: dict = None, title: str = None, ax=None, score=None,
                 show_scores: bool = False, explanation_num: int | None = None) -> None:
        """
        Draw a bar plot for the given bin item and influence values.
        This is an abstract method, and should be implemented by the inheriting class.
        :param bin_item: The bin item to draw the bar plot for.
        :param influence_vals: The influence values for the bin item.
        :param title: The title of the plot. Optional.
        :param ax: The axes to draw the plot on. Optional.
        :param score: The score of the bin item. Optional.
        :param show_scores: Whether to show the scores on the plot. Optional.
        :param explanation_num: The explanation number to display in the title. Optional.
        """
        raise NotImplementedError()

    @staticmethod
    def get_significance(influence, influence_vals_list) -> float:
        """
        A static method to calculate the significance of the influence value,
        in the context of a list of influence values.\n
        Computed as the difference between the influence value and the mean of the influence values,
        divided by the square root of the variance of the influence values.
        :param influence: The influence value.
        :param influence_vals_list: The list of influence values.
        :return: The significance value of the influence.
        """
        influence_var = np.var(influence_vals_list)
        if influence_var == 0:
            return 0.0
        influence_mean = np.mean(influence_vals_list)

        return (influence - influence_mean) / np.sqrt(influence_var)


    def _calc_influence(self, score_dict, max_col_name, results_columns) -> pd.DataFrame:
        # This silly initialization is done because of the way pandas works with concatenation.
        # Concatenating an empty dataframe with another dataframe is deprecated, so we initialize it with a single empty row.
        # This row will be dropped later.
        results = pd.DataFrame([["", "", "", "", "", ""]], columns=results_columns)
        source_name, bins, score, _ = score_dict[max_col_name]

        for current_bin in bins.bins:
            # Compute the influence values for the current bin.
            influence_vals = self.get_influence_col(current_bin)
            influence_vals_list = np.array(list(influence_vals.values()))

            # If all the influence values are NaN, skip the current bin.
            if np.all(np.isnan(influence_vals_list)):
                continue

            # Get the top k (1 in this case) attribute indexes and their corresponding influence values.
            max_values, max_influences = self.get_max_k(influence_vals, 1)

            # Iterate over the top k attributes and their influence values.
            # Compute the significance of the influence value, and if it is above the threshold, build an explanation.
            for max_value, influence_val in zip(max_values, max_influences):
                significance = self.get_significance(influence_val, influence_vals_list)
                if significance < SIGNIFICANCE_THRESHOLD:
                    continue
                explanation = self.build_explanation(current_bin, max_col_name, max_value, source_name)
                explanation = self._try_fix_explanation(explanation, current_bin, max_value)

                # Save the results in a dictionary and append it to the results dataframe.
                new_result = dict(zip(results_columns,
                                      [score, significance, influence_val, explanation, current_bin, influence_vals,
                                       current_bin.get_bin_name(), max_col_name]))
                results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)

        results = results.drop(axis='index', index=0)
        return results


    def draw_figures(self, title: str, scores: pd.Series, K: int, figs_in_row: int, explanations: pd.Series, bins: pd.Series,
                      influence_vals: pd.Series, source_name: str, show_scores: bool,
                     added_text: pd.Series = None, added_text_name: str | None = None)\
            -> tuple[list[str], plt.Figure] | None:
        """
        Draws the figures for the explanations.
        :param title: The title of the plot.
        :param scores: The scores of the attributes.
        :param K: The number of top attributes to consider.
        :param figs_in_row: The number of figures to display in a row.
        :param explanations: The explanations generated for the top K attributes.
        :param bins: The bins of the attributes.
        :param influence_vals: The influence values of the attributes.
        :param source_name: The name of the source DataFrame.
        :param show_scores: Whether to show the scores on the plot.
        :param added_text: Additional text to add to the bottom of each figure. Optional. A dict with explanation as key, and a sub-dict with 'text' and 'position' as keys.
        :param added_text_name: The name of the added text, to be displayed right above the added text.
        :return: A list of matplotlib figures containing the explanations for the top k attributes, after computing the influence.
        """
        figures = []
        # Set the title of the plot to the title if it is not None, otherwise build the operation expression.
        title = title if title else self.build_operation_expression(source_name)

        num_explanations = len(explanations) if explanations is not None else 0

        # If K is greater than 1, and there actually is more than one explanation, we need to set up a grid of subplots.
        # set the number of rows in the plot to the ceiling of the length of the scores divided by figs_in_row.
        if K > 1 and num_explanations > 1:  ###
            rows = math.ceil(len(scores) / figs_in_row)
            fig, axes = plt.subplots(rows, figs_in_row, figsize=(8 * figs_in_row, 9 * rows))
            for ax in axes.reshape(-1):
                ax.set_axis_off()
        else:
            total_text_len = 0
            if title:
                total_text_len += len(title)
            if explanations is not None and num_explanations > 0:
                total_text_len += len(explanations.iloc[0])
            # If the text is so long that it probably won't fit properly in the figure, increase the figure size.
            # Note that 300 is a fairly arbitrary threshold, made on an educated guess that the usual is around
            # 150-250 characters long, and that 300+ is probably around where it starts to get too long.
            if total_text_len > 300:
                fig, axes = plt.subplots(figsize=(9, 11))
            # Otherwise, use a smaller figure size.
            else:
                fig, axes = plt.subplots(figsize=(7, 8))

        # There are issues of the title not being properly centered when there is only one explanation,
        # so we need to check if there is only one explanation, and if so, set the title to be centered.
        if num_explanations == 1:
            # Why does x=0.6 work? I don't know, but it does and it centers the title properly.
            # Not 0.5, for some reason beyond my understanding.
            fig.suptitle(title, fontsize=15, y=1.02, x=0.6)
            axes.set_title(explanations.iloc[0], fontsize=16)
            axes.set_axis_off()
        else:
            fig.suptitle(title, fontsize=20, y=1.02)

        if num_explanations > 1:
            axes = axes.flatten()  # Flatten the axes array for easier indexing.
        else:
            axes = [axes]

        added_text_text = f"{added_text_name}:\n\n" if added_text_name else ""

        # Draw the bar plots for each explanation
        for index, (explanation, current_bin, current_influence_vals, score) in enumerate(
                zip(explanations, bins, influence_vals, scores)):

            explanation_num = index + 1 if (num_explanations > 1 and added_text is not None) else None

            figure = self.draw_bar(current_bin, current_influence_vals, title=explanation,
                                ax=axes[index], score=score,
                                show_scores=show_scores,
                                explanation_num=explanation_num
                                   )
            if figure:
                figures.append(figure)

            if added_text is not None:
                explanation_added_text = added_text.get(explanation, None)
                if explanation_added_text is not None:
                    explanation_added_text = explanation_added_text.get('added_text', "")
                    if not isinstance(explanation_added_text, str):
                        explanation_added_text = str(explanation_added_text)
                    # Replace any wrapping done previously - we want our own custom wrapping here, since we know
                    # what length of text we want to display.
                    explanation_added_text = explanation_added_text.replace("\n", " ")
                    if explanation_num is not None:
                        added_text_text += f"{START_BOLD}[{explanation_num}]{END_BOLD} {explanation_added_text}\n\n"
                    else:
                        added_text_text += f"{explanation_added_text}\n\n"


        # If there is text to add, add it to the bottom left of the plot.
        if added_text_text:
            plt.figtext(0, 0, added_text_text, horizontalalignment='left', verticalalignment='top',
                        fontsize=16, wrap=True,)

        # Adding a bit of a top margin to the plot, to make sure the title doesn't interfere with the plots.
        plt.subplots_adjust(top=0.92)
        plt.tight_layout()

        return figures, fig

    def calc_influence(self, brute_force=False, top_k=TOP_K_DEFAULT,
                       figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None,
                       deleted=None, debug_mode: bool = False, draw_figures: bool = True
                       ) -> Tuple[str, pd.Series, int, int, pd.Series, pd.Series, pd.Series, str, bool] | None | Figure | list[
        Figure] | list[str]:
        """
        Calculate the influence of each attribute in the dataset.

        This method computes the influence of each attribute by determining the change in the measure score
        when each attribute is removed from the dataset. The measure score is calculated using the `calc_measure` method.

        :param brute_force: Whether to use brute force to calculate the influence. Default is False.
        :param top_k: The number of top attributes to consider. Default is TOP_K_DEFAULT.
        :param figs_in_row: The number of figures to display in a row. Default is DEFAULT_FIGS_IN_ROW.
        :param show_scores: Whether to show the scores on the plot. Default is False.
        :param title: The title of the plot. Optional.
        :param deleted: A dictionary of deleted attributes as keys, with the values as a tuple: (dataframe name, bin object, score, column values). Optional.
        :param debug_mode: Developer option. Disables multiprocessing for easier debugging. Default is False.
        :param draw_figures: Whether to draw the figures at this stage. Default is True. If set to false, it is the responsibility of the caller to call the draw_figures method. Possible use case for leaving this false is if the caller wants to add additional information to the figures before drawing them.

        :return: A list (or a single) matplotlib figures containing the explanations for the top k attributes, after
        computing the influence. Alternatively, returns the names of the explained attributes, or None if no explanations were found.
        If the option draw_figures is set to False, returns a tuple containing the parameters needed to draw the figures instead.
        """

        # If deleted is not None, set the score dictionary to the deleted dictionary.
        # Else, set the score dictionary to the score dictionary of the measure.
        if deleted:
            score_dict = deleted
        else:
            score_dict = self.score_dict

        # Get a list of tuples of: (score, column name, bin object, column values) from the score_dict,
        # then sort that list
        score_and_col = [(score_dict[col][2], col, score_dict[col][1], score_dict[col][3])
                         for col in score_dict]

        list_scores_sorted = score_and_col
        list_scores_sorted.sort()

        # Define K, the number of top attributes to consider, as the value of top_k if deleted is None,
        # otherwise set it to the length of the deleted dictionary.
        K = top_k if not deleted else len(deleted.keys())

        # Create a dataframe for the results
        results_columns = ["score", "significance", "influence", "explanation", "bin", "influence_vals"]
        # This silly initialization is done because of the way pandas works with concatenation.
        # Concatenating an empty dataframe with another dataframe is deprecated, so we initialize it with a single empty row.
        # This row will be dropped later.
        results = pd.DataFrame([["", "","","","",""]], columns=results_columns)

        # Iterate over the top K attributes, and get the influence values for each bin.
        # From limited testing, doing this in parallel gives a small performance boost.
        if not debug_mode:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._calc_influence, score_dict, max_col_name, results_columns)
                    for score, max_col_name, bins, _ in list_scores_sorted[:-K - 1:-1]
                ]
                for future in as_completed(futures):
                    results = pd.concat([results, future.result()], ignore_index=True)
        else:
            for score, max_col_name, bins, _ in list_scores_sorted[:-K - 1:-1]:
                results = pd.concat([results, self._calc_influence(score_dict, max_col_name, results_columns)], ignore_index=True)

        results = results.drop(axis='index', index=0)

        # Compute the skyline of the results dataframe.
        results_skyline = results[results_columns[0:2]].astype("float")
        skyline = paretoset(results_skyline, ["diff", "max"])

        # Get the results for the skyline attributes.
        explanations = results[skyline]["explanation"]
        bins = results[skyline]["bin"]
        influence_vals = results[skyline]["influence_vals"]
        scores = results[skyline]["score"]

        source_name = score_dict[list_scores_sorted[-1][1]][0]

        # If there are no interesting explanations, print a message and return.
        if len(scores) == 0:
            print(f'Could not find any interesting explanations for your query over dataset {source_name}.')
            return None

        if draw_figures:
            figures = self.draw_figures(
                title=title,
                scores=scores,
                K=K,
                figs_in_row=figs_in_row,
                explanations=explanations,
                bins=bins,
                influence_vals=influence_vals,
                source_name=source_name,
                show_scores=show_scores,
            )
        else:
            return title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores

        # Return the figures if there are multiple figures, otherwise return a single figure.
        return figures

    def calc_interestingness_only(self) -> None:
        """
        Calculate and display a histogram based on the interestingness score.

        This method sorts the attributes based on their measure scores, selects the attribute with the highest score,
        and generates an explanation based solely on the interestingness score. It then draws a histogram to visualize
        the distribution of the source and result columns for the selected attribute.

        :return: None
        """
        # Get a list of tuples of: (score, column name, bin object, column values) from the score_dict,
        # then sort that list
        score_and_col = [(self.score_dict[col][2], col, self.score_dict[col][1], self.score_dict[col][3])
                         for col in self.score_dict]
        list_scores_sorted = score_and_col
        list_scores_sorted.sort()

        # Select the column name, bins and columns for the attribute with the highest score.
        _, col_name, bins, cols = list_scores_sorted[-1]

        # Get an explanation based solely on the interesting, then draw a histogram using it.
        io_explanation = self.interestingness_only_explanation(cols[0],
                                                               cols[1],
                                                               col_name)

        draw_histogram(cols[0], cols[1], col_name, io_explanation)
