import math
from typing import Dict, List, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from paretoset import paretoset

from fedex_generator.Measures.Bins import Bins, Bin
from fedex_generator.commons.consts import SIGNIFICANCE_THRESHOLD, TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW
from fedex_generator.commons.utils import is_numeric, to_valid_latex

from FEDEx_Generator.src.fedex_generator.Operations.Operation import Operation
from FEDEx_Generator.src.fedex_generator.commons.DatasetRelation import DatasetRelation

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

    def calc_measure(self, operation_object: Operation, scheme: dict, use_only_columns: list, ignore:list=[]) -> Dict[str, float]:
        """
        Calculate the measure for each attribute in the operation_object.

        :param operation_object: The operation object - the operation that was performed, one of the classes in the Operations package.
        :param scheme: The scheme of the columns, in the form of a dictionary, where the key is the column name and the value is the scheme of the column. Columns with the scheme 'i' will be ignored.
        :param use_only_columns: A list of columns to include in the explanation. If empty, all columns will be included.
        :param ignore: A list of columns to ignore in the explanation. Set to [] by default.
        """
        # Set the operation object, the scheme, and the score dictionary to the given values.
        # Also initialize the max_val to -1.
        self.operation_object = operation_object
        self.score_dict = {}
        self.max_val = -1
        self.scheme = scheme

        # Iterate over the attributes in the operation object.
        for attr, dataset_relation in operation_object.iterate_attributes():
            # If the attribute is in the ignore list, skip it.
            if attr in ignore:
                continue

            # Get the column scheme from the scheme dictionary. If not found, set it to 'ni'.
            column_scheme = scheme.get(attr, "ni").lower()
            # If the column scheme is 'i', skip the attribute.
            if column_scheme == "i":
                continue

            # If there are columns in the use_only_columns list, and the attribute is not in the list, skip it.
            if len(use_only_columns) > 0 and attr not in use_only_columns:
                continue

            # Get the source and result columns for the attribute.
            source_col, res_col = self.get_source_and_res_cols(dataset_relation, attr)
            # If the result column is empty, skip the attribute.
            if len(res_col) == 0:
                continue

            # Create bin candidates from the source and result columns, with a bin count specified by the operation object.
            size = operation_object.get_bins_count()

            bin_candidates = Bins(source_col, res_col, size)

            # Compute the measure score for each bin candidate, and get the maximum score.
            measure_score = -np.inf
            for bin_ in bin_candidates.bins:
                measure_score = max(self.calc_measure_internal(bin_), measure_score)

            # Update the score dictionary with the attribute, the source name,
            # the bin candidates, the measure score,
            # and the source and result columns after computing the measure score.
            self.score_dict[attr] = (
                dataset_relation.get_source_name(), bin_candidates, measure_score, (source_col, res_col))

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

    def draw_bar(self, bin_item: Bin, influence_vals: dict = None, title:str=None, ax=None, score=None,
                 show_scores: bool = False) -> None:
        """
        Draw a bar plot for the given bin item and influence values.
        This is an abstract method, and should be implemented by the inheriting class.
        :param bin_item: The bin item to draw the bar plot for.
        :param influence_vals: The influence values for the bin item.
        :param title: The title of the plot. Optional.
        :param ax: The axes to draw the plot on. Optional.
        :param score: The score of the bin item. Optional.
        :param show_scores: Whether to show the scores on the plot. Optional.
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

    def calc_influence(self, brute_force=False, top_k=TOP_K_DEFAULT,
                       figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None,
                       deleted=None) -> matplotlib.pyplot.Figure | List[matplotlib.pyplot.Figure]:
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
        :return: A list of figures if there are multiple figures, otherwise a single figure.
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
        results = pd.DataFrame([], columns=results_columns)
        figures = []

        # Iterate over the top K attributes, and get the influence values for each bin.
        for score, max_col_name, bins, _ in list_scores_sorted[:-K - 1:-1]:
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

                    # Save the results in a dictionary and append it to the results dataframe.
                    new_result = dict(zip(results_columns,
                                          [score, significance, influence_val, explanation, current_bin, influence_vals,
                                           current_bin.get_bin_name(), max_col_name]))
                    results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)

        # Compute the skyline of the results dataframe.
        results_skyline = results[results_columns[0:2]].astype("float")
        skyline = paretoset(results_skyline, ["diff", "max"])

        # Get the results for the skyline attributes.
        explanations = results[skyline]["explanation"]
        bins = results[skyline]["bin"]
        influence_vals = results[skyline]["influence_vals"]
        scores = results[skyline]["score"]

        # If there are no interesting explanations, print a message and return.
        if len(scores) == 0:
            print(f'{source_name} dataset there is not interesting explanation')
            return

        # If K is greater than 1,
        # set the number of rows in the plot to the ceiling of the length of the scores divided by figs_in_row.
        if K > 1:  ###
            rows = math.ceil(len(scores) / figs_in_row)
            fig, axes = plt.subplots(rows, figs_in_row, figsize=(5 * figs_in_row, 6 * rows))
            for ax in axes.reshape(-1):
                ax.set_axis_off()
        else:
            fig, axes = plt.subplots(figsize=(5, 5))

        # Set the title of the plot to the title if it is not None, otherwise build the operation expression.
        title = title if title else self.build_operation_expression(source_name)

        fig.suptitle(title, fontsize=20)

        # Draw the bar plots for each explanation
        for index, (explanation, current_bin, current_influence_vals, score) in enumerate(
                zip(explanations, bins, influence_vals, scores)):

            fig = self.draw_bar(current_bin, current_influence_vals, title=explanation,
                                ax=axes.reshape(-1)[index] if K > 1 else axes, score=score,
                                show_scores=show_scores)  ###
            if fig:
                figures.append(fig)

        plt.tight_layout()

        # Return the figures if there are multiple figures, otherwise return a single figure.
        return figures if len(figures) > 0 else fig

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
