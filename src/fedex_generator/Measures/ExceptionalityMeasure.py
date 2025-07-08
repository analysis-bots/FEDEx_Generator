import math

import numpy as np
import pandas as pd

from fedex_generator.commons import utils
from fedex_generator.Measures.BaseMeasure import BaseMeasure, START_BOLD, END_BOLD
from fedex_generator.Measures.Bins import Bin


class ExceptionalityMeasure(BaseMeasure):
    """
    A class implementing the Exceptionality measure, a measure for interestingness of filter / join / union operations.\n
    This measure is described in the article "FEDEX: An Explainability Framework for Data Exploration Steps" by
    Daniel Deutch, Amir Gilad, Tova Milo, Amit Mualem, Amit Somech.\n
    From the article: "this measure intuitively deems the filter result interesting if it produces an output dataframe
    which significantly deviates from the input dataframe."\n
    This measure is based on the Kolmogorov-Smirnov test, a well known statistical test used to assess whether two
    samples originate from the same probability distribution.\n
    The measure is defined as: \n
    .. math:: I_A (d_{in}, q_f, d_{out}) = KS(\\Pr(d_{in}), \\Pr(d_{out}))\n
    Where:\n
    - :math:`I_A` is the interestingness of column A.\n
    - :math:`d_{in}` is the input dataframe.\n
    - :math:`q_f` is the query operation.\n
    - :math:`d_{out}` is the output dataframe.\n
    - :math:`KS` is the Kolmogorov-Smirnov test.\n
    - :math:`\\Pr(d_{in})` is the probability distribution of the input dataframe.\n
    - :math:`\\Pr(d_{out})` is the probability distribution of the output dataframe.\n
    """

    def __init__(self):
        super().__init__()

    def draw_bar(self, bin_item: Bin, influence_vals: dict = None, title=None, ax=None, score=None,
                 show_scores: bool = False, explanation_num: int | None = None) -> str:
        """
        Draw a bar chart comparing the distribution of values before and after an operation.

        This method generates a bar chart that visualizes the distribution of values in a bin before and after
        an operation. It highlights the most influential value in green if influence values are provided.

        :param bin_item: The bin object containing the data to be visualized.
        :param influence_vals: A dictionary of influence values for each bin value (default is None).
        :param title: The title of the bar chart (default is None).
        :param ax: The matplotlib axes object to draw the bar chart on (default is None).
        :param score: The score to be displayed in the title if show_scores is True (default is None).
        :param show_scores: A boolean indicating whether to display the score in the title (default is False).

        :return: The name of the bin.
        """
        # Get the source and result columns
        res_col = bin_item.get_binned_result_column()
        src_col = bin_item.get_binned_source_column()

        # Get the probabilities of the source and result columns
        res_probs = res_col.value_counts(normalize=True)
        src_probs = None if src_col is None else src_col.value_counts(normalize=True)
        labels = set(src_probs.keys()).union(res_probs.keys())

        # Get the 25 most influential values if there are more than 25 values
        MAX_BARS = 25
        if len(labels) > MAX_BARS:
            labels, _ = self.get_max_k(influence_vals, MAX_BARS)

        labels = sorted(labels)
        probabilities = [100. * src_probs.get(item, 0) for item in labels]
        probabilities2 = [100 * res_probs.get(item, 0) for item in labels]

        width = 0.35
        ind = np.arange(len(labels))

        # Draw the bar chart
        result_bar = ax.bar(ind + width, probabilities2, width, label="After Filter")

        ax.bar(ind, probabilities, width, label="Before Filter")
        ax.legend(loc='best')

        # Highlight the most influential value in green, if influence values are provided
        if influence_vals:
            max_label, _ = self.get_max_k(influence_vals, 1)
            max_label = max_label[0]
            result_bar[labels.index(max_label)].set_color('green')

        # Set the labels and title of the bar chart
        ax.set_xticks(ind + width / 2)
        label_tags = tuple([utils.to_valid_latex(bin_item.get_bin_representation(i)) for i in labels])
        tags_max_length = max([len(tag) for tag in label_tags])
        ax.set_xticklabels(label_tags, rotation='vertical' if tags_max_length >= 4 else 'horizontal')

        ax.set_xlabel(utils.to_valid_latex(bin_item.get_bin_name() + " values"), fontsize=20)
        ax.set_ylabel("frequency(\\%)", fontsize=16)

        # Set the title of the bar chart. If show_scores is True, display the score in the title
        if explanation_num is not None:
            title = f"{START_BOLD}[{explanation_num}]{END_BOLD} {title}" if title is not None else f"{START_BOLD}[{explanation_num}]{END_BOLD}"
        if title is not None:
            if show_scores:
                ax.set_title(f'score: {score}\n {utils.to_valid_latex(title)}', fontdict={'fontsize': 20})
            else:
                ax.set_title(utils.to_valid_latex(title), fontdict={'fontsize': 20})

        # Rotate the x-axis labels by 45 degrees if there are more than 8 labels
        if len(labels) > 8:
            ax.tick_params(axis='x', rotation=45, tickdir='in')

        ax.set_axis_on()
        return bin_item.get_bin_name()  ####

    def interestingness_only_explanation(self, source_col, result_col, col_name):
        # If the column is categorical, compare the most common value before and after the operation
        if utils.is_categorical(source_col):
            vc = source_col.value_counts()
            source_max = utils.max_key(vc)
            vc = result_col.value_counts()
            result_max = utils.max_key(vc)
            return f"The distribution of column '{col_name}' changed significantly.\n" \
                   f"The most common value was {source_max} and now it is {result_max}."

        # Otherwise, compare the mean and standard deviation of the column before and after the operation
        std_source = np.sqrt(np.var(source_col))
        mean_source = np.mean(source_col)
        std = np.sqrt(np.var(result_col))
        mean = np.mean(result_col)

        return f"The distribution of column '{col_name}' changed significantly.\n" \
               f" The mean was {mean_source:.2f} and the standard " \
               f"deviation was {std_source:.2f}, and now the mean is {mean:.2f} and the standard deviation is {std:.2f}."

    def calc_measure_internal(self, bin: Bin):
        return ExceptionalityMeasure.kstest(bin.source_column.dropna(),
                                            bin.result_column.dropna())  # / len(source_col.dropna().value_counts())

    @staticmethod
    def kstest(s, r) -> float:
        """
        Calculate the Kolmogorov-Smirnov statistic between two samples.
        Uses the :func:`commons.kstest.ks_2samp` function to calculate the statistic.
        :param s: The first sample.
        :param r: The second sample.
        :return: The Kolmogorov-Smirnov statistic between the two samples.
        """
        s = np.array(s)
        s = s[s == s]
        r = np.array(r)
        r = r[r == r]
        return 0 if (len(r) == 0 or len(s) == 0) else utils.ks_2samp(s, r).statistic

    def calc_influence_col(self, current_bin: Bin):
        # Get the source and result columns
        bin_values = current_bin.get_bin_values()
        source_col = current_bin.get_source_by_values(bin_values)
        res_col = current_bin.get_result_by_values(bin_values)

        # Truncate the bin values to the 15 most common values
        if len(bin_values) > 15:
            bin_values = list(source_col.value_counts().nlargest(15).keys())

        # Get the measure score for the entire bin
        score_all = ExceptionalityMeasure.kstest(source_col, res_col)

        # Compute the influence of each value in the bin, as the difference between the score of the entire bin and the
        # score of the bin without the value
        influence = []
        for value in bin_values:
            source_col_only_list = current_bin.get_source_by_values([b for b in bin_values if b != value])
            res_col_only_list = current_bin.get_result_by_values([b for b in bin_values if b != value])

            score_without_bin = ExceptionalityMeasure.kstest(source_col_only_list, res_col_only_list)
            influence.append(score_all - score_without_bin)

        return influence

    def build_operation_expression(self, source_name):
        from fedex_generator.Operations.Filter import Filter
        from fedex_generator.Operations.Join import Join

        if isinstance(self.operation_object, Filter):
            if self.operation_object.value is not None and self.operation_object.operation_str is not None:
                return f'Dataframe {self.operation_object.source_name}, ' \
                       f'filtered on attribute {self.operation_object.attribute} ' \
                       f'{self.operation_object.operation_str} {self.operation_object.value}'
            else:
                return f'Dataframe {self.operation_object.source_name}, ' \
                       f'filtered on attribute {self.operation_object.attribute}'
        elif isinstance(self.operation_object, Join):
            return f'{self.operation_object.right_name} joined with {self.operation_object.left_name} by {self.operation_object.attribute}'

    def build_explanation(self, current_bin: Bin, col_name, max_value, source_name):
        # Get the source and result columns, then calculate the probabilities of each value in the columns
        source_col = current_bin.get_binned_source_column()
        res_col = current_bin.get_binned_result_column()

        res_probs = res_col.value_counts(normalize=True)
        source_probs = source_col.value_counts(normalize=True)

        # Add the missing values to the dictionaries with a probability of 0
        for bin_value in current_bin.get_bin_values():
            res_probs[bin_value] = res_probs.get(bin_value, 0)
            source_probs[bin_value] = source_probs.get(bin_value, 0)

        additional_explanation = []
        if current_bin.name == "NumericBin":
            # Get the index of the most influential value
            values = current_bin.get_bin_values()
            index = values.index(max_value)

            values_range_str = "below {}".format(utils.format_bin_item(values[1])) if max_value == 0 else \
                "above {}".format(utils.format_bin_item(max_value)) if index == len(values) - 1 else \
                    "between {} and {}".format(utils.format_bin_item(values[index]),
                                               utils.format_bin_item(values[index + 1]))

            # Compute the factor - the ratio between the probability of the value in the result column and the
            # probability of the value in the source column
            factor = res_probs.get(max_value, 0) / source_probs[max_value]
            proportion = "less" if factor < 1 else "more"

            # If the factor is between 0 and 1, then the value now appears less frequently, and we set the factor
            # to 1 / factor for a more intuitive explanation
            # (e.g. appears 3 times less frequently instead of 0.33 times)
            if (factor < 1 and factor > 0):
                factor = 1 / factor

            # If the factor is 0, the value no longer exists in the result column
            if factor == 0:
                additional_explanation.append(
                    f"{START_BOLD}{utils.to_valid_latex(col_name, True)}{END_BOLD} values "
                    f"{START_BOLD}{utils.to_valid_latex(values_range_str, True)}{END_BOLD}\n"
                    f"are {START_BOLD}no{END_BOLD} {START_BOLD}longer{END_BOLD} {START_BOLD}exist{END_BOLD} (was {round(source_probs[max_value], 3) * 100}%)")
            else:
                appear_test = f'{utils.smart_round(factor)} times {proportion}'
                additional_explanation.append(
                    f"{START_BOLD}{utils.to_valid_latex(col_name, True)}{END_BOLD} values "
                    f"{START_BOLD}{utils.to_valid_latex(values_range_str, True)}{END_BOLD} (in green)\n"
                    f"appear {START_BOLD}"
                    f"{utils.to_valid_latex(appear_test, True)}"
                    f"{END_BOLD} than before")
        else:
            factor = res_probs.get(max_value, 0) / source_probs[max_value]

            source_prob = 100 * source_probs[max_value]
            res_prob = 100 * res_probs.get(max_value, 0)
            max_value_rep = current_bin.get_bin_representation(max_value)

            if factor == 0:
                proportion_sentes = f' frequency was {source_prob:.1f}% now {res_prob:.1f}%'
            elif factor < 1:
                proportion = "less"
                factor = 1 / factor
                proportion_sentes = f"appear {START_BOLD} " \
                                    f"{utils.to_valid_latex(f'{utils.smart_round(factor)} times {proportion}', True)}" \
                                    f" {END_BOLD} than before"
            else:
                proportion = "more"
                proportion_sentes = f"appear {START_BOLD} " \
                                    f"{utils.to_valid_latex(f'{utils.smart_round(factor)} times {proportion}', True)}" \
                                    f" {END_BOLD} than before"

            additional_explanation.append(
                f"{START_BOLD}{utils.to_valid_latex(col_name, True)}{END_BOLD} value "
                f"{START_BOLD}{utils.to_valid_latex(max_value_rep, True)}"
                f"{END_BOLD} (in green)\n{proportion_sentes}")

        influence_top_example = ", ".join(additional_explanation)

        return influence_top_example
