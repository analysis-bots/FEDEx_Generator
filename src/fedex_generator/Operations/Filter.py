import operator
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame

from fedex_generator.commons.consts import TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW
from fedex_generator.commons import utils
from fedex_generator.commons.DatasetRelation import DatasetRelation
from fedex_generator.Operations import Operation
from fedex_generator.Measures.ExceptionalityMeasure import ExceptionalityMeasure
from typing import List, Generator, Tuple

operators = {
    "==": operator.eq,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "!=": operator.ne,
    "between": lambda x, tup: x.apply(lambda item: tup[0] <= item < tup[1])
}


def do_operation(a, b, op_str):
    return operators[op_str](a, b)


class Filter(Operation.Operation):
    """
    An implementation of the filter operation, fit for the explainability framework.\n
        Provides a .explain() method for explaining the operation, as well as methods used for producing the explanation.
    """

    def __init__(self, source_df: DataFrame, source_scheme: dict, attribute: str = None,
                 operation_str: str = None, value=None, result_df: DataFrame = None, use_sampling: bool = True):
        """
        :param source_df: The source DataFrame, before the filter operation.
        :param source_scheme: The scheme of the source DataFrame.
        :param attribute: The attribute to filter by.
        :param operation_str: The operation to perform. Only needed if result_df is None.
        :param value: The value to filter by. Only needed if result_df is None.
        :param result_df: The resulting DataFrame after the filter operation.
        :param use_sampling: Whether to use sampling to speed up the generation of explanations. Note that this may
        affect the accuracy and quality of the explanations. Default is True.
        """
        super().__init__(source_scheme)
        self.source_df = source_df.reset_index()
        self.attribute = attribute
        self.source_scheme = source_scheme
        self.cor_deleted_atts = {}
        self.not_presented = {}
        self.corr = self.source_df.corr(numeric_only=True)
        self.type = 'filter'

        # If the result_df is not given, we calculate it.
        if result_df is None:
            self.operation_str = operation_str
            self.value = value
            self.result_df = self.source_df[do_operation(self.source_df[attribute], value, operation_str)]
        else:
            self.result_df = result_df
            self.result_name = utils.get_calling_params_name(result_df)

        if use_sampling:
            # If sampling is used, we want to have a backup of the original source and result DataFrames, for
            # any potential future need of them.
            self.unsampled_source_df = source_df
            self.unsampled_result_df = self.result_df
            self.source_df = self.sample(self.source_df)
            self.result_df = self.sample(self.result_df)
        self.source_name = utils.get_calling_params_name(source_df)
        self._high_correlated_columns = None

    def get_correlated_attributes(self) -> List[str]:
        """
        Get the attributes that are highly correlated with the specified attribute.

        This function calculates the correlation matrix for the numeric columns in the source DataFrame.
        It then identifies and returns the columns that have a correlation coefficient greater than 0.85
        with the specified attribute.

        :return: A list of attributes that are highly correlated with the specified attribute.
        """
        # Avoid repeating the calculation every single time.
        if self._high_correlated_columns is not None:
            return self._high_correlated_columns
        # For performance, we only take the first 10000 rows. We also copy the df because otherwise we
        # are modifying the original df.
        numeric_df = self.source_df.head(10000).copy()

        # For every non-numeric column, we map it to a numeric value by sorting the unique values and assigning
        # them a number.
        for column in numeric_df:
            try:
                if utils.is_numeric(numeric_df[column]):
                    continue

                items = sorted(numeric_df[column].dropna().unique())
                items_map = dict(zip(items, range(len(items))))
                # Changed from numeric_df[column] = numeric_df[column].map(items_map) to avoid SettingWithCopyWarning
                # and future deprecation.
                numeric_df.loc[:, column] = numeric_df[column].map(items_map)
            except Exception as e:
                print(e)

        corr = numeric_df.corr()
        high_correlated_columns = []
        if self.attribute in corr:
            df = corr[self.attribute]

            df = df[df > 0.85].dropna()
            high_correlated_columns = list(df.index)

        self._high_correlated_columns = high_correlated_columns

        return high_correlated_columns

    def iterate_attributes(self) -> Generator[Tuple[str, DatasetRelation], None, None]:
        """
        Iterate over the attributes of the result DataFrame.

        This function yields attributes from the result DataFrame that are not the index,
        the specified attribute, or highly correlated with the specified attribute.
        It also skips attributes marked as 'i' in the source scheme.

        :yield: A tuple containing the attribute name and a DatasetRelation object with the source and result DF.
        """
        high_correlated_columns = self.get_correlated_attributes()

        for attr in self.result_df.columns:
            if attr.lower() == "index" or attr.lower() == self.attribute.lower() or \
                    self.source_scheme.get(attr, None) == 'i' or attr in high_correlated_columns:
                continue
            yield attr, DatasetRelation(self.source_df, self.result_df, self.source_name)

    def get_source_col(self, filter_attr, filter_values, bins) -> pd.Series | None:
        """
        Get the binned column from the source DataFrame for the specified filter attribute and values.

        This function bins the specified attribute in the source DataFrame into the given number of bins and
        returns the binned column that matches the specified filter values.

        :param filter_attr: The attribute to filter by.
        :param filter_values: The values to filter by.
        :param bins: The number of bins to use for binning the attribute.
        :return: The binned column that matches the filter values, or None if the attribute is not in the source DataFrame.
        """
        if filter_attr not in self.source_df:
            return None

        binned_col = pd.cut(self.source_df[filter_attr], bins=bins, labels=False, include_lowest=True,
                            duplicates='drop')

        return binned_col[binned_col.isin(filter_values)]

    def explain(self, schema=None, attributes=None, top_k=TOP_K_DEFAULT,
                figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None,
                corr_TH: float = 0.7, explainer='fedex', consider='right', cont=None, attr=None, ignore=[]) -> None:
        """
        Explain for filter operation

        :param schema: dictionary with new columns names, in case {'col_name': 'i'} will be ignored in the explanation
        :param attributes: only this attributes will be included in the explanation calculation
        :param top_k: top k explanations number, default one explanation only.
        :param show_scores: show scores on explanation
        :param figs_in_row: number of explanations figs in one row
        :param title: explanation title
        :param corr_TH: Correlation threshold for deleting correlated attributes
        :param explainer: Which explainer to use. Currently, only 'fedex' is supported for this operation.
        :param consider: Unused but kept for compatibility.
        :param cont: Unused but kept for compatibility.
        :param attr: Unused but kept for compatibility.
        :param ignore: Unused but kept for compatibility.

        :return: explain figures
        """

        if attributes is None:
            attributes = []

        if schema is None:
            schema = {}

        # if use_sampling:
        #     source_df_backup, result_df_backup = self.source_df, self.result_df
        #     self.source_df, self.result_df = self.sample(self.source_df), self.sample(self.result_df)

        # The measure used for filer operations is always the ExceptionalityMeasure.
        measure = ExceptionalityMeasure()
        scores = measure.calc_measure(self, schema, attributes)

        self.delete_correlated_atts(measure, TH=corr_TH)

        # Get the explanation figures.
        figures = measure.calc_influence(utils.max_key(scores), top_k=top_k, figs_in_row=figs_in_row,
                                         show_scores=show_scores, title=title)
        if figures:
            self.correlated_notes(figures, top_k)

        # if use_sampling:
        #     self.source_df, self.result_df = source_df_backup, result_df_backup
        return None

    def present_deleted_correlated(self, figs_in_row: int = DEFAULT_FIGS_IN_ROW):
        """
        Present the attributes that were deleted due to high correlation.

        This method calculates the influence of the deleted attributes and generates figures to visualize them.
        It uses the ExceptionalityMeasure to compute the influence values and displays the figures in a grid layout.

        :param figs_in_row: The number of figures to display in one row. Default is the value of DEFAULT_FIGS_IN_ROW.
        """
        measure = ExceptionalityMeasure()
        measure.calc_influence(deleted=self.not_presented, figs_in_row=figs_in_row)

    def correlated_notes(self, figures, top_k) -> None:
        """
        Add notes about correlated attributes to the figures.

        This method iterates through the figures and checks for attributes that were deleted due to high correlation.
        If such attributes are found, it adds a note to the figure explaining which attributes were deleted and why.
        It also stores the deleted attributes in the `not_presented` dictionary for later use.

        :param figures: The list of figures to add notes to.
        :param top_k: The number of top attributes to consider.
        """
        txt = ""
        lentxt = 0
        self.not_presented = {}

        # Go over the figures and check if any of the attributes were deleted due to high correlation.
        for i in range(len(figures)):
            for cor_del in self.cor_deleted_atts.keys():
                # If the attribute was deleted and it is not the last figure, add a note to the figure.
                if figures[i] == cor_del[1] and i < (top_k - 1):
                    lentxt += 1
                    txt += "[" + str(lentxt) + "] " + "The attribute " + cor_del[
                        0] + " is not presented as it correlates with " + cor_del[1] + " (cor: " + str(
                        round(self.corr[cor_del[0]][cor_del[1]], 2)) + ")\n"
                    self.not_presented[cor_del[0]] = self.cor_deleted_atts[cor_del]
        if lentxt > 0:
            txt += "\nIn order to view the not presented attributes, please execute the following: df.present_deleted_correlated()"

        # Add the notes to the figures and show them.
        plt.figtext(0, 0, txt, horizontalalignment='left', verticalalignment='top')

    def delete_correlated_atts(self, measure, TH=0.7):
        """
        Delete attributes that are highly correlated with each other.

        This method identifies pairs of attributes in the correlation matrix that have a correlation coefficient
        greater than the specified threshold (TH). It then deletes the attribute with the lower measure score
        from the measure's score dictionary.

        :param measure: The measure object containing the score dictionary.
        :param TH: The correlation threshold for deleting attributes. Default is 0.7.
        """

        self.cor_deleted_atts = {}
        corelated_atts = []
        attributes = self.corr.keys()
        numattributes = len(attributes)

        for att in range(numattributes):
            # For each pair of attributes, check if they are highly correlated.
            for att1 in range(att, numattributes):
                cor = self.corr[attributes[att]][attributes[att1]]
                # If they are, and they are not the same attribute, add them to the list of correlated attributes.
                if (cor > TH or cor < -TH) and not att == att1:
                    corelated_atts.append([attributes[att], attributes[att1]])

        # For each pair of correlated attributes, delete the attribute with the lower measure score.
        # Also, store the deleted attribute in the `cor_deleted_atts` dictionary.
        for cor in corelated_atts:
            if (len(set(cor) - set(measure.score_dict))) == 0:
                if measure.score_dict[cor[0]][2] > measure.score_dict[cor[1]][2]:
                    self.cor_deleted_atts[cor[1], cor[0]] = measure.score_dict[cor[1]]
                    del measure.score_dict[cor[1]]
                else:
                    self.cor_deleted_atts[cor[0], cor[1]] = measure.score_dict[cor[0]]
                    del measure.score_dict[cor[0]]
