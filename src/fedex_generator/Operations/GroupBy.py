import pandas as pd
from pandas import Series, DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy

from fedex_generator.commons.consts import TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW
from fedex_generator.commons import utils
from fedex_generator.commons.DatasetRelation import DatasetRelation
from fedex_generator.Operations import Operation
from fedex_generator.Measures.NormalizedDiversityMeasure import NormalizedDiversityMeasure
from fedex_generator.Measures.DiversityMeasure import DiversityMeasure
from external_explainers import OutlierExplainer

from typing import Generator, List, Tuple


class GroupBy(Operation.Operation):
    """
    Implementation of the GroupBy operation, fit for the FEDEx explainability framework.\n
    Provides a .explain() method for explaining the operation, as well as methods used for producing the explanation.
    """

    def __init__(self, source_df, source_scheme, group_attributes, agg_dict, result_df=None, source_name=None,
                 operation=None):
        """
        :param source_df: The source DataFrame, before the groupby operation.
        :param source_scheme: The scheme of the source DataFrame.
        :param group_attributes: The attributes to group by.
        :param agg_dict: The aggregation dictionary.
        :param result_df: The resulting DataFrame after the groupby operation.
        :param source_name: The name of the source DataFrame.
        :param operation: The operation to perform.
        """
        super().__init__(source_scheme)
        # Set the attributes
        self.source_scheme = source_scheme
        self.group_attributes = group_attributes
        self.agg_dict = agg_dict
        self.source_name = source_name
        self.source_df = source_df
        self.source_name = utils.get_calling_params_name(source_df)

        # If the result DataFrame is None, perform the groupby and aggregation
        if result_df is None:
            self.source_name = utils.get_calling_params_name(source_df)
            source_df = source_df.reset_index()
            group_attributes = self.get_one_to_many_attributes(source_df, group_attributes)
            self.result_df = source_df.groupby(group_attributes).agg(agg_dict)
        else:
            self.result_df = result_df
            self.result_name = utils.get_calling_params_name(result_df)


    def iterate_attributes(self) -> Generator[Tuple[str, DatasetRelation], None, None]:
        """
        Iterate over the attributes of the result DataFrame.

        This generator function yields each attribute of the result DataFrame along with a DatasetRelation object with the result DF.
        It skips the attribute if it is named 'index'.

        :yield: A tuple containing the attribute name and a DatasetRelation object.
        """
        for attr in self.result_df.columns:
            if attr.lower() == "index":
                continue
            yield attr, DatasetRelation(None, self.result_df, self.source_name)

    def get_source_col(self, filter_attr, filter_values, bins):
        return None

    def explain(self, schema: dict=None, attributes: List[str]=None, top_k: int=TOP_K_DEFAULT, explainer: str='fedex',
                target=None, dir: str | int=None, control=None, hold_out=[],
                figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None,
                corr_TH: float = 0.7, consider='right', cont=None, attr=None, ignore=[],
                use_sampling=True, sample_size: int | float = Operation.SAMPLE_SIZE):
        """
        Explain for group by operation
        :param schema: dictionary with new columns names, in case {'col_name': 'i'} will be ignored in the explanation
        :param attributes: only this attributes will be included in the explanation calculation
        :param top_k: top k explanations number, default one explanation only.
        :param show_scores: show scores on explanation
        :param figs_in_row: number of explanations figs in one row
        :param title: explanation title
        :param dir: direction of the outlier. Can be 'high' or 'low', or the corresponding integer values 1 and -1 (HIGH and LOW constants).
        :param use_sampling: whether to use sampling for the explanation.
        :param sample_size: the sample size to use when sampling the data. Can be an integer or a float between 0 and 1. Default is 5000.

        :return: explain figures
        """

        if explainer == 'outlier':
            res_col = None
            measure = OutlierExplainer()

            # Get the result column. Despite this being a loop - the 2nd returned value of iterate_attributes is always the same.
            # The change is only in the ignored first value. We simply need a loop to use the generator.
            for attr, dataset_relation in self.iterate_attributes():
                _, res_col = dataset_relation.get_source(attr), dataset_relation.get_result(attr)
                res_col = res_col[~res_col.isnull()]
                break

            # Get the aggregation attribute and method
            try:
                agg = list(self.agg_dict.items())[0]
            except:
                agg = self.agg_dict.items()
            agg_attr, agg_method = agg[0], agg[1][0]

            if dir == 'high':
                dir = 1
            elif dir == 'low':
                dir = -1

            if type(self.group_attributes) == list:
                g_attr = self.group_attributes[0]
            else:
                g_attr = self.group_attributes

            return measure.explain(res_col, self.source_df, g_attr, agg_attr, agg_method, target, dir, control,
                                           hold_out)
        if schema is None:
            schema = {}

        if attributes is None:
            attributes = []


        backup_source_df, backup_res_df = None, None
        if use_sampling:
            backup_source_df, backup_res_df = self.source_df, self.result_df
            self.source_df, self.result_df = self.sample(self.source_df), self.sample(self.result_df)

        # Unless the outlier explainer is used, the diversity measure is always used for the groupby operation.
        measure = DiversityMeasure()
        scores = measure.calc_measure(self, schema, attributes, ignore=ignore, unsampled_source_df=backup_source_df,
                                      unsampled_res_df=backup_res_df)
        figures = measure.calc_influence(utils.max_key(scores), top_k=top_k, figs_in_row=figs_in_row,
                                         show_scores=show_scores, title=title)

        if use_sampling:
            self.source_df, self.result_df = backup_source_df, backup_res_df

        return figures

    @staticmethod
    def get_one_to_many_attributes(df, group_attributes):
        """
        Identify and append one-to-many relationship attributes to the group attributes list.

        This function iterates over the columns of the DataFrame and checks if there is a one-to-many relationship
        between the specified group attributes and other columns. If such a relationship is found, the corresponding
        column is appended to the group attributes list.

        :param df: The DataFrame to check for one-to-many relationships.
        :param group_attributes: The list of attributes to group by.
        :return: The updated list of group attributes including one-to-many relationship attributes.
        """
        for col in group_attributes:
            for candidate_col in df:
                if candidate_col in group_attributes:
                    continue

                if GroupBy._is_one_to_many(df, col, candidate_col):
                    group_attributes.append(candidate_col)

        return group_attributes

    @staticmethod
    def _is_one_to_many(df: DataFrame, col1: str, col2: str) -> bool:
        """
        Check if there is a one-to-many relationship between two columns in a DataFrame.

        This function determines if there is a one-to-many relationship between the specified columns
        by checking if each unique value in `col1` maps to a unique value in `col2`.

        :param df: The DataFrame to check for the one-to-many relationship.
        :param col1: The first column to check.
        :param col2: The second column to check.
        :return: True if there is a one-to-many relationship, False otherwise.
        """
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df)

        # Check if the first 1000 rows have a one-to-many relationship. This is done first for performance reasons.
        first_max_cheap_check = df[[col1, col2]].head(1000).groupby(col1).nunique()[col2].max()
        # If the first 1000 rows do not have a one-to-many relationship, return False
        if first_max_cheap_check != 1:
            return False

        # Check if the entire DataFrame has a one-to-many relationship
        first_max = df[[col1, col2]].groupby(col1).nunique()[col2].max()
        return first_max == 1

    def _get_columns_names(self):
        """
        Generate a list of column names for the result DataFrame.

        This method processes the columns of the result DataFrame and constructs a list of column names.
        It handles cases where columns are tuples (e.g., from multi-index DataFrames) by joining the tuple elements.
        It also appends the aggregation method to the column names if applicable.

        :return: A list of column names for the result DataFrame.
        """
        columns = []
        for column in list(self.result_df.columns):
            if isinstance(column, tuple):
                columns.append("_".join(column))
            else:
                if column in self.agg_dict:
                    columns.append(f'{column}_{self.agg_dict[column]}')
                elif isinstance(self.agg_dict, str) and not column.endswith(f'_{self.agg_dict}'):
                    columns.append(f'{column}_{self.agg_dict}')
                else:
                    columns.append(f'{column}')

        return columns
