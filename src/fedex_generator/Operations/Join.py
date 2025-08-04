import pandas as pd
from pandas import DataFrame
from typing import Generator, Tuple, List

from fedex_generator.Measures.ShapleyMeasure import ShapleyMeasure
from fedex_generator.commons.consts import TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW
from fedex_generator.commons import utils
from fedex_generator.commons.DatasetRelation import DatasetRelation
from fedex_generator.Operations import Operation
from fedex_generator.Measures.ExceptionalityMeasure import ExceptionalityMeasure


class Join(Operation.Operation):
    """
    Implementation of the Join operation, fit for the FEDEx explainability framework.\n
    Provides a .explain() method for explaining the operation, as well as methods used for producing the explanation.
    """

    def __init__(self, left_df: DataFrame, right_df: DataFrame, source_scheme: dict, attribute: str,
                 result_df: DataFrame = None, left_name: str = None, right_name: str = None):
        """
        :param left_df: DataFrame to join on the left side.
        :param right_df: DataFrame to join on the right side.
        :param source_scheme: The scheme of the source DataFrame, as a dictionary.
        :param attribute: The attribute to join on.
        :param result_df: The resulting DataFrame after the join operation. Optional.
        :param left_name: Name of the left DataFrame. Optional.
        :param right_name: Name of the right DataFrame. Optional.
        """
        super().__init__(source_scheme)
        self.source_scheme = source_scheme
        self.attribute = attribute

        # If a result DataFrame is not provided, perform the join operation
        if result_df is None:
            left_name = utils.get_calling_params_name(left_df)
            right_name = utils.get_calling_params_name(right_df)
            left_df = left_df.copy().reset_index()
            right_df = right_df.copy()
            left_df.columns = [col if col in ["index", attribute] else left_name + "_" + col
                               for col in left_df]
            right_df.columns = [col if col in ["index", attribute] else right_name + "_" + col
                                for col in right_df]
            result_df = pd.merge(left_df, right_df, on=[attribute])

        self.right_name = right_name
        self.left_name = left_name
        self.left_df = left_df
        self.right_df = right_df
        self.result_df = result_df
        self._measure = None

    def iterate_attributes(self) -> Generator[Tuple[str, DatasetRelation], None, None]:
        """
        Iterate over the attributes of the left and right DataFrames.

        This method generates tuples containing an attribute and its corresponding DatasetRelation.
        It first yields attributes from the left DataFrame, then from the right DataFrame, excluding any attributes
        that are common between the two DataFrames.

        :yield: Tuples of attribute name and DatasetRelation objects with the left and right DataFrames and the result DataFrame.
        """
        for attr in self.left_df.columns:
            if isinstance(attr, str) and attr.lower() == "index":
                continue
            yield attr, DatasetRelation(self.left_df, self.result_df, self.left_name)

        for attr in set(self.right_df.columns) - set(self.left_df.columns):
            if isinstance(attr, str) and attr.lower() == "index":
                continue
            yield attr, DatasetRelation(self.right_df, self.result_df, self.right_name)

    def explain(self, schema: dict = None, attributes: List[str] = None, top_k: int = TOP_K_DEFAULT,
                figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None,
                corr_TH: float = 0.7, explainer='fedex', consider='right', cont=None, attr=None, ignore=[],
                use_sampling: bool = True, sample_size: int | float = Operation.SAMPLE_SIZE,
                debug_mode: bool = False, draw_figures: bool = False, return_scores: bool = False,
                measure_only: bool = False
                ) -> None | Tuple[str, pd.Series, int, int, pd.Series, pd.Series, pd.Series, str, bool] | Tuple:
        """
        Explain for filter operation

        :param schema: dictionary with new columns names, in case {'col_name': 'i'} will be ignored in the explanation
        :param attributes: only this attributes will be included in the explanation calculation
        :param top_k: top k explanations number, default one explanation only.
        :param show_scores: show scores on explanation
        :param figs_in_row: number of explanations figs in one row
        :param title: explanation title
        :param use_sampling: whether to use sampling or not
        :param sample_size: the size of the sample to use. Can be a percentage of the dataframe size if below 1. Default is 5000.
        :param debug_mode: Developer option. Disables multiprocessing for easier debugging. Default is False. Can possibly add more debug options in the future.
        :param draw_figures: Whether or not to draw the figures in this stage. Defaults to True. Set to False if you want to draw the figures later.
        :param return_scores: Whether or not to return the scores. Defaults to False.

        :return: explain figures
        """

        if explainer == 'shapley':
            measure = ShapleyMeasure()
            # else: score = 0
            top_fact, facts = measure.get_most_contributing_fact(self, self.left_df, self.right_df, self.attribute,
                                                                 None, consider=consider, top_k=top_k, cont=cont,
                                                                 att=attr)
            # exp = f'The result of joining dataframes \'{self.left_name}\' and \'{self.right_name}\' on attribute \'{self.attribute}\' is not empty.\nThe following fact from dataframe \'{self.left_name}\' has significantly contributed to this result:\n'
            # facts = list({k: v for k, v in sorted(facts.items(), key=lambda item: -item[1])}.items())
            # fact_idx = 0
            # exp += str(facts[fact_idx][0])
            # top_k -= 1
            # while top_k > 0:
            #     fact_idx += 1
            #     exp += '\n and then\n'
            #     exp += str(facts[fact_idx][0])
            #     top_k -= 1
            return None
        if attributes is None:
            attributes = []

        if schema is None:
            schema = {}

        backup_left_df, backup_right_df, backup_res_df, combined_source_df = None, None, None, None
        if use_sampling:
            backup_left_df, backup_right_df, backup_res_df = self.left_df, self.right_df, self.result_df
            self.left_df, self.right_df, self.result_df = self.sample(self.left_df, sample_size), self.sample(
                self.right_df, sample_size), self.sample(self.result_df, sample_size)
            # If sampling is used, calc_measure needs an unsampled source_df to create bins for its score dict, otherwise
            # the explanation creation will be affected by the sampling, and not just the measure calculation.
            # Therefore, we concat the left and right DataFrames to create the unsampled source DataFrame.
            combined_source_df = pd.concat([self.left_df, self.right_df], axis=0)

        # When using the FEDEx explainer, the exceptionality measure is used to calculate the explanation.
        measure = ExceptionalityMeasure()
        self._measure = measure
        scores = measure.calc_measure(self, schema, attributes, unsampled_source_df=combined_source_df,
                                      unsampled_res_df=backup_res_df, debug_mode=debug_mode)

        if use_sampling:
            self.left_df, self.right_df, self.result_df = backup_left_df, backup_right_df, backup_res_df

        if measure_only:
            return scores

        ret_val = measure.calc_influence(utils.max_key(scores), top_k=top_k, figs_in_row=figs_in_row,
                                      show_scores=show_scores, title=title, debug_mode=debug_mode,
                                      draw_figures=draw_figures)

        if return_scores:
            return ret_val, scores
        else:
            return ret_val, None

    def draw_figures(self, title: str, scores: pd.Series, K: int, figs_in_row: int, explanations: pd.Series, bins: pd.Series,
                      influence_vals: pd.Series, source_name: str, show_scores: bool,
                     added_text: dict | None = None, added_text_name: str | None = None) -> tuple:
        return self._measure.draw_figures(
            title=title, scores=scores, K=K, figs_in_row=figs_in_row,
            explanations=explanations, bins=bins, influence_vals=influence_vals,
            source_name=source_name, show_scores=show_scores, added_text=added_text,
            added_text_name=added_text_name
        )