from typing import List
import pandas as pd
import numpy as np

from fedex_generator.commons.consts import TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW

SAMPLE_SIZE = 5000
RANDOM_SEED = 42


class Operation:
    """
    An abstract class for operations within the FEDEx explainability framework.\n
    All implemented operations should inherit from this class.
    """

    def __init__(self, scheme: dict):
        """
        :param scheme: the scheme of the dataset
        """
        self.scheme = scheme
        self.bins_count = 500

    def set_bins_count(self, n: int) -> None:
        """
        Set the number of bins used when explaining the operation.
        :param n: the number of bins
        """
        self.bins_count = n

    def get_bins_count(self) -> int:
        """
        :return: The number of bins used when explaining the operation.
        """
        return self.bins_count

    def explain(self, schema: dict = None, attributes: List[str] = None, top_k: int = TOP_K_DEFAULT,
                figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None,
                corr_TH: float = 0.7, use_sampling: bool = True):
        """
        Explain for operation
        :param schema: dictionary with new columns names, in case {'col_name': 'i'} will be ignored in the explanation
        :param attributes: only this attributes will be included in the explanation calculation
        :param top_k: top k explanations number, default one explanation only.
        :param figs_in_row: number of explanations figs in one row
        :param show_scores: show scores on explanation
        :param title: explanation title


        :return: explain figures
        """
        raise NotImplementedError()

    def present_deleted_correlated(self, figs_in_row: int = DEFAULT_FIGS_IN_ROW):
        """
        Present the attributes that were deleted due to high correlation.
        :param figs_in_row: number of explanations figs in one row.
        """
        return NotImplementedError()

    def sample(self, df: pd.DataFrame, sample_size: int | float = SAMPLE_SIZE) -> pd.DataFrame:
        """
        Uniformly sample the dataframe to the given sample size.
        :param df: The dataframe to sample.
        :param sample_size: The sample size to use. Default is SAMPLE_SIZE. If the sample size is below 1,
         it is considered a percentage of the dataframe size.
        :return: The sampled dataframe.
        """
        # If the sample size is below 1, we consider it to be a percentage of the dataframe size.
        if sample_size <= 0:
            raise ValueError("Sample size must be a positive number.")
        if 0 < sample_size < 1:
            sample_size = int(df.shape[0] * sample_size)
        # If the sample size is below the default sample size, we use the default sample size.
        # We do this because we know, from empirical testing, that our default size is a good balance between
        # performance and accuracy, and going below that size can lead to too big a loss in accuracy for
        # a very small gain in performance.
        if sample_size < SAMPLE_SIZE:
            sample_size = SAMPLE_SIZE
        if df.shape[0] <= sample_size:
            return df
        else:
            # We use a set seed so that the user will always get the same explanation when using sampling.
            generator = np.random.default_rng(RANDOM_SEED)
            uniform_indexes = generator.choice(df.index, sample_size, replace=False)
            return df.loc[uniform_indexes]
