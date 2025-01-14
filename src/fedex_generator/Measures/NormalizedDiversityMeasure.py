import numpy as np

from fedex_generator.Measures.DiversityMeasure import DiversityMeasure
from fedex_generator.commons import utils


class NormalizedDiversityMeasure(DiversityMeasure):
    """
    A normalized version of the diversity measure.\n
    See the DiversityMeasure class for more information.\n
    The only difference is in the calculation of the variance. The square root of the variance is divided by the mean
    of the appearances of the values in the array, where as in the DiversityMeasure class, this is not done.
    """

    def __init__(self):
        super().__init__()

    def calc_var(self, pd_array):
        if utils.is_numeric(pd_array):
            return np.abs(np.nanstd(pd_array) / np.nanmean(pd_array))

        appearances = (pd_array.value_counts()).to_numpy()
        mean = np.mean(appearances)
        variance = np.sum(np.power(appearances - mean, 2)) / len(appearances)
        print(variance, np.sqrt(variance) / mean)

        return np.sqrt(variance) / mean
