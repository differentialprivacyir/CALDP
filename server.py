from __future__ import division
import math
import numpy as np


class Server:
    def __init__(self) -> None:
        pass

    def frequency_calculation(self, reports):
            if reports is not None:
                summation_data = [sum(x) for x in zip(*reports)]
                norm_est_freq = [np.nan_to_num(e / sum(summation_data)) for e in summation_data]
            
            return norm_est_freq

    def grr_frequency_estimation(self, reports, epsilon, domain):
        if reports is not None:
            n = len(reports)
            const = (math.pow(math.e, epsilon)) + domain-1
            p = math.pow(math.e, epsilon) / const
            q = 1 / const
            summation_data = [sum(x) for x in zip(*reports)]
            estimated_data = [((v - n * q) / (p - q)).clip(0) for v in summation_data]
            norm_est_freq = [np.nan_to_num(e / sum(estimated_data)) for e in estimated_data]
        return norm_est_freq

    def new_frequency_estimation(self, reports, epsilon, domain, maxcorr):
        if reports is not None:
            n = len(reports)
            const = math.pow(math.e, epsilon) + domain-1 + maxcorr
            p = math.pow(math.e, epsilon) / const
            q = (maxcorr + 1) / const
            summation_data = [sum(x) for x in zip(*reports)]
            estimated_data = [((v - n * q) / (p - q)).clip(0) for v in summation_data]
            norm_est_freq = [np.nan_to_num(e / sum(estimated_data)) for e in estimated_data]
        return norm_est_freq