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

    def encode(self, data, domain):
        days = len(data)
        encodelst = list()
        for day in range(days):
            lst = np.zeros(domain, int)
            lst1 = list(map(lambda x : np.put(lst, x, 1), data[day]))
            encodelst.append(list(lst))
        return encodelst

    def frequency(self, data, privetised, epsilon, domain, maxcorr=None):
        client_number = len(data)
        clientrawdata_encode = []
        allprivetised_encode = []
        rawsum = []
        privetisedsum = []
        for client in range(client_number):
            clientrawdata = data[client]
            clientprivdata = privetised[client]
            clientrawdata_encode.append(self.encode(clientrawdata, domain))
            allprivetised_encode.append(self.encode(clientprivdata, domain))
            rawsum.append([sum(x) for x in zip(*clientrawdata_encode[client])])
            privetisedsum.append([sum(x) for x in zip(*allprivetised_encode[client])])
            
        totalraw_frequency = self.frequency_calculation(rawsum)

        totalprivetisedre_fquency = self.new_frequency_estimation(privetisedsum, epsilon, domain, maxcorr)
        
        return totalraw_frequency, totalprivetisedre_fquency
