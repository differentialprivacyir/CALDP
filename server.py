from __future__ import division
import math
import numpy as np
import itertools
from minepy import MINE



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
            estimated_data = [((v - n * q) / (p - q)).clip(0) for v in reports]
        return estimated_data

    def encode(self, data, domain):
        days = len(data)
        encodelst = list()
        for day in range(days):
            lst = np.zeros(domain, int)
            lst1 = list(map(lambda x : np.put(lst, x, 1), data[day]))
            encodelst.append(list(lst))
        return encodelst

    def frequency(self, privetised, epsilon, domain, maxcorr=None):
        client_number = len(privetised)
        allprivetised_encode = []
        estimated_data = []
        for client in range(client_number):
            clientprivdata = privetised[client]
            allprivetised_encode.append(self.encode(clientprivdata, domain))
            privetisedsum = [sum(x) for x in zip(*allprivetised_encode[client])]
            totalprivetisedre_fquency_client = self.new_frequency_estimation(privetisedsum, epsilon, domain, maxcorr[client])
            estimated_data.append(totalprivetisedre_fquency_client)
        
        totalprivetisedre_fquency = [sum(x) for x in zip(*estimated_data)]
        totalprivetisedre_fquency_all = [np.nan_to_num(e / sum(totalprivetisedre_fquency)) for e in totalprivetisedre_fquency]
        
        return totalprivetisedre_fquency_all
    
class SCorrelationCal:
    def __init__(self, data, fixthreshold = None):
        self.data = data
        self.n_rows = data.shape[0]
        self.n_cols = data.shape[1]
        self.fixthreshold = fixthreshold
    
    # Return unique list of tuples from domain
    def TupleGen (self, domain):
        tuplelist = []
        tuplelist.extend(list(itertools.permutations(domain, r=2)))
        tuplelist = list(map(lambda item: sorted(item), tuplelist))
        tuplelist.sort()
        tuple_list = list(tuplelist for tuplelist,_ in itertools.groupby(tuplelist))
        return tuple_list
    
    # Calculate joint probability of each tuple
    def TupleJointProb(self, tuple):
        sum = 0
        for row in range(self.n_rows):
            if all(x in self.data[row] for x in tuple):
                sum += 1
        prob = sum / self.n_rows
        return prob
    
    # Return successive overlapping pairs taken from the input
    def pairwise(self, iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = itertools.tee(iterable)
        next(b, None)
        pairslist = list(zip(a, b))
        return pairslist

    # Calculate Correlation between existance of two singla data in one day. Return a list, first element
    # contain the tuple and second element is their joint probability over all data user.
    def InterCorrCal(self, datadomain):
        tuples = self.TupleGen(datadomain)
        joinprobmatrix = []
        for item in tuples:
            joinprob = self.TupleJointProb(item)
            if joinprob >= self.fixthreshold:
                joinprobmatrix.append([tuple(item), joinprob])
        
        return joinprobmatrix
    
    # Caclualte correlation of a user data between different days, using mutual information
    # from information theory
    def IntraCorrCal(self):
        IntraMI = []
        for item1 in self.data:
            preIntraMI = []
            for item2 in self.data:
                mine =MINE()
                mine.compute_score(item1,item2)
                score = mine.mic()
                preIntraMI.append(score)
            IntraMI.append(preIntraMI)
        return IntraMI

    # Calculate correlation of one single data with its next data using Markov Chain
    def NextDayCorrCal(self, states):
        transitionName = []
        transitionName.extend(list(itertools.product(states, repeat=2)))
        totaltransition = (self.n_cols-1) * self.n_rows
        transitionMatrix = []
        for pairs in transitionName:
            totalpairs = 0
            for x_index in range (self.n_rows):
                totalpairs += np.shape(np.asarray(pairs in list(self.pairwise(list(self.data[x_index])))).nonzero())[1]
            prob = totalpairs / totaltransition

            if prob >= self.fixthreshold:
                transitionMatrix.append([tuple(pairs), prob])
        return transitionMatrix
