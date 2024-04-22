
from __future__ import division
import math
import numpy as np
import itertools
from minepy import MINE


class Client():
    def __init__(self, epsilon, d, corr=True, maxcorr=None, allcorr = None):
        self.epsilon = epsilon
        self.domain = d
        self.corr = corr
        self.maxcorr = maxcorr
        self.allcorr = allcorr
        self.updateparam(self.epsilon, self.domain)

    def updateparam(self, epsilon=None, d=None, corr=None):
        self.epsilon = epsilon if not None else self.epsilon
        self.domain = d if not None else self.domain
        if corr:
            const = math.pow(math.e, self.epsilon) + self.domain-1 + self.maxcorr
            self.p = math.pow(math.e, self.epsilon) / const
            self.q = (self.maxcorr + 1) / const
        else:
            const = math.pow(math.e, self.epsilon) + self.domain - 1
            self.p = math.pow(math.e, self.epsilon) / const
            self.q = 1 / const

    
    def _perturbe(self, data):
        if self.corr and data in self.allcorr:
            self.updateparam(self.epsilon, self.domain, corr=True)
        
        if self.corr:
            if np.random.rand() > (self.p - self.q):
                pert_data = np.random.randint(0, self.domain)
            else:
                pert_data = data
        else:
            if np.random.rand() < self.p:
                pert_data = data
            else:
                pert_data = np.random.randint(0, self.domain-2)
                if pert_data == data:
                    pert_data = self.domain-1

        return pert_data

    def privetise(self, data):
        privlst = []
        for day in range(len(data)):
            privday = list(map(lambda x : self._perturbe(x), data[day]))
            privlst.append(privday)

        return privlst


class CorrelationCal:
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