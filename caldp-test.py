from __future__ import division
from numpy import shape
import itertools
from minepy import MINE
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from statistics import mean
import matplotlib.pyplot as plt
from server import Server
from client import Client

epsilon = 1
fixthreshold = 0.8
calperpoint = 50
domain = 20
client_number = 10000
maxdays = 30


class CorrelationCal:
    def __init__(self, data):
        self.data = data
        self.n_rows = data.shape[0]
        self.n_cols = data.shape[1]
    
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
                totalpairs += shape(np.asarray(pairs in list(self.pairwise(list(self.data[x_index])))).nonzero())[1]
            prob = totalpairs / totaltransition

            transitionMatrix.append([tuple(pairs), prob])
        return transitionMatrix

# Calls CorrelationCal function from CorrCal file to calculate three types of correlation fr each user
def _corrcal(data, domain, maxcorr = False, days = maxdays, correlation = True):
    client_number = len(data)
    InterMICorrdegree = dict()
    NextMICorrdegree = dict()
    max_corr = dict()
    for client in range(client_number):
        if correlation:
            clientdata = data[client][:days]
            clientdatacorr = CorrelationCal(np.array(clientdata))
            InterMICorrdegree[client]  = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
            NextMICorrdegree[client] = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))

        if maxcorr:
            inter = list(item[1] for item in InterMICorrdegree[client])
            next = list(item[1] for item in NextMICorrdegree[client])
            max_corr[client] = max(inter + next)

    return InterMICorrdegree, NextMICorrdegree, max_corr

def newcorrcal(data, domain):
    clientdata = data
    clientdatacorr = CorrelationCal(np.array(clientdata))
    InterMICorrdegree  = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
    NextMICorrdegree = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))
    inter = list(item[1] for item in InterMICorrdegree)
    next = list(item[1] for item in NextMICorrdegree) 
    #intra = list(itertools.chain(*IntraMICorrdegree))
    max_corr = max(inter + next)
    return InterMICorrdegree, NextMICorrdegree, max_corr

def correlationcal (data, domain):
    client_number = len(data)
    N = multiprocessing.cpu_count()
    with multiprocessing.Pool(N-1) as pool:
        correlationresult = list(ar.get() for ar in [pool.apply_async(newcorrcal, args=(data[client], domain)) for client in range(client_number)])
        pool.close()
        pool.join()
    
    InterMICorrdegree = list(map(lambda x : x[0], correlationresult))
    NextMICorrdegree  = list(map(lambda x : x[1], correlationresult))
    max_corr          = list(map(lambda x : x[2], correlationresult))

    return InterMICorrdegree, NextMICorrdegree, max_corr

def privetise(data, epsilon, domain, maxcorr = None, correlation=None, days = maxdays, allcorr = None):
    client_number = len(data)
    allprivetised = []
    for client in range(client_number):
        clientrawdata = data[client][:days]
        if correlation:
            _client = Client(epsilon, domain, correlation, maxcorr=maxcorr[client], allcorr=allcorr)
        else:
            _client = Client(epsilon, domain)
        privetised = _client.privetise(clientrawdata)
        allprivetised.append(privetised)
    return allprivetised

# Calling another functions and save correlation results in CSV files
def rollout (rawdata, day, epsilon, allcorr, rawmaxcorr):
    rawdata_day = list(map(lambda x: x[:day], rawdata ))
    newprivdata = privetise(rawdata, epsilon, domain, maxcorr=rawmaxcorr, correlation=True, days=day, allcorr=allcorr)

    newcorr = _corrcal(newprivdata, domain, True, days=day)
    newmaxcorr = mean(newcorr[2].values())

    rawfrequency, perturbedfrequency = Server().frequency(rawdata_day, newprivdata, epsilon, domain, maxcorr=newmaxcorr)

    return rawfrequency, perturbedfrequency

def main():
    rawfreq = list()
    perturbedfreq = list()
    ###########################################RawSection######################################################
    rawdata = list(map(lambda x : [eval(i) for i in x ], pd.read_csv('normal.csv').to_numpy().tolist()))
    rawcorr = correlationcal(rawdata, domain)

    rawintercorr = list(map(lambda x: [item[1] for item in x], rawcorr[0]))
    rawintercorr_pd = pd.DataFrame(rawintercorr).transpose()
    rawintercorr_pd.index = list(map(lambda x: [item[0] for item in x], rawcorr[0]))[0]

    rawintercorrdata = list(map(lambda x : rawintercorr_pd.index[rawintercorr_pd[x] >= fixthreshold].tolist() ,rawintercorr_pd))
    rawintercorrdata = list(map(lambda x : list(item[0] for item in x), rawintercorrdata))
    rawintercorrdata_list = list(itertools.chain(*rawintercorrdata))

    rawnextdaycorr = list(map(lambda x: [item[1] for item in x], rawcorr[1]))
    rawnextdaycorr_pd = pd.DataFrame(rawnextdaycorr).transpose()
    rawnextdaycorr_pd.index = list(map(lambda x: [item[0] for item in x], rawcorr[1]))[0]

    rawnextcorrdata = list(map(lambda x : rawnextdaycorr_pd.index[rawnextdaycorr_pd[x] >= fixthreshold].tolist(),rawnextdaycorr_pd))
    rawnextcorrdata = list(map(lambda x : list(item[0] for item in x), rawnextcorrdata))
    rawnextcorrdata_list = list(itertools.chain(*rawnextcorrdata))

    allcordata = set(rawintercorrdata_list + rawnextcorrdata_list)

    ####################################################
    
    ##### Calculate Frequency for each epsilon #####
    N = multiprocessing.cpu_count()
    with multiprocessing.Pool(N-1) as pool:
        freq = list(ar.get() for ar in [pool.apply_async(rollout, args=(rawdata, maxdays, epsilon, allcordata, rawcorr[2])) for cal in range(calperpoint)])
        rawfreq = np.array(list((map(lambda x : x[0], freq))))
        rawfreqmean = np.mean(rawfreq, axis=0)
        perturbedfreq = np.array(list(map(lambda x : x[1], freq)))
        perturbedfreqmean = np.mean(perturbedfreq, axis=0)
        pool.close()
        pool.join()

    ##### Plot figures and save results in csv #####
    x_axis = np.arange(len(rawfreqmean))
    plt.bar(x_axis - 0.25, rawfreqmean, label='Real Freq', width=0.5)
    plt.bar(x_axis + 0.25, perturbedfreqmean, label='Est Freq', width=0.5)
    plt.xlabel('Domain')
    plt.ylabel('Freq')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
    plt.show()


if __name__ == "__main__":
    main()