from __future__ import division
from statistics import mean
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from time import sleep, time
from datetime import datetime, timedelta
import DataGenerator
import math
from minepy import MINE

maxepsilon = 50     # The epsilon for perturbation
calperpoint = 20    # Calculation done per point
fixthreshold = 0.8

maxdays = DataGenerator.days
domain = DataGenerator.domain_size
locperday = DataGenerator.locperday
client_number = DataGenerator.client_number


##################### Perturbation Calculation Class#####################
class RRClient():
    def __init__(self, epsilon, d, corr=None, maxcorr=None, allcorr = None):
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
        
        if not self.corr:
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
################################################################################

######################### Correlation Calculation Class ########################
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
                totalpairs += np.shape(np.where(pairs in list(self.pairwise(list(self.data[x_index])))))[1]
            prob = totalpairs / totaltransition

            transitionMatrix.append([tuple(pairs), prob])
        return transitionMatrix
################################################################################        

# Calls CorrelationCal function from CorrCal file to calculate three types of correlation fr each user
def _corrcalall(data, domain):
    client_number = len(data)
    InterMICorrdegree = dict()
    NextMICorrdegree = dict()
    IntraMICorrdegree = dict()
    max_corr = dict()
    for client in range(client_number):
        clientdata = data[client]
        clientdatacorr = CorrelationCal(np.array(clientdata))
        IntraMICorrdegree[client] = clientdatacorr.IntraCorrCal()
        InterMICorrdegree[client]  = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
        NextMICorrdegree[client] = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))

        inter = list(item[1] for item in InterMICorrdegree[client])
        next = list(item[1] for item in NextMICorrdegree[client]) 
        intra = list(itertools.chain(*IntraMICorrdegree))
        max_corr[client] = max(inter + next + intra)

    return InterMICorrdegree, IntraMICorrdegree, NextMICorrdegree, max_corr

def _corrcal(data, domain):
    clientdata = data
    clientdatacorr = CorrelationCal(np.array(clientdata))
    IntraMICorrdegree = clientdatacorr.IntraCorrCal()
    InterMICorrdegree  = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
    NextMICorrdegree = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))
    inter = list(item[1] for item in InterMICorrdegree)
    next = list(item[1] for item in NextMICorrdegree) 
    intra = list(itertools.chain(*IntraMICorrdegree))
    
    max_corr = max(inter + next + intra)
    
    return InterMICorrdegree, IntraMICorrdegree, NextMICorrdegree, max_corr

def correlationcal (data, domain):
    client_number = len(data)
    N = multiprocessing.cpu_count()
    with multiprocessing.Pool(N) as pool:
        correlationresult = list(ar.get() for ar in [pool.apply_async(_corrcal, args=(data[client], domain)) for client in range(client_number)])
        pool.close()
        pool.join()
    
    InterMICorrdegree = list(map(lambda x : x[0], correlationresult))
    IntraMICorrdegree = list(map(lambda x : x[1], correlationresult))
    NextMICorrdegree  = list(map(lambda x : x[2], correlationresult))
    max_corr          = list(map(lambda x : x[3], correlationresult))

    return InterMICorrdegree, IntraMICorrdegree, NextMICorrdegree, max_corr


# Calls RRClient class and privetise raw data based on GRR and proposed method
def privetise(data, epsilon, domain, maxcorr = None, correlation=None, allcorr = None):
    client_number = len(data)
    allprivetised = []
    for client in range(client_number):
        clientrawdata = data[client]
        if correlation:
            _client = RRClient(epsilon, domain, correlation, maxcorr=maxcorr[client], allcorr=allcorr)
        else:
            _client = RRClient(epsilon, domain)
        privetised = _client.privetise(clientrawdata)
        allprivetised.append(privetised)
    return allprivetised

# Privetise data and call _corrcal function
def rollout (cal, rawdata, epsilon, allcorr, maxcorr):
    grrprivedata = privetise(rawdata, epsilon, domain)
    newprivdata = privetise(rawdata, epsilon, domain, maxcorr=maxcorr, correlation=True, allcorr=allcorr)
    
    g_start_time = time()
    grrcorr = _corrcalall(grrprivedata, domain)
    print (f'GRR correlation for cal {cal} last for {timedelta(seconds=(time() - g_start_time))}')
    
    n_start_time = time()
    newcorr = _corrcalall(newprivdata, domain)
    print (f'New correlation for cal {cal} last for {timedelta(seconds=(time() - n_start_time))}')

    return grrcorr, newcorr

def main():
    ###########################################RawSection######################################################
    rawdata = list(map(lambda x : [eval(i) for i in x ], pd.read_csv('uniform.csv').to_numpy().tolist()))
    rawcorr = correlationcal(rawdata, domain)

    rawintracorr = rawcorr[1]
    rawintracorr_pd = pd.DataFrame(rawintracorr)

    rawintercorr = list(map(lambda x: [item[1] for item in x], rawcorr[0]))
    rawintercorr_pd = pd.DataFrame(rawintercorr).transpose()
    rawintercorr_pd.index = list(map(lambda x: [item[0] for item in x], rawcorr[0]))[0]
    
    r_start_time = time()
    
    rawintercorrdata = list(map(lambda x : rawintercorr_pd.index[rawintercorr_pd[x] >= fixthreshold].tolist() ,rawintercorr_pd))
    rawintercorrdata = list(map(lambda x : list(item[0] for item in x), rawintercorrdata))
    rawintercorrdata_list = list(itertools.chain(*rawintercorrdata))

    rawnextdaycorr = list(map(lambda x: [item[1] for item in x], rawcorr[2]))
    rawnextdaycorr_pd = pd.DataFrame(rawnextdaycorr).transpose()
    rawnextdaycorr_pd.index = list(map(lambda x: [item[0] for item in x], rawcorr[2]))[0]

    rawnextcorrdata = list(map(lambda x : rawnextdaycorr_pd.index[rawnextdaycorr_pd[x] >= fixthreshold].tolist(),rawnextdaycorr_pd))
    rawnextcorrdata = list(map(lambda x : list(item[0] for item in x), rawnextcorrdata))
    rawnextcorrdata_list = list(itertools.chain(*rawnextcorrdata))

    allcordata = set(rawintercorrdata_list + rawnextcorrdata_list)

    print (f'Raw calculation last for {timedelta(seconds=(time() - r_start_time))}')
    ####################################################
    
    for fixepsilon in range (5, maxepsilon+5, 5):

        fixepsilon = fixepsilon/10

        ########## Correlation Calculation Section #########
        N = multiprocessing.cpu_count()
        with multiprocessing.Pool(N) as pool:
            correlation = list(ar.get() for ar in [pool.apply_async(rollout, args=(cal, rawdata, fixepsilon, allcordata, rawcorr[3])) for cal in range(calperpoint)])
            pool.close()
            pool.join()
        ####################################################
        
        #################### GRR Section ###################
        #### Calculate GRR Inter correlation value ####
        grrintercorr_all = list(map(lambda x : list(x[0][0].values()), correlation))
        grrintercorr_all_clip = list(map(lambda x: list(map(lambda y: [item[1] for item in y], x)) ,grrintercorr_all))
        grrinter_index_list = list(map(lambda x : x[0], grrintercorr_all[0][0]))
        grrintercorr_pd_all = list(map(lambda x : pd.DataFrame(x), grrintercorr_all_clip))
        grrintercorr_concat = pd.concat(grrintercorr_pd_all)
        by_row_index = grrintercorr_concat.groupby(grrintercorr_concat.index)
        
        grrintercorr_pd_max = by_row_index.max().transpose()
        grrintercorr_pd_max.index = grrinter_index_list

        grrintercorr_pd_min = by_row_index.min().transpose()
        grrintercorr_pd_min.index = grrinter_index_list

        grrintercorr_pd_mean = by_row_index.mean().transpose()
        grrintercorr_pd_mean.index = grrinter_index_list
        
        #### Calculate GRR Intra correlation value ####
        grrintracorr_all = list(map(lambda x : list(x[0][1].values()), correlation))
        grrintracorr_max = [
        [
            [
                max(sublist[i][j] for sublist in sublist_group) for j in range(len(sublist_group[0]))
            ] for i in range(len(sublist_group[0]))
        ] for sublist_group in zip(*grrintracorr_all)]
        
        grrintracorr_pd_max = pd.DataFrame(grrintracorr_max)

        grrintracorr_min = [
        [
            [
                min(sublist[i][j] for sublist in sublist_group) for j in range(len(sublist_group[0]))
            ] for i in range(len(sublist_group[0]))
        ] for sublist_group in zip(*grrintracorr_all)]
        
        grrintracorr_pd_min = pd.DataFrame(grrintracorr_min)

        grrintracorr_mean = [
        [
            [
                mean(sublist[i][j] for sublist in sublist_group) for j in range(len(sublist_group[0]))
            ] for i in range(len(sublist_group[0]))
        ] for sublist_group in zip(*grrintracorr_all)]
        
        grrintracorr_pd_mean = pd.DataFrame(grrintracorr_mean)

        #### Calculate GRR NextDay correlation value ####        
        grrnextcorr_all  = list(map(lambda x : list(x[0][2].values()), correlation))
        grrnextcorr_all_clip = list(map(lambda x: list(map(lambda y: [item[1] for item in y], x)) ,grrnextcorr_all))
        grrnext_index_list = list(map(lambda x : x[0], grrnextcorr_all[0][0]))
        grrnextcorr_pd_all = list(map(lambda x : pd.DataFrame(x), grrnextcorr_all_clip))
        grrnextcorr_concat = pd.concat(grrnextcorr_pd_all)
        by_row_index = grrnextcorr_concat.groupby(grrnextcorr_concat.index)
        
        grrnextdaycorr_pd_max = by_row_index.max().transpose()
        grrnextdaycorr_pd_max.index = grrnext_index_list

        grrnextdaycorr_pd_min = by_row_index.min().transpose()
        grrnextdaycorr_pd_min.index = grrnext_index_list

        grrnextdaycorr_pd_mean = by_row_index.mean().transpose()
        grrnextdaycorr_pd_mean.index = grrnext_index_list
        ####################################################

        #################### New Section ###################
        #### Calculate New Inter correlation value ####
        newintercorr_all = list(map(lambda x : x[1][0], correlation))
        newintercorr_all_clip = list(map(lambda x: list(map(lambda y: [item[1] for item in y], list(x.values()))) ,newintercorr_all))
        newinter_index_list = list(map(lambda x : x[0], newintercorr_all[0][0]))
        newintercorr_pd_all = list(map(lambda x : pd.DataFrame(x), newintercorr_all_clip))
        newintercorr_concat = pd.concat(newintercorr_pd_all)
        by_row_index = newintercorr_concat.groupby(newintercorr_concat.index)
        
        newintercorr_pd_max = by_row_index.max().transpose()
        newintercorr_pd_max.index = newinter_index_list

        newintercorr_pd_min = by_row_index.min().transpose()
        newintercorr_pd_min.index = newinter_index_list

        newintercorr_pd_mean = by_row_index.mean().transpose()
        newintercorr_pd_mean.index = newinter_index_list
        
        #### Calculate new Intra correlation value ####
        newintracorr_all = list(map(lambda x : list(x[1][1].values()), correlation))
        newintracorr_max = [
        [
            [
                max(sublist[i][j] for sublist in sublist_group) for j in range(len(sublist_group[0]))
            ] for i in range(len(sublist_group[0]))
        ] for sublist_group in zip(*newintracorr_all)]
        
        newintracorr_pd_max = pd.DataFrame(newintracorr_max)

        newintracorr_min = [
        [
            [
                min(sublist[i][j] for sublist in sublist_group) for j in range(len(sublist_group[0]))
            ] for i in range(len(sublist_group[0]))
        ] for sublist_group in zip(*newintracorr_all)]
        
        newintracorr_pd_min = pd.DataFrame(newintracorr_min)

        newintracorr_mean = [
        [
            [
                mean(sublist[i][j] for sublist in sublist_group) for j in range(len(sublist_group[0]))
            ] for i in range(len(sublist_group[0]))
        ] for sublist_group in zip(*newintracorr_all)]
        
        newintracorr_pd_mean = pd.DataFrame(newintracorr_mean)

        #### Calculate new NextDay correlation max value ####        
        newnextcorr_all  = list(map(lambda x : list(x[1][2].values()), correlation))
        newnextcorr_all_clip = list(map(lambda x: list(map(lambda y: [item[1] for item in y], x)) ,newnextcorr_all))
        newnext_index_list = list(map(lambda x : x[0], newnextcorr_all[0][0]))
        newnextcorr_pd_all = list(map(lambda x : pd.DataFrame(x), newnextcorr_all_clip))
        newnextcorr_concat = pd.concat(newnextcorr_pd_all)
        by_row_index = newnextcorr_concat.groupby(newnextcorr_concat.index)
        
        newnextdaycorr_pd_max = by_row_index.max().transpose()
        newnextdaycorr_pd_max.index = newnext_index_list

        newnextdaycorr_pd_min = by_row_index.min().transpose()
        newnextdaycorr_pd_min.index = newnext_index_list

        newnextdaycorr_pd_mean = by_row_index.mean().transpose()
        newnextdaycorr_pd_mean.index = newnext_index_list
        #########################################################################
        rawintercorr_pd.to_csv('RawInterCorr'+f'{fixepsilon}'+'.csv', index=False)
        rawintracorr_pd.to_csv('RawIntraCorr'+f'{fixepsilon}'+'.csv', index=False)
        rawnextdaycorr_pd.to_csv('RawNextCorr'+f'{fixepsilon}'+'.csv', index=False)

        ####################################max#####################################
        grrintercorr_pd_max.to_csv('GRRInterCorr_max_'+f'{fixepsilon}'+'.csv', index=False)
        newintercorr_pd_max.to_csv('NewInterCorr_max_'+f'{fixepsilon}'+'.csv', index=False)

        
        grrintracorr_pd_max.to_csv('GRRIntraCorr_max_'+f'{fixepsilon}'+'.csv', index=False)
        newintracorr_pd_max.to_csv('NewIntraCorr_max_'+f'{fixepsilon}'+'.csv', index=False)

        grrnextdaycorr_pd_max.to_csv('GRRNextCorr_max_'+f'{fixepsilon}'+'.csv', index=False)
        newnextdaycorr_pd_max.to_csv('NewNextCorr_max_'+f'{fixepsilon}'+'.csv', index=False)
        sleep(5)

        ####################################min#####################################
        grrintercorr_pd_min.to_csv('GRRInterCorr_min_'+f'{fixepsilon}'+'.csv', index=False)
        newintercorr_pd_min.to_csv('NewInterCorr_min_'+f'{fixepsilon}'+'.csv', index=False)

        
        grrintracorr_pd_min.to_csv('GRRIntraCorr_min_'+f'{fixepsilon}'+'.csv', index=False)
        newintracorr_pd_min.to_csv('NewIntraCorr_min_'+f'{fixepsilon}'+'.csv', index=False)

        grrnextdaycorr_pd_min.to_csv('GRRNextCorr_min_'+f'{fixepsilon}'+'.csv', index=False)
        newnextdaycorr_pd_min.to_csv('NewNextCorr_min_'+f'{fixepsilon}'+'.csv', index=False)
        sleep(5)

        ####################################mean#####################################
        grrintercorr_pd_mean.to_csv('GRRInterCorr_mean_'+f'{fixepsilon}'+'.csv', index=False)
        newintercorr_pd_mean.to_csv('NewInterCorr_mean_'+f'{fixepsilon}'+'.csv', index=False)

        
        grrintracorr_pd_mean.to_csv('GRRIntraCorr_mean_'+f'{fixepsilon}'+'.csv', index=False)
        newintracorr_pd_mean.to_csv('NewIntraCorr_mean_'+f'{fixepsilon}'+'.csv', index=False)

        grrnextdaycorr_pd_mean.to_csv('GRRNextCorr_mean_'+f'{fixepsilon}'+'.csv', index=False)
        newnextdaycorr_pd_mean.to_csv('NewNextCorr_mean_'+f'{fixepsilon}'+'.csv', index=False)
        sleep(5)

if __name__ == "__main__":
    start_time = time()
    print (f'\nExcecution for max Correlation started at {(datetime.now())}')
    main()
    print (f'\nTotal excecution last for {timedelta(seconds=(time() - start_time))}')