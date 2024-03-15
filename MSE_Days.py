from numpy import shape, where
import itertools
from minepy import MINE
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from statistics import mean
import matplotlib.pyplot as plt
from time import time
from datetime import datetime, timedelta
import math
from sklearn.metrics import mean_squared_error

fixepsilon = 1
maxepsilon = 5
fixthreshold = 0.8
maxthreshold = 1
calperpoint = 50
domain = 20
locperday = 10
client_number = 10000
maxdays = 30

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

class MSE:
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

    def encode(self, data, domain):
        days = len(data)
        encodelst = list()
        for day in range(days):
            lst = np.zeros(domain, int)
            lst1 = list(map(lambda x : np.put(lst, x, 1), data[day]))
            encodelst.append(list(lst))
        return encodelst

    def mse(self, data, privetised, epsilon, domain, grr=False, new=False, maxcorr=None):
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

        if grr:
            totalprivetisedre_fquency = self.grr_frequency_estimation(privetisedsum, epsilon, domain)
        elif new:
            totalprivetisedre_fquency = self.new_frequency_estimation(privetisedsum, epsilon, domain, maxcorr)
        
        Mean_square_error = mean_squared_error(totalprivetisedre_fquency, totalraw_frequency)
        return Mean_square_error

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
            IntraMICorrdegree = clientdatacorr.IntraCorrCal()

        if maxcorr:
            inter = list(item[1] for item in InterMICorrdegree[client])
            next = list(item[1] for item in NextMICorrdegree[client])
            intra = list(itertools.chain(*IntraMICorrdegree))
            max_corr[client] = max(inter + next + intra)

    return InterMICorrdegree, NextMICorrdegree, max_corr

def newcorrcal(data, domain):
    clientdata = data
    clientdatacorr = CorrelationCal(np.array(clientdata))
    InterMICorrdegree  = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
    NextMICorrdegree = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))
    IntraMICorrdegree = clientdatacorr.IntraCorrCal()
    inter = list(item[1] for item in InterMICorrdegree)
    next = list(item[1] for item in NextMICorrdegree) 
    intra = list(itertools.chain(*IntraMICorrdegree))
    max_corr = max(inter + next + intra)
    return InterMICorrdegree, NextMICorrdegree, max_corr

def correlationcal (data, domain):
    client_number = len(data)
    N = multiprocessing.cpu_count()
    with multiprocessing.Pool(N) as pool:
        correlationresult = list(ar.get() for ar in [pool.apply_async(newcorrcal, args=(data[client], domain)) for client in range(client_number)])
        pool.close()
        pool.join()
    
    InterMICorrdegree = list(map(lambda x : x[0], correlationresult))
    NextMICorrdegree  = list(map(lambda x : x[1], correlationresult))
    max_corr          = list(map(lambda x : x[2], correlationresult))

    return InterMICorrdegree, NextMICorrdegree, max_corr

# Calls RRClient class and privetise raw data based on GRR and proposed method
def privetise(data, epsilon, domain, maxcorr = None, correlation=None, days = maxdays, allcorr = None):
    client_number = len(data)
    allprivetised = []
    for client in range(client_number):
        clientrawdata = data[client][:days]
        if correlation:
            _client = RRClient(epsilon, domain, correlation, maxcorr=maxcorr[client], allcorr=allcorr)
        else:
            _client = RRClient(epsilon, domain)
        privetised = _client.privetise(clientrawdata)
        allprivetised.append(privetised)
    return allprivetised

# Calling another functions and save correlation results in CSV files
def rollout (rawdata, day, epsilon, allcorr, rawmaxcorr):
    rawdata_day = list(map(lambda x: x[:day], rawdata ))
    grrprivedata = privetise(rawdata, epsilon, domain, days=day)
    newprivdata = privetise(rawdata, epsilon, domain, maxcorr=rawmaxcorr, correlation=True, days=day, allcorr=allcorr)

    newcorr = _corrcal(newprivdata, domain, True, days=day)
    newmaxcorr = mean(newcorr[2].values())

    grrmse = MSE().mse(rawdata_day, grrprivedata, epsilon, domain, grr=True)
    newmse = MSE().mse(rawdata_day, newprivdata, epsilon, domain, new=True, maxcorr=newmaxcorr)

    return grrmse, newmse

def main():
    grrmsedays = list()
    newmsedays = list()
    days = np.arange(1, maxdays+1, 1)

    ###########################################RawSection######################################################
    r_start_time = time()
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

    print (f'Raw calculation last for {timedelta(seconds=(time() - r_start_time))}')
    ####################################################
    
    N = multiprocessing.cpu_count()

    ##### Calculate MSE calperpoint times for each day #####
    msedays_start_time = time()
    print (f'\nMSE Calculation with varying day started at {datetime.fromtimestamp(msedays_start_time)}')
    for day in days:
        mseday_start_time = time()
        with multiprocessing.Pool(N) as pool:
            msedays = list(ar.get() for ar in [pool.apply_async(rollout, args=(rawdata, day, fixepsilon, allcordata, rawcorr[2])) for cal in range(calperpoint)])
            grr_mse_mean = mean(map(lambda x : x[0], msedays))
            new_mse_mean = mean(map(lambda x : x[1], msedays))
            grrmsedays.append(grr_mse_mean)
            newmsedays.append(new_mse_mean)
            pool.close()
            pool.join()
        print (f'MSE Calculation for day {day} last for {timedelta(seconds=(time() - mseday_start_time))}', flush=True)   
    print (f'MSE Calculation with varying day last for {timedelta(seconds=(time() - msedays_start_time))}', flush=True)
    
    
    ##### Plot figures and save results in csv #####
    plt.plot(days, grrmsedays, label='GRR MSE')
    plt.plot(days, newmsedays, label='Proposed Method MSE')
    plt.xlabel('Days')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('daynormal.png')
    plt.close()

    df1 = pd.DataFrame(list(zip(days, grrmsedays, newmsedays)), columns = ['Days','GRR MSE', 'Proposed Method MSE'])
    df1.to_csv('daynormal.csv', index=False)



if __name__ == "__main__":
    start_time = time()
    main()
    print (f'\nTotal excecution last for {timedelta(seconds=(time() - start_time))}')