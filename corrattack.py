from __future__ import division
from minepy import MINE
import numpy as np
import pandas as pd
from time import time, sleep
from datetime import datetime, timedelta
import multiprocessing
import math
import itertools
import matplotlib.pyplot as plt
import datagenerator

domain = datagenerator.domain_size
locperday = datagenerator.locperday
maxdays = datagenerator.days

fixthreshold = 0.8  # A data considered to be correlated if its correlation exceeds The threshold
calperpoint = 50
maxepsilon = 5
gamma = 0.1
theta = 0.02

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
            self.maxcorr = self.maxcorr*10
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
                score = 0
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

class corrattack:
    def __init__(self, data, domain, index):
        self.data = data
        self.domain = domain
        self.index = index
        self.rows = data.shape[0]
        self.maxcorr = None
        self.p = None
        self.q = None
    
    def corrmatrix(self):
        indexs= np.arange(0, self.index, 1)
        locations = np.arange(0, self.domain, 1)
        probmatrix = dict() #Stores conditional probability: Pr(x_i = alpha | x_k = betha) where x_i, x_k \in {locperday} and alpha,betha \in {Domain}
                            #The keys are : [i, k, alpha, betha] and values are conditional probability value
        for i in indexs:
            for k in indexs:
                if k != i:
                    for alpha in locations:
                        for betha in locations:
                            alpha_betha = self.data.apply(lambda x : True if x[i]==alpha and x[k]==betha else False)   #Get a bool series representing which row satisfies the condition i.e. True for rows which location_i==alpha nad location_k==betha
                            num_alpha_betha = len(alpha_betha[alpha_betha==True].index) # Find the number of occurance that alpha in location i and betha in location k exist simultaneously

                            bethas = self.data.apply(lambda x : True if x[k]==betha else False)
                            num_bethas = len(bethas[bethas==True].index)

                            index = [i, k, alpha, betha]
                            if num_bethas:
                                p = num_alpha_betha / num_bethas # Conditional probability : p(x_i = alpha | x_k = betha)
                                probmatrix[str(index)] = p
                            else:
                                probmatrix[str(index)] = 0
                else:
                    pass
        return probmatrix
    
    def updateprob (self, epsilon, maxcorr, privetdata, probmatrix):
        const = math.pow(math.e, epsilon) + domain-1 + maxcorr
        self.p = math.pow(math.e, epsilon) / const
        self.q = (maxcorr + 1) / const
        default_noiseprob = np.full((domain, domain), self.q)
        for i in range(default_noiseprob.shape[0]):
            default_noiseprob[i,i] = self.p
        
        sumcorr = np.zeros((locperday, domain)) # Stores the number of noncorrelated values of domain per location.
        for i in range (locperday):
            for alpha in range (domain):
                for j in range (len(privetdata[0])):
                    try:
                        index = str([i, j, alpha, privetdata[0][j]])
                        prob = probmatrix[index]
                        if prob < theta:
                            sumcorr[i, alpha] += 1
                    except Exception:
                        pass
        sumcorr = np.where(sumcorr >= (gamma*locperday), 0, sumcorr) # Set corresponding value/location to zero which has number of noncorrelated less than gamma*domain

        return default_noiseprob, sumcorr
    
    def attackerprob(self, noiseprob, sumcorr, privetdata):
        attackerprobdays = dict()
        for day in range (len(privetdata)):
            attacker_prob = np.zeros((locperday, domain))
            for i in range(sumcorr.shape[0]): # i is location index
                privedata = privetdata[day][i]
                noiseprobmatrix = noiseprob   # Assign noise prob matrix to location index, then update it based on correlation
                zeroindice = np.where(sumcorr[i] == 0)[0] # Find zero value indices which represent values in location i that has number of noncorrelated less than gamma*domain
                nonzeroindice = np.where(sumcorr[i] != 0)[0]
                if zeroindice.shape[0] > 0 and nonzeroindice.shape[0] > 0: # If all values are correlated or all values are not correlated don't update probability matrix of location i
                    for x in np.nditer(zeroindice):
                        noiseprobmatrix[x] = np.zeros(domain) #Zero means corresponding index(value) of location has 0 probability in point of attacker
                    colindex = 0
                    for col in noiseprobmatrix.T:
                        if col[colindex] == 0 :
                            qbar = 1 / nonzeroindice.shape[0]
                            for x in range (len(col)): #For each location value (Domain values) in column, if not equal to zero, set to q'
                                if col[x] != 0:
                                    col[x] = qbar
                        else:
                            pbar = (self.p)/(self.p + (nonzeroindice.shape[0] - 1)*self.q) 
                            qbar = (self.q)/(self.p + (nonzeroindice.shape[0] - 1)*self.q)
                            for x in range (len(col)):
                                if col[x] != 0:
                                    col[x] = qbar
                            col[colindex] = pbar
                        colindex =+1 # Trace the value in column that should be set to self.p
                else:
                    pass
                
                for j in range(sumcorr.shape[1]):
                    element = noiseprobmatrix[j,privedata]
                    attacker_prob[i,j] = element
            
            attackerprobdays[day] = attacker_prob
        
        return attackerprobdays

    def attackererror (self, probmatrix): # Compute the attacker error with distance
        erroralldays = dict()
        truelocation = self.data
        for day in range (len(probmatrix)):
            errorday = 0
            probmatrixday = probmatrix[day]
            truedaylocation = truelocation[day]
            for time in range(probmatrixday.shape[0]):
                for location in range(probmatrixday.shape[1]):
                    prob = probmatrixday[time, location]
                    truetimelocation = truedaylocation[time]
                    dist = abs((location-truetimelocation))
                    error = dist * prob
                    errorday += error
            erroralldays[day] = errorday/probmatrixday.shape[0]
        
        return erroralldays

def attackererror_grr (data, pertubdata, epsilon): # Compute the attacker error with distance
        erroralldays_grr = dict()
        for day in range (maxdays):
            errorday = 0
            const = math.pow(math.e, epsilon) + domain - 1
            p = math.pow(math.e, epsilon) / const
            q = 1 / const
            truedaylocation = data[day]
            pertubdaylocation = pertubdata[day]
            for time in range(locperday):
                for location in range(domain):
                    truetimelocation = truedaylocation[time]
                    pertubetimelocation = pertubdaylocation[time]
                    dist = abs((location-truetimelocation))
                    if location==pertubetimelocation:
                        prob = p
                    else:
                        prob = q
                    error = dist * prob
                    errorday += error
            erroralldays_grr[day] = errorday/locperday
        
        return erroralldays_grr

def attackererror_new (data, pertubdata, epsilon, maxcorr): # Compute the attacker error with distance
        erroralldays_new = dict()
        for day in range (maxdays):
            errorday = 0
            # maxcorr = maxcorr*10
            const = math.pow(math.e, epsilon) + domain-1 + maxcorr
            p = math.pow(math.e, epsilon) / const
            q = (maxcorr + 1) / const
            truedaylocation = data[day]
            pertubdaylocation = pertubdata[day]
            for time in range(locperday):
                for location in range(domain):
                    truetimelocation = truedaylocation[time]
                    pertubetimelocation = pertubdaylocation[time]
                    dist = abs((location-truetimelocation))
                    if location==pertubetimelocation:
                        prob = p
                    else:
                        prob = q
                    error = dist * prob
                    errorday += error
            erroralldays_new[day] = errorday/locperday
        
        return erroralldays_new

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

def _corrcal(data, domain):
    clientdata = data
    clientdatacorr = CorrelationCal(np.array(clientdata))
    InterMICorrdegree  = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
    NextMICorrdegree = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))
    IntraMICorrdegree = clientdatacorr.IntraCorrCal()

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

def estimerror (data, client, epsilon, privetized):
    attack = corrattack(data.iloc[client], domain, locperday)
    clientprobmatrix = attack.corrmatrix()
    maxcorr = max(clientprobmatrix.values())
    default_noiseprob, sumcorr = attack.updateprob(epsilon, maxcorr, privetized[client], clientprobmatrix)
    probperlocalldays = attack.attackerprob(default_noiseprob, sumcorr, privetized[client])
    attackererror = attack.attackererror(probperlocalldays)

    return attackererror
 
def rollout(cal, epsilon, rawdata, allcordata, maxcorr):
    #################### Raw Section ###################
    nstart = time()
    newprivdata = privetise(rawdata, epsilon, domain, maxcorr, True, maxdays, allcordata)
    grrprivedata = privetise(rawdata, epsilon, domain, days=maxdays)
    print (f'Privatization of round {cal} last for {timedelta(seconds=time() - nstart)}')
    ####################################################

    rawdata = pd.DataFrame(rawdata)
    clients = rawdata.shape[0]

    allattackererror_new = list()
    allattackererror_grr = list()

    allattackererror_new_befor = list()
    allattackererror_grr_befor = list()

    
    for client in range(clients):
        cstart = time()
        allattackererror_new.append(estimerror(rawdata, client, epsilon, newprivdata))
        allattackererror_grr.append(estimerror(rawdata, client, epsilon, grrprivedata))
        allattackererror_grr_befor.append(attackererror_grr(rawdata.iloc[client],grrprivedata[client], epsilon))
        allattackererror_new_befor.append(attackererror_new(rawdata.iloc[client],newprivdata[client], epsilon, maxcorr[client]))
        
        print (f'Error for client {client} and round {cal} last for {timedelta(seconds=time() - cstart)}')


    attackererror_new_df = pd.DataFrame(allattackererror_new)
    attackererror_grr_df = pd.DataFrame(allattackererror_grr)

    allattackererror_new_befor_df = pd.DataFrame(allattackererror_new_befor)
    allattackererror_grr_befor_df = pd.DataFrame(allattackererror_grr_befor)

    print (f'All error estimation for round {cal} last for {timedelta(seconds=time() - nstart)}')

    return attackererror_new_df, attackererror_grr_df, allattackererror_new_befor_df, allattackererror_grr_befor_df

                      
def main():
    r_start_time = time()

    ###########################################RawSection######################################################
    rawdata = list(map(lambda x : [eval(i) for i in x ], pd.read_csv('normal.csv').to_numpy().tolist()))
    rawcorr = correlationcal(rawdata, domain)

    rawintercorr = list(map(lambda x: [item[1] for item in x], rawcorr[0]))
    rawintercorr_pd = pd.DataFrame(rawintercorr).transpose()
    rawintercorr_pd.index = list(map(lambda x: [item[0] for item in x], rawcorr[0]))[0]

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
    ############################################################################################################

    epsilon_attacker_error_new = list()
    epsilon_attacker_error_new_all = list()

    epsilon_attacker_error_grr = list()

    epsilon_attacker_error_grr_all = list()

    epsilon_attacker_error_new_before = list()
    epsilon_attacker_error_grr_before = list()
    

    epsilon_attacker_error_new_all_before = list()
    epsilon_attacker_error_grr_all_before = list()

    epsilons = np.arange(0.5, maxepsilon+0.5, 0.5)

    columnindex = list()
    for epsilon in epsilons:
        e_start_time = time()

        N = multiprocessing.cpu_count()
        with multiprocessing.Pool(N) as pool:
            corrlist = list(ar.get() for ar in [pool.apply_async(rollout, args=(cal, epsilon, rawdata, allcordata, rawcorr[3]))for cal in range(calperpoint)])
            pool.close()
            pool.join()

        newlist_before = list(map(lambda x : x[0], corrlist))
        grrlist_before = list(map(lambda x : x[1], corrlist))
        newlist = list(map(lambda x : x[2], corrlist))
        grrlist = list(map(lambda x : x[3], corrlist))
        


        attackererrorconcat_new = pd.concat(newlist)
        by_row_index_new = attackererrorconcat_new.groupby(attackererrorconcat_new.index)
        newerror_mean = by_row_index_new.mean().transpose() # Mean value of each calculation result
        newerror_mean_aggregated = newerror_mean.mean(axis=1) #Set axixs=1 to calculate mean value of users data for each day
        newerror_mean_aggregated_all = newerror_mean_aggregated.mean()
        epsilon_attacker_error_new.append(newerror_mean_aggregated) # Append error per epsilon. For each epsilon error calculated for each day avraged over users
        epsilon_attacker_error_new_all.append(newerror_mean_aggregated_all)
        
        attackererrorconcat_grr = pd.concat(grrlist)
        by_row_index_grr = attackererrorconcat_grr.groupby(attackererrorconcat_grr.index)
        grrerror_mean = by_row_index_grr.mean().transpose()
        grrerror_mean_aggregated = grrerror_mean.mean(axis=1)
        grrerror_mean_aggregated_all = grrerror_mean_aggregated.mean()
        epsilon_attacker_error_grr.append(grrerror_mean_aggregated)
        epsilon_attacker_error_grr_all.append(grrerror_mean_aggregated_all)

        attackererrorconcat_new_before = pd.concat(newlist_before)
        by_row_index_new_before = attackererrorconcat_new_before.groupby(attackererrorconcat_new_before.index)
        newerror_mean_before = by_row_index_new_before.mean().transpose()
        newerror_mean_aggregated_before = newerror_mean_before.mean(axis=1)
        newerror_mean_aggregated_all_before = newerror_mean_aggregated_before.mean()
        epsilon_attacker_error_new_before.append(newerror_mean_aggregated_before)
        epsilon_attacker_error_new_all_before.append(newerror_mean_aggregated_all_before)
        
        attackererrorconcat_grr_before = pd.concat(grrlist_before)
        by_row_index_grr_before = attackererrorconcat_grr_before.groupby(attackererrorconcat_grr_before.index)
        grrerror_mean_before = by_row_index_grr_before.mean().transpose()
        grrerror_mean_aggregated_before = grrerror_mean_before.mean(axis=1)
        grrerror_mean_aggregated_all_before = grrerror_mean_aggregated_before.mean()
        epsilon_attacker_error_grr_before.append(grrerror_mean_aggregated_before)
        epsilon_attacker_error_grr_all_before.append(grrerror_mean_aggregated_all_before)
        
        columnindex.append(f'Epsilon = {epsilon}')
        
        print (f'Error Calculation with epsilon {epsilon} last for {timedelta(seconds=(time() - e_start_time))}')

    
    new_df = pd.concat(epsilon_attacker_error_new, axis=1)
    new_df.columns = columnindex
    new_df.index = np.arange(start=1, stop=new_df.shape[0]+1, step=1)

    grr_df = pd.concat(epsilon_attacker_error_grr, axis=1)
    grr_df.columns = columnindex
    grr_df.index = np.arange(start=1, stop=new_df.shape[0]+1, step=1)

    new_df_before = pd.concat(epsilon_attacker_error_new_before, axis=1)
    new_df_before.columns = columnindex
    new_df_before.index = np.arange(start=1, stop=new_df_before.shape[0]+1, step=1)

    grr_df_before = pd.concat(epsilon_attacker_error_grr_before, axis=1)
    grr_df_before.columns = columnindex
    grr_df_before.index = np.arange(start=1, stop=new_df.shape[0]+1, step=1)

    
    new_df.to_csv('new.csv')
    sleep(5)
    grr_df.to_csv('grr.csv')
    sleep(5)

    new_df_before.to_csv('new_before.csv')
    sleep(5)
    grr_df_before.to_csv('grr_before.csv')
    sleep(5)

    ax = new_df.plot(xlabel='Days', ylabel='Error', title='New Method')
    plt.legend()
    plt.savefig('new.png')
    plt.close()
    
    ax1 = grr_df.plot(xlabel='Days', ylabel='Error', title='grr Method')
    plt.legend()
    plt.savefig('grr.png')
    plt.close()

    ax2 = grr_df_before.plot(xlabel='Days', ylabel='Error', title='grr Method_before')
    plt.legend()
    plt.savefig('grr_before.png')
    plt.close()

    ax3 = new_df_before.plot(xlabel='Days', ylabel='Error', title='New Method_before')
    plt.legend()
    plt.savefig('new_before.png')
    plt.close()


    plt.plot(epsilons, epsilon_attacker_error_new_all, label='Attacker Error with New Method')
    plt.plot(epsilons, epsilon_attacker_error_grr_all, label='Attacker Error with GRR Method')
    plt.plot(epsilons, epsilon_attacker_error_grr_all_before, label='Attacker Error with GRR Method_before')
    plt.plot(epsilons, epsilon_attacker_error_new_all_before, label='Attacker Error with NEW Method_before')

    plt.xlabel('Epsilons')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('ErrorsAll.png')
    plt.close()
    


if __name__ == "__main__":
    start_time = time()
    print(f'Attack began at {datetime.now()}')
    main()
    print(f'Attack last for {timedelta(seconds=(time()-start_time))}') 