from __future__ import division
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from server import Server, SCorrelationCal
from client import Client, CorrelationCal

epsilon = 1
fixthreshold = 0.8
domain = 20
client_number = 100
maxdays = 30

######################Calculate the real frquency######################
def calrealfreq(data):
    client_number = len(data)
    clientrawdata_encode = []    
    rawsum = []

    for client in range(client_number):
        clientrawdata = data[client]
        clientrawdata_encode.append(encode(clientrawdata, domain))
        rawsum.append([sum(x) for x in zip(*clientrawdata_encode[client])])
        
    totalraw_frequency = frequency_calculation(rawsum)
    
    return totalraw_frequency

def encode(data, domain):
        days = len(data)
        encodelst = list()
        for day in range(days):
            lst = np.zeros(domain, int)
            lst1 = list(map(lambda x : np.put(lst, x, 1), data[day]))
            encodelst.append(list(lst))
        return encodelst

def frequency_calculation(reports):
            if reports is not None:
                summation_data = [sum(x) for x in zip(*reports)]
                norm_est_freq = [np.nan_to_num(e / sum(summation_data)) for e in summation_data]
            
            return norm_est_freq

##################################################################
def main():

    rawdata = list(map(lambda x : [eval(i) for i in x ], pd.read_csv('normal.csv').to_numpy().tolist()))

    allprivetised = []
    allmaxcorr_pr = []
    # For each client calculate the correlation and perturbe data
    for client in range(client_number):
        # Client side
        clientdata = rawdata[client]
        clientdatacorr = CorrelationCal(np.array(clientdata), fixthreshold)
        InterMICorrdegree_client = clientdatacorr.InterCorrCal(list(np.arange(0, domain)))
        NextMICorrdegree_client = clientdatacorr.NextDayCorrCal(list(np.arange(0, domain)))
        IntraMICorrdegree_client = clientdatacorr.IntraCorrCal()
        inter_client = list(item[1] for item in InterMICorrdegree_client)
        next_client = list(item[1] for item in NextMICorrdegree_client) 
        intra_client_all = list(itertools.chain(*IntraMICorrdegree_client))
        
        intra_client = [x for x in intra_client_all if x >= fixthreshold]
        rawintercorrdata_list = list(item[0] for item in InterMICorrdegree_client)
        rawnextcorrdata_list = list(item[0] for item in NextMICorrdegree_client)

        if inter_client or next_client:
            max_corr_client = max(inter_client + next_client)
        else:
             max_corr_client = 0

        allcordata = set(rawintercorrdata_list + rawnextcorrdata_list)

        _client = Client(epsilon, domain, True, max_corr_client, allcordata)
        privetised = _client.privetise(clientdata)
        allprivetised.append(privetised)

        # Server side
        clientdatacorr_pr = SCorrelationCal(np.array(privetised), fixthreshold)
        InterMICorrdegree_client_pr = clientdatacorr_pr.InterCorrCal(list(np.arange(0, domain)))
        NextMICorrdegree_client_pr = clientdatacorr_pr.NextDayCorrCal(list(np.arange(0, domain)))
        IntraMICorrdegree_client_pr = clientdatacorr_pr.IntraCorrCal()
        inter_client_pr = list(item[1] for item in InterMICorrdegree_client_pr)
        next_client_pr = list(item[1] for item in NextMICorrdegree_client_pr)
        intra_client_all_pr = list(itertools.chain(*IntraMICorrdegree_client_pr))
        intra_client_pr = [x for x in intra_client_all_pr if x > fixthreshold]
        if inter_client_pr or next_client_pr:
            max_corr_client_pr = max(inter_client_pr + next_client_pr)
        else:
            max_corr_client_pr = 0
        
        allmaxcorr_pr.append(max_corr_client_pr)
    
    # Find the real data and perturbed data frequency
    rawfrequency = calrealfreq(rawdata)  
    perturbedfrequency = Server().frequency(allprivetised, epsilon, domain, allmaxcorr_pr)

    ##### Plot figures #####
    x_axis = np.arange(len(rawfrequency))
    plt.bar(x_axis - 0.25, rawfrequency, label='Real Freq', width=0.5)
    plt.bar(x_axis + 0.25, perturbedfrequency, label='Est Freq', width=0.5)
    plt.xlabel('Domain')
    plt.ylabel('Freq')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
    plt.show()

if __name__ == "__main__":
    main()