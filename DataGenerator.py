

import numpy as np
import pandas as pd


domain_size = 20
locperday = 10
client_number = 10000
days = 30

alldata = []
for i in range (client_number):
    clientdata =[]
    for day in range (days):
        data = np.random.normal(loc=10, scale=6, size=locperday)
        data[data > 19] = 19
        data[data < 0] = 0
        data = data.astype(int)
        clientdata.append((list(data)))
    alldata.append(clientdata)
df = pd.DataFrame(alldata)
df.to_csv('normal.csv', index=False)


