import numpy as np
import pandas as pd

mm=[]
data=pd.read_csv('262.csv',header=None,sep=' ')
for dd in data[1]:
    data=str.split(dd,',')
    data_int=[int(i) for i in data]
    print(np.mean(data_int))
    mm.append(np.mean(data_int))
print(np.average(mm))
