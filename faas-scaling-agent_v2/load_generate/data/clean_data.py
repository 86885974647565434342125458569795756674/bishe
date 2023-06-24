import pandas as pd
import os
import shutil
import numpy as np

data_dir='f_data'
save_dir='ff_data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files=os.listdir(data_dir)
for file_name in files:
    data=pd.read_csv('{}/{}'.format(data_dir,file_name),header=None,sep=' ')
    data=str.split(data[1][0],',')
    '''
    data_new=[]
    num=0
    for i in data:
        if i!='0':
            num=0
            data_new.append(i)
        elif num!=6:
            num=num+1
            data_new.append(i)
    '''
    data_int=[int(i) for i in data]
    if np.average(data_int)>20*60:
        continue
    i=0
    for x in data_int:
        if x==0:
            i=i+1
    if i>0.5*len(data_int):
        #print(i)
        continue
    if np.average(data_int)>5*60 and np.mean(data_int)>200:
        shutil.copy('{}/{}'.format(data_dir,file_name),save_dir)

