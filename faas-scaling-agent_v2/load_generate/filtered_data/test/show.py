import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

data_dir='.'

files=os.listdir(data_dir)

step_interval=60

for file_name in files:
    if not file_name.endswith('csv'):
        continue    
    data=pd.read_csv('{}/{}'.format(data_dir,file_name),header=None,sep=' ')
    data=str.split(data[1][0],',')
   
    data_int=[int(i)/60 for i in data]
    plt.figure()
    plt.bar([i * step_interval for i in range(120)], data_int[:120], label='value of rps', width=step_interval-2)

    plt.tick_params(labelsize=10)
    font = {
        'size': 10
    }
    plt.xlabel('Seconds', fontdict=font)
    leg = plt.legend(prop=font)
    leg.set_draggable(state=True)
    plt.savefig(file_name+'.png',bbox_inches='tight')
    print(np.mean(data_int[:120]))
    print(np.std(data_int[:120])/np.mean(data_int[:120]))
