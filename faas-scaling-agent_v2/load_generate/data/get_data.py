import pandas as pd
import os

save_path='f_data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

data_dir='MSRTQps'
file_names=os.listdir(data_dir)
file_names.sort()

function_p={}

for file_name in file_names:
    if file_name.endswith('.csv'):
        data=pd.read_csv('{}/{}'.format(data_dir,file_name),header=0,sep=',')
        for timestamp,msname,metric,value in zip(data['timestamp'],data['msname'],data['metric'],data['value']):
                if metric=='providerRPC_MCR':
                    if msname in function_p.keys():
                        try:
                            function_p[msname][int(timestamp/60000)]=function_p[msname][int(timestamp/60000)]+int(value*60)
                        except :
                            pass
                    else:
                        function_p[msname]=[0]*12*60
                        try:
                            function_p[msname][int(timestamp/60000)]=int(value*60)
                        except :
                            pass
                            #print(timestamp/60000)

for msname in function_p.keys():
    if os.path.exists('{}/{}'.format(save_path,msname)+'.csv'):
        data=pd.read_csv('{}/{}'.format(save_path,msname)+'.csv',header=None,sep=' ')
        data=str.split(data[1][0],',')
        data=[int(i) for i in data]
        c=[i+j for i,j in zip(data,function_p[msname])]
    else:
        c=function_p[msname]
    with open('{}/{}'.format(save_path,msname)+'.csv','w') as f:
        s=msname+' '+','.join('%s' %id for id in c)
        f.write(s)
