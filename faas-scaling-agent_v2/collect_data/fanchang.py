import os
import json
import csv
import pandas as pd

data = pd.read_csv('{}'.format("function_profile_table.csv"), header=0, sep=',')
function_profiler = {}
        
for instance_num, rps, cpu, mem in zip(data['instance_num'], data['rps'], data['cpu'], data['mem']):
	if rps in function_profiler.keys():
#cpu*instance_num>function_profiler[rps][1] and mem*instance_num>function_profiler[rps][2]
		if  cpu*instance_num>function_profiler[rps][1] and mem*instance_num>function_profiler[rps][2]:
			function_profiler[rps]= (instance_num,cpu*instance_num, mem*instance_num)
	else:
		function_profiler[rps] = (instance_num,cpu*instance_num,mem*instance_num)
               
re=[] 
for rps in function_profiler:
	if function_profiler[rps][0]!=5:
		print("{}:{}\n".format(rps,function_profiler[rps][0]))
		re.append(rps)

data = pd.read_csv('{}'.format("function_profile_table.csv"), header=0, sep=',')

for instance_num, rps, cpu, mem, vio in zip(data['instance_num'], data['rps'], data['cpu'], data['mem'], data['avg_vio_rate']):
	if rps ==14 :
		print('{}:{}:{}:{}:{}\n'.format(rps,instance_num,instance_num*cpu,instance_num*mem,vio))                

data = pd.read_csv('{}'.format("function_profile_table.csv"), header=0, sep=',')

for instance_num, rps, cpu, mem, vio in zip(data['instance_num'], data['rps'], data['cpu'], data['mem'], data['avg_vio_rate']):
	if rps==19:
		print('{}:{}:{}:{}:{}\n'.format(rps,instance_num,instance_num*cpu,instance_num*mem,vio))     

