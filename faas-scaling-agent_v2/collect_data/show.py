import os
import json
import csv
import argparse

# calculate the results with the colected data to build function_profile_table
def calculate_data(data_dir, function_name):
    print('instance_num,rps,cpu,mem,avg_vio_rate')
    with open ('function_profile_table.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['instance_num','rps','cpu','mem','avg_vio_rate'])

    # data
    sub_dir_names = os.listdir(data_dir)
    sub_dir_names.sort()
    for sub_dir_name in sub_dir_names:
        instance_num, rps = sub_dir_name.split('-')
        cpu_lists, mem_lists, vio_rates = [], [], []
        init_vio_cnt, init_inv_cnt = -1, -1
        file_names = os.listdir('{}/{}'.format(data_dir, sub_dir_name))
        file_names.sort()
        # 有序
        for file_name in file_names:
            if file_name.endswith('function_status.json'):
                with open('{}/{}/{}'.format(data_dir, sub_dir_name, file_name), 'r') as f:
                    status_arr = json.load(f)
                    for status in status_arr:
                        if status['name'] == function_name:
                            if 'latencyVioCount' in status.keys():
                                violation_cnt = status['latencyVioCount']
                            else:
                                violation_cnt = 0

                            if 'invocationCount' in status.keys():
                                invocation_cnt = status['invocationCount']
                            else:
                                invocation_cnt = 0

                            if init_inv_cnt == -1:
                                # 初始数量
                                init_vio_cnt = violation_cnt
                                init_inv_cnt = invocation_cnt

                            # 没有变化
                            if invocation_cnt == init_inv_cnt:
                                vio_rates.append(0)
                            else:
                                # 违约增量/调用增量
                                # 每个文件的违约率
                                vio_rates.append((violation_cnt - init_vio_cnt) / (invocation_cnt - init_inv_cnt))

                            if 'usage' in status.keys() and 'cpu' in status['usage'].keys():
                                # 单个副本
                                cpu_lists.append(status['usage']['cpu']/status['availableReplicas'])
                            else:
                                cpu_lists.append(0)
                            
                            if 'usage' in status.keys() and 'totalMemoryBytes' in status['usage'].keys():
                                # 单个副本
                                mem_lists.append(status['usage']['totalMemoryBytes']/status['availableReplicas'])
                            else:
                                mem_lists.append(0)

        # 每个文件夹（实例数-rps）的平均违约率
        print('{},{},{},{},{}'.format(instance_num, rps, sum(cpu_lists)/len(cpu_lists), sum(mem_lists)/len(mem_lists), sum(vio_rates)/len(vio_rates)))
        with open ('function_profile_table.csv','a+') as f:
            writer = csv.writer(f)
            writer.writerow([instance_num,rps,sum(cpu_lists)/len(cpu_lists),sum(mem_lists)/len(mem_lists),sum(vio_rates)/len(vio_rates)])

    print(max(cpu_lists))
    print(max(mem_lists))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function-name', type=str, default='hello-python')
    parser.add_argument('--save-root', type=str, default='data', help='the path to save the collected data')
    args = parser.parse_known_args()[0]

    calculate_data(args.save_root, args.function_name)
