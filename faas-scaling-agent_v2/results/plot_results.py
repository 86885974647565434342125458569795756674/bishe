import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import argparse

# extract the data for analysis from the saved function_status files
def get_data(path, function_name='hello-python', step_interval=15):
    files = os.listdir(path)
    files.sort()

    invocation_list, violation_list = [], []
    violation_rate_list, cpu_seconds_list ,mem_seconds_list= [], [], []
    instance_num_list, rps_list = [], []
    init_invocation, init_violation = -1, -1
    prev_invocation = -1
    total_cpu_seconds = 0
    total_mem_seconds = 0
    for file in files:
        if file.endswith('function_status.json'):
            with open('{}/{}'.format(path, file)) as f:
                status_arr = json.load(f)
                for status in status_arr:
                    if status['name'] == function_name:
                        if 'latencyVioCount' in status.keys():
                            violation = status['latencyVioCount']
                        else:
                            violation = 0

                        if 'invocationCount' in status.keys():
                            invocation = status['invocationCount']
                        else:
                            invocation = 0

                        if init_invocation == -1:
                            # 初始值
                            init_invocation = invocation
                            init_violation = violation

                        if invocation == init_invocation:
                            # 初始为0
                            violation_rate_list.append(0)
                        else:
                            violation_rate_list.append((violation-init_violation)/(invocation-init_invocation))

                        if 'usage' in status.keys() and 'cpu' in status['usage'].keys():
                            total_cpu_seconds += status['usage']['cpu'] * step_interval
                        
                        if 'usage' in status.keys() and 'totalMemoryBytes' in status['usage'].keys():
                            total_mem_seconds += status['usage']['totalMemoryBytes'] * step_interval

                        if prev_invocation == -1:
                            # 初始为0
                            rps_list.append(0)
                        else:
                            rps_list.append((invocation - prev_invocation) / step_interval)

                        prev_invocation = invocation
                        cpu_seconds_list.append(total_cpu_seconds)
                        mem_seconds_list.append(total_mem_seconds)
                        invocation_list.append(invocation-init_invocation)
                        violation_list.append(violation-init_violation)
                        instance_num_list.append(status['availableReplicas'])
                        break
    return violation_rate_list, cpu_seconds_list, mem_seconds_list, invocation_list, rps_list, instance_num_list


def plot_instance_num_vs_rps(path, name, function_name='hello-python', step_interval=15, clip=None):
    plt.figure()
    _, _, _, _, rps_list, instance_num_list = get_data(path, function_name, step_interval)
    if clip is None:
        plt.bar([i * step_interval for i in range(len(instance_num_list))], instance_num_list, label='instance_num', width=step_interval-2)
        plt.bar([i * step_interval for i in range(len(rps_list))], [-val for val in rps_list], label='negative value of rps', width=step_interval-2)
    else:
        plt.bar([i * step_interval for i in range(clip)], instance_num_list[:clip], label='instance_num', width=step_interval-2)
        plt.bar([i * step_interval for i in range(clip)], [-val for val in rps_list[:clip]], label='negative value of rps', width=step_interval-2)

    plt.tick_params(labelsize=10)
    font = {
        'size': 10
    }
    plt.xlabel('Seconds', fontdict=font)
    leg = plt.legend(prop=font)
    leg.set_draggable(state=True)
    plt.savefig(name,bbox_inches='tight')

def plot_violation_rate(paths, labels, name, function_name='hello-python', step_interval=15):
    plt.figure()
    plt.ylim(0, 0.025)
    for path, label in zip(paths, labels):
        violation_rate_list, _, _, _, _, _ = get_data(path, function_name, step_interval)
        plt.plot([i*step_interval for i in range(len(violation_rate_list))], violation_rate_list, label=label)
    plt.tick_params(labelsize=10)
    font = {
        'size': 10
    }
    plt.xlabel('Seconds', fontdict=font)
    plt.ylabel('Average Violation Rate', fontdict=font)
    leg = plt.legend(prop=font)
    leg.set_draggable(state=True)
    plt.grid()
    plt.savefig(name,bbox_inches='tight')

def plot_cpu_second(paths, labels, name, function_name='hello-python', step_interval=15):
    plt.figure()
    for path, label in zip(paths, labels):
        _, cpu_seconds_list, _, invocation_list, _, _ = get_data(path, function_name, step_interval)
        plt.plot(invocation_list, cpu_seconds_list, label=label)
    plt.tick_params(labelsize=10)
    font = {
        'size': 10
    }
    plt.xlabel('Invocation Number', fontdict=font)
    plt.ylabel('Total CPU Seconds', fontdict=font)
    leg = plt.legend(prop=font)
    leg.set_draggable(state=True)
    plt.grid()
    plt.savefig(name,bbox_inches='tight')

def plot_mem_second(paths, labels, name, function_name='hello-python', step_interval=15):
    plt.figure()
    for path, label in zip(paths, labels):
        _, _, mem_seconds_list, invocation_list, _, _ = get_data(path, function_name, step_interval)
        plt.plot(invocation_list, mem_seconds_list, label=label)
    plt.tick_params(labelsize=10)
    font = {
        'size': 10
    }
    plt.xlabel('Invocation Number', fontdict=font)
    plt.ylabel('Total MEMORY Seconds', fontdict=font)
    leg = plt.legend(prop=font)
    leg.set_draggable(state=True)
    plt.grid()
    plt.savefig(name,bbox_inches='tight')

def plot_instance(paths, labels, name, function_name='hello-python', step_interval=15):
    for path, label in zip(paths, labels):
        plt.figure()
        _, _, _, _, _, instance_num_list = get_data(path, function_name, step_interval)
        #plt.plot(invocation_list,instance_num_list, label=label)
        plt.bar([i * step_interval for i in range(len(instance_num_list))], instance_num_list, width=step_interval-2,label=label)
        #plt.bar([i * step_interval for i in range(70)], instance_num_list[:70], width=step_interval-2,label=label)
        plt.tick_params(labelsize=10)
        font = {'size': 10}
        plt.xlabel('Seconds', fontdict=font)
        plt.ylabel('Instance Number', fontdict=font)
        leg = plt.legend(prop=font)
        leg.set_draggable(state=True)
        plt.grid()
        plt.savefig(name+'/'+label+'_instance_nums.png',bbox_inches='tight')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-img-dir', type=str, default='img')
    parser.add_argument('--data-path', type=str, default='data')
    args = parser.parse_known_args()[0]

    if not os.path.exists(args.save_img_dir):
        os.makedirs(args.save_img_dir)
    
    plot_instance_num_vs_rps('../rl-agent/{}/rl'.format(args.data_path),args.save_img_dir+'/instance_vs_rps.png')
    plot_instance_num_vs_rps('../rl-agent/{}/rl'.format(args.data_path),args.save_img_dir+'/instance_vs_rps_clip.png', clip=240)
    

 
    paths=['../rl-agent/{}/rl'.format(args.data_path), '../rl-agent/{}/qps-2'.format(args.data_path), '../rl-agent/{}/qps-3'.format(args.data_path), '../rl-agent/{}/qps-4'.format(args.data_path), '../rl-agent/{}/qps-5'.format(args.data_path), '../rl-agent/{}/vps-0'.format(args.data_path), '../rl-agent/{}/vps-1'.format(args.data_path)]
    labels=['RL', 'OpenFaaS-QPS>2', 'OpenFaaS-QPS>3', 'OpenFaaS-QPS>4', 'OpenFaaS-QPS>5', 'OpenFaaS-VPS>0','OpenFaaS-VPS>1']
    plot_violation_rate(paths, labels,args.save_img_dir+'/vios.png')
    plot_cpu_second(paths, labels,args.save_img_dir+'/cpus.png')
    plot_mem_second(paths, labels,args.save_img_dir+'/memorys.png')
    plot_instance(paths, labels,args.save_img_dir)

    print('algorithm average_violation_rate normalized_cpu_usage normalized_mem_usage')
    for path, label in zip(paths, labels):
        violation_rate_list, cpu_seconds_list, mem_seconds_list, invocation_list, _, _ = get_data(path)
        print(label, violation_rate_list[-1], cpu_seconds_list[-1]/invocation_list[-1], mem_seconds_list[-1]/invocation_list[-1])

