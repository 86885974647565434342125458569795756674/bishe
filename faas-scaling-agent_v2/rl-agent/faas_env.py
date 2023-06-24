import gym
import requests
import json
import numpy as np
from datetime import datetime
import os


class OpenFaaS(object):
    def __init__(self, args):
        """   environment related settings  """
        self.ip = args.ip
        self.port = args.port
        # 保存状态
        self.save_log_dir = args.save_log_dir
        # drl或报警
        self.use_drl = args.use_drl
        self.cpu_limit = args.cpu_limit
        self.mem_limit = args.mem_limit
        self.instance_num = 1
        self.max_instance_num = args.max_instance_num
        self.max_rps = args.max_rps
        self.function_name = args.function_name
        self.step_interval = args.step_interval
        self.prev_invocation, self.prev_violation = 0, 0
        if not os.path.exists(self.save_log_dir):
            os.makedirs(self.save_log_dir)

        """   drl related settings  """
        # the obervation is :
        # * the instance number
        # * rps, constrainted to [0, self.max_rps] then divided by max_rps
        # * cpu usage, divided by cpu limit defined by user
        # * mem usage, divided by mem limit defined by user
        # * violate rate in the previous step
        self.observation_high = np.array([self.max_instance_num, 1, 1, 1, 1],dtype=np.float32)
        self.observation_low = np.array([1, 0, 0, 0, 0],dtype=np.float32)
        self.observation_space = gym.spaces.Box(self.observation_low, self.observation_high, dtype=np.float32)

        # the action is:
        # * 0: scale down
        # * 1: do nothing
        # * 2: scale up + 1
        # * 3: scale up + 2
        # * 4: scale up + 3
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed=0):
        self.instance_num, self.prev_invocation, self.prev_violation, avg_cpu_usage, avg_mem_usage = self.fetch_data()
        return np.array([self.instance_num, 0, avg_cpu_usage/self.cpu_limit, avg_mem_usage/self.mem_limit, 0]), {}

    def step(self, act):
        self.instance_num, invocation, violation, avg_cpu_usage, avg_mem_usage = self.fetch_data()
        invocation_inc = invocation-self.prev_invocation
        violation_inc = violation-self.prev_violation

        vio_rate = 0
        if invocation_inc > 0:
            vio_rate = violation_inc/invocation_inc

        rps = min(self.max_rps, invocation_inc/self.step_interval)
        self.prev_invocation = invocation
        self.prev_violation = violation

        # real perform action
        prev_instance_num = self.instance_num
        if act == 0:
            self.instance_num = max(1, self.instance_num - 1)
        elif act == 2:
            self.instance_num = min(self.max_instance_num, self.instance_num + 1)
        elif act == 3:
            self.instance_num = min(self.max_instance_num, self.instance_num + 2)
        elif act == 4:
            self.instance_num = min(self.max_instance_num, self.instance_num + 3)
        if prev_instance_num != self.instance_num:
            self.scale()

        return np.array([self.instance_num, rps/self.max_rps, avg_cpu_usage/self.cpu_limit, avg_mem_usage/self.mem_limit, vio_rate]), 0, False, False, {}

    def scale(self):
        if self.use_drl:
            data = {
                'replicas': self.instance_num
            }
            requests.post('http://{}:{}/system/scale-function/{}?namespace=openfaas-fn'.format(self.ip, self.port, self.function_name), json=data, auth=('admin', 'admin'))

    def fetch_data(self):
        instance_num, invocation, violation, avg_cpu_usage, avg_mem_usage = 1, 0, 0, 0, 0
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        r = requests.get('http://{}:{}/system/functions'.format(self.ip, self.port), auth=('admin', 'admin'))
        with open('{}/{}_function_status.json'.format(self.save_log_dir, date_str), 'w+') as f:
            f.write(r.text)
        status_arr = json.loads(r.text)
        for status in status_arr:
            if status['name'] == self.function_name:
                if 'availableReplicas' in status.keys():
                    instance_num = status['availableReplicas']
                else:
                    instance_num = status['replicas']
                if 'invocationCount' in status.keys():
                    invocation = status['invocationCount']
                if 'latencyVioCount' in status.keys():
                    violation = status['latencyVioCount']
                if 'usage' in status.keys() and 'cpu' in status['usage'].keys():
                    avg_cpu_usage = status['usage']['cpu']/instance_num
                if 'usage' in status.keys() and 'totalMemoryBytes' in status['usage'].keys():
                    avg_mem_usage = status['usage']['totalMemoryBytes']/instance_num
                break
        return instance_num, invocation, violation, avg_cpu_usage, avg_mem_usage

