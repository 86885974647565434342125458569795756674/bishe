import os
import pandas as pd
import gym
import numpy as np
import random

class SimEnv():
    def __init__(self, args) -> None:
        """
        The class that defines the simulation environment for openfaas,
        we use SimEnv to speed our training
        :param args: Command line argument
        """

        """   environment related settings  """
        self.is_test = args.is_test
        self.cpu_limit = args.cpu_limit
        self.mem_limit = args.mem_limit
        self.instance_num = 1
        self.max_instance_num = args.max_instance_num
        self.max_rps = args.max_rps
        self.target_vio_rate = args.target_vio_rate
        # function_profile_table.csv (instance_num,qps,cpu,avg_vio_rate)
        self.function_profile_file = args.function_profile_file
        # function_profiler: [rps][instance_num] -> (cpu, vio_rate)
        self.function_profiler = self.init_profiler()
        # rps2instance_num: rps -> the expected instance number that can meet the latency slo
        self.rps2instance_num = self.init_rps2instance_num()
        self.step_interval = args.step_interval
        self.all_requests = self.init_requests(args.workload_data_path)

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
        self.max_step = args.max_step

        """   others  """
        self.request_i, self.request_j = 0, 0
        self.step_cnt = 0

    def reset(self, seed=0):
        """
        :return: observation, info
        """
        self.step_cnt = 0
        # In the test mode, we execute the request in the test request data from the beginning to the end;
        # In training mode, we randomly select a function request data and start at a random time point
        self.request_i = random.randint(0, len(self.all_requests)-1)
        if self.is_test:
            self.request_j = 0
        else:
            self.request_j = random.randint(1, len(self.all_requests[self.request_i])-self.max_step-1)
        
        # 初始实例数
        self.instance_num = 1
        
        # 初始请求
        request_num = self.all_requests[self.request_i][self.request_j-1]
        rps = min(self.max_rps, int(request_num/self.step_interval))
        
        # 初始状态
        cpu_usage, mem_usage, vio_rate, _ = self.get_estimated_status(rps, self.instance_num)
        
        return np.array([self.instance_num, rps/self.max_rps, cpu_usage/self.cpu_limit, mem_usage/self.mem_limit, vio_rate]), {}

    def step(self, act):
        """
        :return: obervation, reward, done, truncate, info
        """
        # 新请求
        request_num = self.all_requests[self.request_i][self.request_j]
        next_rps = min(self.max_rps, int(request_num/self.step_interval))
        prev_instance_num = self.instance_num

        # 新实例数
        if act == 0:
            self.instance_num = max(1, self.instance_num-1)
        elif act == 2:
            self.instance_num = min(self.max_instance_num, self.instance_num+1)
        elif act == 3:
            self.instance_num = min(self.max_instance_num, self.instance_num+2)
        elif act == 4:
            self.instance_num = min(self.max_instance_num, self.instance_num+3)

        # 新状态
        cpu_usage, mem_usage, vio_rate, expected_instance_num = self.get_estimated_status(next_rps, self.instance_num)
        info = {
            'prev_instance_num': prev_instance_num,
            'instance_num': self.instance_num,
            'rps': next_rps,
            'cpu': cpu_usage,
            'mem': mem_usage,
            'vio_rate': vio_rate,
        }

        # despite the next_rps can not be obtained when making action, we can also use it to calculate the reward
        # 期望==动作，奖励
        if expected_instance_num < prev_instance_num:
            r = (act == 0)
            info['expected_act'] = 0
        elif expected_instance_num > prev_instance_num:
            if (expected_instance_num-prev_instance_num) == 1:
                r = (act == 2)
                info['expected_act'] = 2
            elif (expected_instance_num - prev_instance_num) == 2:
                r = (act == 3)
                info['expected_act'] = 3
            else:
                r = (act == 4)
                info['expected_act'] = 4
        else:
            r = (act == 1)
            info['expected_act'] = 1

        self.request_j += 1
        self.step_cnt += 1

        # In the test mode, we execute the request in the test request data from the beginning to the end;
        # In training mode, we early truncated the trajectory for better training
        if self.is_test:
            done = (self.request_j == len(self.all_requests[self.request_i]))
        else:
            done = (self.step_cnt == self.max_step)

        s_ = np.array([self.instance_num, next_rps/self.max_rps, cpu_usage/self.cpu_limit, mem_usage/self.mem_limit, vio_rate])
        return s_, r, done, done, info

    def init_requests(self, path):
        all_requests = []
        files = os.listdir(path)
        # 一个函数多天组成一个文件
        for file in files:
            # 两列
            data = pd.read_csv('{}/{}'.format(path, file), header=None, sep=' ')
            function_requests = []
            # 分行（天）
            for val in data[1]:
                # 每一天
                str_arr = str.split(val, ',')
                # 分一分钟
                for s in str_arr:
                    qpi = int(int(s)/(60/self.step_interval))
                    for _ in range(int((60/self.step_interval))):
                        function_requests.append(qpi)
            all_requests.append(function_requests)
        return all_requests

    def init_profiler(self):
        """
            read the function_profile_file to build the map [rps][instance_num] -> (cpu, mem, vio_rate)
        """
        # 第0行是header,跳过  ,分割列
        data = pd.read_csv('{}'.format(self.function_profile_file), header=0, sep=',')
        function_profiler = {}
        # 分别取每一列元素组成元组
        for instance_num, rps, cpu, mem, vio_rate in zip(data['instance_num'], data['rps'], data['cpu'], data['mem'], data['avg_vio_rate']):
            if rps in function_profiler.keys():
                function_profiler[rps][instance_num] = (cpu, mem, vio_rate)
            else:
                function_profiler[rps] = {}
                function_profiler[rps][instance_num] = (cpu, mem, vio_rate)
        return function_profiler

    def init_rps2instance_num(self):
        """
            return a map: (rps->the expected_instance_num to meet the latency slo constraint)
        """
        rps2instance_num = {}
        for rps in self.function_profiler.keys():
            rps2instance_num[rps] = self.max_instance_num
            for instance_num in range(1, self.max_instance_num+1):
                if self.function_profiler[rps][instance_num][2] <= self.target_vio_rate:
                    rps2instance_num[rps] = instance_num
                    break
        return rps2instance_num

    def get_estimated_status(self, rps, instance_num):
        """
            return the average cpu usage, average mem usage, average latencty violation rate, and the expected_instance_num to meet the latency slo constraint
        """
        # TODO: tabulation method is used here for a demo, this can be further optimized in future work
        if rps <= 0:
            return 0, 0, 0, 1
        elif rps <= 20:
            # expected_instance_num 
            return self.function_profiler[rps][instance_num][0], self.function_profiler[rps][instance_num][1], self.function_profiler[rps][instance_num][2], self.rps2instance_num[rps]
        else:
            expected_instance_num = rps/4
            return self.cpu_limit, self.mem_limit, 1, expected_instance_num
