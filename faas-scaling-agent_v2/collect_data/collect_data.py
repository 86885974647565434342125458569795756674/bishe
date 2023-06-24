import time
import requests
from datetime import datetime, timedelta
import argparse
from multiprocessing import Process
import os
import json
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

def invoke(url, save_dir, idx):
    # 调用函数的延迟
    start = time.time()
    requests.get(url, auth=('admin', 'admin'))
    with open('{}/latency_{}.log'.format(save_dir, idx), 'a+') as f:
        f.write('{}\n'.format(time.time()-start))

def run_workload(url, rps, save_dir, running_time):
    #  You could even use both at once, adding the process pool executor as a secondary executor.
    executors = {
        # a worker count of rps+1
        'default': ThreadPoolExecutor(rps+3),

        #  make use of multiple CPU cores. 
        # a worker count of 1
        'processpool': ProcessPoolExecutor(3)
    }
    scheduler = BackgroundScheduler(executors=executors)
    now = datetime.now()

    # running_time：发请求的总时间（s），每一秒都发rps次请求
    for sec in range(running_time):
        # 请求一秒请求rps次
        for idx in range(rps):
            # date: use when you want to run the job just once at a certain point of time
            # 特定的时间点触发，只执行一次。run_date (datetime 或 str)	作业的运行日期或时间
            scheduler.add_job(invoke, 'date', run_date=now+timedelta(seconds=3+sec),
                              args=[url, save_dir, idx])
    scheduler.start()
    # wait for additional 30 seconds to ensure that all tasks are completed
    time.sleep(running_time+30)


def collect_function_status(url, save_dir, per_collect_num, collect_interval):
    # 隔一段时间采集一次数据
    for _ in range(per_collect_num):
        time.sleep(collect_interval)
        r = requests.get(url, auth=('admin', 'admin'))
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open('{}/{}_function_status.json'.format(save_dir, date_str), 'w') as f:
            f.write(r.text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='the ip where you deploy OpenFaaS')
    parser.add_argument('--port', type=int, default=31112, help='the port where you deploy OpenFaaS')
    parser.add_argument('--function-name', type=str, default='hello-python')
    parser.add_argument('--max-instance-num', type=int, default=4)
    parser.add_argument('--max-rps', type=int, default=20)
    parser.add_argument('--per-collect-num', type=int, default=10, help='the collect number for each pair (instance-num,qps)')
    parser.add_argument('--collect-interval', type=int, default=10, help='the interval for each collect, in seconds')
    parser.add_argument('--save-root', type=str, default='data', help='the path to save the collected data')
    args = parser.parse_known_args()[0]

    # 实例个数
    for instance_num in range(1, args.max_instance_num+1):
       # instance_num=5
        data = {
            "replicas": instance_num
        }
        r = requests.post('http://{}:{}/system/scale-function/{}?namespace=openfaas-fn'.format(args.ip, args.port, args.function_name),
                          json=data, auth=('admin', 'admin'))
        
        print('wait for the instance to start...')
        time.sleep(15)

        # 每秒请求rps次
        for rps in range(1, args.max_rps+1):
         #   if rps<20:
        #        continue

            print('start collecting data, function-name={}, instance-num={}, rps={}'.format(args.function_name, instance_num, rps))

            save_dir = '{}/{}-{}'.format(args.save_root, instance_num, rps)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # we set the time for run_workload process higher than the collect_function_status process
            # to ensure p1 is running after we collect all data

            # 采样次数，采样间隔
            running_time = (args.per_collect_num+2) * args.collect_interval
            url1 = 'http://{}:{}/function/{}'.format(args.ip, args.port, args.function_name)
            p1 = Process(target=run_workload, args=(url1, rps, save_dir, running_time))
            p1.start()

            url2 = 'http://{}:{}/system/functions'.format(args.ip, args.port)
            p2 = Process(target=collect_function_status, args=(url2, save_dir, args.per_collect_num, args.collect_interval))
            p2.start()

            p1.join()
            p2.join()

            print('done...')
