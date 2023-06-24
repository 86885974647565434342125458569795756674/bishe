import time
import requests
import argparse
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd


def invoke(url, params=None):
    r = requests.get(url=url, auth=('admin', 'admin'), params=params)

class Function:
    function = ''
    # list.size=1400
    invoke_counts = [0] * 1400

    def __init__(self, function, invoke_counts):
        self.function = function
        # map把invoke_counts每个元素变成int，list变成列表
        self.invoke_counts = list(map(int, invoke_counts))


def read_data(path, n=1):
    """read invoke data

    Args:
        path (str, required): data path.
        n (int, optional): data num. Defaults to 1.
    """
    res=[]
    data = pd.read_csv(path, header=None, sep=' ')
    i=0
    for d0,d1 in zip(data[0],data[1]):
        if i==n:
            break
        else:
            i=i+1    
        item=d1.split(',')
        func=Function(d0,item)
        res.append(func)
    return res


def job_func(url, text):
    print('invoke time', datetime.now(), text)
    invoke(url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='the ip where you deploy OpenFaaS')
    parser.add_argument('--port', type=int, default=31112, help='the port where you deploy OpenFaaS')
    parser.add_argument('--function-name', type=str, default='hello-python')

    parser.add_argument('--workload-path', type=str, default='filtered_data/test/262.csv')
    parser.add_argument('--n', type=int, default=1, help='the number of data to generate workload, one line is approximate to one day')
    args = parser.parse_known_args()[0]

    url = 'http://{}:{}/function/{}'.format(args.ip, args.port, args.function_name)
    datas = read_data(args.workload_path, args.n)
    scheduler = BackgroundScheduler()
    now = datetime.now()
    for f in datas:
        for minute, call in enumerate(f.invoke_counts):
            if call > 0:
                if call < 60:
                    # 取整除，一次调用间隔的秒数
                    interval = 60 // call
                    # 一分钟请求call次
                    for i in range(call):
                        scheduler.add_job(job_func, 'date',
                                          run_date=now + timedelta(minutes=minute, seconds=interval),
                                          args=[url, str(minute) + 'm and' + str(interval)])
                else:
                    # 一秒调用次数
                    c = call // 60
                    for t in range(60):
                        for i in range(c):
                            scheduler.add_job(job_func, 'date',
                                              run_date=now + timedelta(minutes=minute, seconds=t),
                                              args=[url, str(minute) + 'm and' + str(1)])

    print(len(scheduler.get_jobs()))
    scheduler.start()
    while True:
        pass

