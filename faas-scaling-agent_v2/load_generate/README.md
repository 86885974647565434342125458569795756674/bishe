# 项目说明

该目录下包括：
* `filtered_data`：筛选后的工作流数据集
* `load_generate.py`：读取工作流数据集并发起请求

# filtered_data

该项目使用微软Azure的开源数据集来模拟真实业务的请求到达情况，建议先阅读该数据集的[说明文档](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
)。

为了方便实验，我们在该数据集的基础上进一步做以下处理：

1. 选择数据集中请求量比较大的函数，因为我们的研究场景是弹性伸缩，如果选择的函数请求量较少，就不构成弹性伸缩的需要，无法验证算法的有效性。
2. 将数据集中大量请求量为0的时间段进行压缩，使用以下正则表达式进行替换：pattern=`(,0){5,}`，替换为`,0,0,0,0,0`。这一步的目的是考虑到如果长时间调用量为0，上下文的相关性其实可以忽略不计，对于请求预测来说完全可以视为两个独立的周期；另一个目的其实也是为了缩短测试的时间，如果大量请求量为0的话这段时间是不需要弹性伸缩的。
3. 将处理后的数据集划分为测试集和训练集，训练集用于模拟环境强化学习训练，测试集用于真实场景验证训练后的效果。这里将`262.csv`作为测试集主要是考虑到实验时设备的资源限制，选择了这个请求量不是特别大且能够触发OpenFaaS弹性伸缩的负载。

# 运行说明

运行：
```bash
$ python load_generate.py --ip=OpenFaaS的IP地址 --port=OpenFaaS的端口 --function-name=你要测试的函数名
```

其他重要参数可运行`python load_generate.py --help`查看或直接查看源代码

# 其他

该程序其实存在两个小问题：

1. 当`python apscheduler`同时运行的任务数超过了默认的最大实例数，部分任务会被跳过并提示：`Run time of job "job_func (trigger: date[2023-01-01 17:13:00 CST], next run at: 2023-01-01 17:13:00 CST)" was missed by 0:00:01.280233`
2. 调用`apscheduler`的`add_job`函数添加任务的代码（如下）会漏掉部分请求，如`c = call // 60`，实际发起的请求数为`c * 60`，余数部分被漏掉了。当`call=100`，`c=1`，就有40个请求被漏掉了。 

```python
for minute, call in enumerate(f.invoke_counts):
    if call > 0:
        if call < 60:
            interval = 60 // call
            for i in range(call):
                scheduler.add_job(job_func, 'date',
                                  run_date=now + timedelta(minutes=minute, seconds=interval),
                                  args=[url, str(minute) + 'm and' + str(interval)])
        else:
            c = call // 60
            for t in range(60):
                for i in range(c):
                    scheduler.add_job(job_func, 'date',
                                      run_date=now + timedelta(minutes=minute, seconds=t),
                                      args=[url, str(minute) + 'm and' + str(1)])
```

这两个问题会导致请求不完全严格按照数据集中的请求量发起，但不会影响Demo的实验结论，所以这里暂时没有改。后续的实验可能需要对此处问题进行修复。