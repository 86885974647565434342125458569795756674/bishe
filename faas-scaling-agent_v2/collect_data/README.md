# collect_data

## 项目说明

该目录下包括：
* `collect_data.py`：数据采集程序，用于实验前的数据采集，测试不同实例数和不同RPS(Request per Second)下的系统平均CPU使用率和延迟SLO违约率，用于构建仿真环境以加速训练
* `data.tar.gz`：之前虚拟机下实验采集的样例数据，供参考

## 运行说明

运行脚本：
```bash
$ python collect_data.py --ip=OpenFaaS的IP地址 --port=OpenFaaS的端口 --function-name=你要测试的函数名
```

其他重要参数可运行`python collect_data.py --help`查看或直接查看源代码

> 运行前请关闭OpenFaaS的弹性伸缩功能，可以在部署OpenFaaS时删除`prometheus-cfg.yml`中的告警规则或者关掉Prometheus和AlertManager的实例。
>
> `../others/yaml-rl`目录提供了之前Demo实验时部署OpenFaaS的配置文件，里面删除了`prometheus-cfg.yml`中的告警规则，可以直接使用该目录下的配置文件部署OpenFaaS。

程序运行时会模拟不同实例数和不同RPS，并从OpenFaaS的`/system/functions`接口收集函数状态，将接口返回的数据保存在`{%Y%m%d_%H%M%S}_function_status.json`文件；同时会记录函数调用的时延，保存在`latency_{idx}.log`文件中，`idx`用于区分记录数据的线程。

默认情况下，所有测试时记录的数据保存在`./data/{instance_num}-{rps}`）。数据采集结束后，
会读取该目录下的数据并计算得到`function_profile_table`，该表格可用于构建仿真环境，格式为：
```
instance_num,rps,cpu,avg_vio_rate
1,1,0.04938485533333333,0.0
1,2,0.1032049326,0.0
1,3,0.16078478473333335,0.0
1,4,0.17561768386666668,0.008565673834510625
...
```
