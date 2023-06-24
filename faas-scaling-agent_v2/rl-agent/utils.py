import torch
import argparse

#SUPPORTED_ALGOS = ['dqn', 'drqn', 'pg', 'qrdqn', 'fqf', 'iqn', 'sac']
SUPPORTED_ALGOS = ['dqn', 'drqn', 'pg', 'sac']

def get_args():
    parser = argparse.ArgumentParser()
    """         parameters for drl         """
    parser.add_argument('--task', type=str, default='Sim-Env')
    parser.add_argument('--reward-threshold', type=float, default=990)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--is-test', action='store_true', help='is testing or not, default is training')
    parser.add_argument('--algo', type=str, default='drqn', help='supported algorithm: dqn, drqn, pg, sac, default is drqn')
    # dqn
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    #the number of transitions the collector would collect before the network update, i.e., trainer will collect
    #"step_per_collect" transitions and do some policy network update repeatedly in each epoch.
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    # drqn
    parser.add_argument('--stack-num', type=int, default=4)
    # parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--layer-num', type=int, default=2)
    # ppo
    # parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    # pg
    # parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--episode-per-collect', type=int, default=4)
    # parser.add_argument('--repeat-per-collect', type=int, default=2)
    # parser.add_argument('--rew-norm', type=int, default=1)
    # qrdqn
    parser.add_argument('--num-quantiles', type=int, default=200)
    # fqf
    # parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--fraction-lr', type=float, default=2.5e-9)
    parser.add_argument('--num-fractions', type=int, default=32)
    parser.add_argument('--num-cosines', type=int, default=64)
    # parser.add_argument('--ent-coef', type=float, default=10.)
    # iqn
    # parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--sample-size', type=int, default=32)
    parser.add_argument('--online-sample-size', type=int, default=8)
    parser.add_argument('--target-sample-size', type=int, default=8)
    # sac
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    # parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    # parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--auto-alpha', action="store_true", default=False)

    """         parameters for env         """
    parser.add_argument('--cpu-limit', type=float, default=0.2, help='the cpu limit set by user when deploy function')
    parser.add_argument('--mem-limit', type=float, default=20000000, help='the mem limit set by user when deploy function')
    parser.add_argument('--max-instance-num', type=int, default=4)
    parser.add_argument('--step-interval', type=int, default=10, help='the time interval to trigger rl, which is in seconds')
    parser.add_argument('--workload-data-path', type=str, default='../load_generate/filtered_data/train')
    parser.add_argument('--max-rps', type=int, default=30, help='the maximum number of requests per second')
    parser.add_argument('--target-vio-rate', type=float, default=0.1)
    parser.add_argument('--function-profile-file', type=str, default='../collect_data/function_profile_table.csv',
                        help='the path of function-profile-file, in the format of (instance_num,qps,cpu,mem,avg_vio_rate)')
    parser.add_argument('--max-step', type=int, default=1000, help='the maximum number of steps for an episode')

    """         specific parameters for real-world env         """
    parser.add_argument('--use-drl', action='store_true', help='whether use drl or not, default is not')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='the ip where you deploy OpenFaaS')
    parser.add_argument('--port', type=int, default=31112, help='the port where you deploy OpenFaaS')
    parser.add_argument('--function-name', type=str, default='hello-python')
    parser.add_argument('--save-log-dir', type=str, default='data',
                        help='when running real-world env, we save the collected function status for later analysis')

    args = parser.parse_known_args()[0]
    assert args.algo in SUPPORTED_ALGOS, 'Invalid algorigthm'
    return args

