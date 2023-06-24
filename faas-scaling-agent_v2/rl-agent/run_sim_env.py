import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch, Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy, PPOPolicy, PGPolicy, QRDQNPolicy, C51Policy, FQFPolicy, IQNPolicy, RainbowPolicy, DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net, Recurrent
from tianshou.utils.net.discrete import Actor, Critic, FractionProposalNetwork, FullQuantileFunction, ImplicitQuantileNetwork, NoisyLinear

from sim_env import SimEnv
from utils import get_args


#import gymnasium as gym
def train(args=get_args()):
    env = SimEnv(args)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.n
    ''' 
    train_envs = DummyVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(args.training_num)])
    '''    
    train_envs = DummyVectorEnv(
        [lambda: SimEnv(args) for _ in range(args.training_num)]
    )
    
    test_envs = DummyVectorEnv([lambda: SimEnv(args) for _ in range(args.test_num)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # log
    log_path = os.path.join(args.logdir, args.task, args.algo.lower())
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    if args.algo.lower() == 'dqn':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            # state_shape-hidden_sizes[i]-action_shape
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            # dueling=(Q_param, V_param),
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            # 多少次learn就更新目标网络，每次learn都更新Q
            target_update_freq=args.target_update_freq,
        )
        # buffer
        if args.prioritized_replay:
            buf = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                alpha=args.alpha,
                beta=args.beta,
            )
        else:
            buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
        # collector
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 10000:
                policy.set_eps(args.eps_train)
            elif env_step <= 50000:
                eps = args.eps_train - (env_step - 10000) / \
                    40000 * (0.9 * args.eps_train)
                policy.set_eps(eps)
            else:
                policy.set_eps(0.1 * args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            update_per_step=args.update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        )
    elif args.algo.lower() == 'drqn':
        # model
        net = Recurrent(args.layer_num, args.state_shape, args.action_shape,
                        args.device).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )
        # collector
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            stack_num=args.stack_num,
            ignore_obs_next=True
        )
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        # the stack_num is for RNN training: sample framestack obs
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def train_fn(epoch, env_step):
            policy.set_eps(args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            update_per_step=args.update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )
    elif args.algo.lower() == 'ppo':
        # model
        net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        if torch.cuda.is_available():
            actor = DataParallelNet(
                Actor(net, args.action_shape, device=None).to(args.device)
            )
            critic = DataParallelNet(Critic(net, device=None).to(args.device))
        else:
            actor = Actor(net, args.action_shape, device=args.device).to(args.device)
            critic = Critic(net, device=args.device).to(args.device)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            action_space=env.action_space,
            deterministic_eval=True,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv
        )
        # collector
        train_collector = Collector(
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
        )
        test_collector = Collector(policy, test_envs)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )
    elif args.algo.lower() == 'pg':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            softmax=True
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = PGPolicy(
            net,
            optim,
            dist,
            args.gamma,
            reward_normalization=args.rew_norm,
            action_space=env.action_space,
        )
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # collector
        train_collector = Collector(
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
        )
        test_collector = Collector(policy, test_envs)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        )
    elif args.algo.lower() == 'qrdqn':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            softmax=False,
            num_atoms=args.num_quantiles,
        )
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = QRDQNPolicy(
            net,
            optim,
            args.gamma,
            args.num_quantiles,
            args.n_step,
            target_update_freq=args.target_update_freq,
        ).to(args.device)
        # buffer
        if args.prioritized_replay:
            buf = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                alpha=args.alpha,
                beta=args.beta,
            )
        else:
            buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
        # collector
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 10000:
                policy.set_eps(args.eps_train)
            elif env_step <= 50000:
                eps = args.eps_train - (env_step - 10000) / \
                    40000 * (0.9 * args.eps_train)
                policy.set_eps(eps)
            else:
                policy.set_eps(0.1 * args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
        )
    elif args.algo.lower() == 'fqf':
        # model
        feature_net = Net(
            args.state_shape,
            args.hidden_sizes[-1],
            hidden_sizes=args.hidden_sizes[:-1],
            device=args.device,
            softmax=False
        )
        net = FullQuantileFunction(
            feature_net,
            args.action_shape,
            args.hidden_sizes,
            num_cosines=args.num_cosines,
            device=args.device
        )
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        fraction_net = FractionProposalNetwork(args.num_fractions, net.input_dim)
        fraction_optim = torch.optim.RMSprop(
            fraction_net.parameters(), lr=args.fraction_lr
        )
        policy = FQFPolicy(
            net,
            optim,
            fraction_net,
            fraction_optim,
            args.gamma,
            args.num_fractions,
            args.ent_coef,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)
        # buffer
        if args.prioritized_replay:
            buf = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                alpha=args.alpha,
                beta=args.beta
            )
        else:
            buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
        # collector
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 10000:
                policy.set_eps(args.eps_train)
            elif env_step <= 50000:
                eps = args.eps_train - (env_step - 10000) / \
                    40000 * (0.9 * args.eps_train)
                policy.set_eps(eps)
            else:
                policy.set_eps(0.1 * args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step
        )
    elif args.algo.lower() == 'iqn':
        # model
        feature_net = Net(
            args.state_shape,
            args.hidden_sizes[-1],
            hidden_sizes=args.hidden_sizes[:-1],
            device=args.device,
            softmax=False
        )
        net = ImplicitQuantileNetwork(
            feature_net,
            args.action_shape,
            num_cosines=args.num_cosines,
            device=args.device
        )
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = IQNPolicy(
            net,
            optim,
            args.gamma,
            args.sample_size,
            args.online_sample_size,
            args.target_sample_size,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)
        # buffer
        if args.prioritized_replay:
            buf = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                alpha=args.alpha,
                beta=args.beta
            )
        else:
            buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
        # collector
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 10000:
                policy.set_eps(args.eps_train)
            elif env_step <= 50000:
                eps = args.eps_train - (env_step - 10000) / \
                    40000 * (0.9 * args.eps_train)
                policy.set_eps(eps)
            else:
                policy.set_eps(0.1 * args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step
        )
    elif args.algo.lower() == 'sac':
        # model
        net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        actor = Actor(net, args.action_shape, softmax_output=False,
                    device=args.device).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic1 = Critic(net_c1, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic2 = Critic(net_c2, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        # better not to use auto alpha in CartPole
        if args.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            args.tau,
            args.gamma,
            args.alpha,
            estimation_step=args.n_step,
            reward_normalization=args.rew_norm
        )
        # collector
        train_collector = Collector(
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
        )
        test_collector = Collector(policy, test_envs)
        
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False
        )

    assert stop_fn(result['best_reward'])


def test(args=get_args()):
    env = SimEnv(args)

    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.algo.lower() == 'dqn':
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            # dueling=(Q_param, V_param),
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
        )
    elif args.algo.lower() == 'drqn':
        # model
        net = Recurrent(args.layer_num, args.state_shape, args.action_shape,
                        args.device).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )
    elif args.algo.lower() == 'ppo':
        net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        if torch.cuda.is_available():
            actor = DataParallelNet(
                Actor(net, args.action_shape, device=None).to(args.device)
            )
            critic = DataParallelNet(Critic(net, device=None).to(args.device))
        else:
            actor = Actor(net, args.action_shape, device=args.device).to(args.device)
            critic = Critic(net, device=args.device).to(args.device)
        actor_critic = ActorCritic(actor, critic)
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            action_space=env.action_space,
            deterministic_eval=True,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv
        )
    elif args.algo.lower() == 'pg':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            softmax=True
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        dist = torch.distributions.Categorical
        policy = PGPolicy(
            net,
            optim,
            dist,
            args.gamma,
            reward_normalization=args.rew_norm,
            action_space=env.action_space,
        )
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
    elif args.algo.lower() == 'qrdqn':
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            softmax=False,
            num_atoms=args.num_quantiles,
        )
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = QRDQNPolicy(
            net,
            optim,
            args.gamma,
            args.num_quantiles,
            args.n_step,
            target_update_freq=args.target_update_freq,
        ).to(args.device)
    elif args.algo.lower() == 'fqf':
        # model
        feature_net = Net(
            args.state_shape,
            args.hidden_sizes[-1],
            hidden_sizes=args.hidden_sizes[:-1],
            device=args.device,
            softmax=False
        )
        net = FullQuantileFunction(
            feature_net,
            args.action_shape,
            args.hidden_sizes,
            num_cosines=args.num_cosines,
            device=args.device
        )
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        fraction_net = FractionProposalNetwork(args.num_fractions, net.input_dim)
        fraction_optim = torch.optim.RMSprop(
            fraction_net.parameters(), lr=args.fraction_lr
        )
        policy = FQFPolicy(
            net,
            optim,
            fraction_net,
            fraction_optim,
            args.gamma,
            args.num_fractions,
            args.ent_coef,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)
    elif args.algo.lower() == 'iqn':
        # model
        feature_net = Net(
            args.state_shape,
            args.hidden_sizes[-1],
            hidden_sizes=args.hidden_sizes[:-1],
            device=args.device,
            softmax=False
        )
        net = ImplicitQuantileNetwork(
            feature_net,
            args.action_shape,
            num_cosines=args.num_cosines,
            device=args.device
        )
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = IQNPolicy(
            net,
            optim,
            args.gamma,
            args.sample_size,
            args.online_sample_size,
            args.target_sample_size,
            args.n_step,
            target_update_freq=args.target_update_freq
        ).to(args.device)
    elif args.algo.lower() == 'sac':
        # model
        net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        actor = Actor(net, args.action_shape, softmax_output=False,
                    device=args.device).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic1 = Critic(net_c1, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic2 = Critic(net_c2, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        # better not to use auto alpha in CartPole
        if args.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            args.tau,
            args.gamma,
            args.alpha,
            estimation_step=args.n_step,
            reward_normalization=args.rew_norm
        )

    policy.load_state_dict(torch.load('{}/{}/{}/policy.pth'.format(args.logdir, args.task, args.algo.lower())))
    policy.eval()

    reward_list, avg_instance_num, cpu_list, mem_list, vio_list = [], [], [], [], []
    for _ in range(1):
        s, _ = env.reset()
        while True:
            batch = Batch(obs=[s], info=None)
            act = policy(batch).act[0]
            s, r, done, _, info = env.step(act)
            # print(s, act, r, info)
            reward_list.append(r)
            avg_instance_num.append(info['instance_num'])
            cpu_list.append(info['cpu'] * info['instance_num'])
            mem_list.append(info['mem'] * info['instance_num'])
            vio_list.append(info['vio_rate'])
            if done:
                break
    print('average reward={}\naverage instance num={}\naverage cpu usage={}\naverage mem usage={}\naverage violate rate={}\n'.format(
        sum(reward_list)/len(reward_list), sum(avg_instance_num)/len(avg_instance_num),
        sum(cpu_list)/len(cpu_list), sum(mem_list)/len(mem_list), sum(vio_list)/len(vio_list)))


if __name__ == '__main__':
    args = get_args()
    if args.is_test:
        test(args)
    else:
        train(args)
