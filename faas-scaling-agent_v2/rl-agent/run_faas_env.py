import os
import torch
import time

from tianshou.data import Batch
from tianshou.policy import DQNPolicy, PPOPolicy, PGPolicy, QRDQNPolicy, C51Policy, FQFPolicy, IQNPolicy, RainbowPolicy, DiscreteSACPolicy
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net, Recurrent
from tianshou.utils.net.discrete import Actor, Critic, FractionProposalNetwork, FullQuantileFunction, ImplicitQuantileNetwork, NoisyLinear
from utils import get_args
from faas_env import OpenFaaS


if __name__ == '__main__':
    args = get_args()

    env = OpenFaaS(args)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.n

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


    policy.load_state_dict(torch.load('{}/{}/{}/policy.pth'.format(args.logdir, args.task, args.algo)))
    policy.eval()
    print('rl-started...')
    s, _ = env.reset()
    while True:
        batch = Batch(obs=[s], info=None)
        act = policy(batch).act[0]
        s, _, done, _, _ = env.step(act)
        print(s, act, env.instance_num)
        time.sleep(env.step_interval)
