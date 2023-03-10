import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import core
import logging
# from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import traceback
import wandb
import imageio

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cuda'):
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        # print(f'shapes in finish_path: {self.rew_buf[path_slice].shape}, {last_val.shape}, {self.rew_buf[path_slice]}, {last_val}')
        rews = torch.cat((self.rew_buf[path_slice], last_val.unsqueeze(0)))
        vals = torch.cat((self.val_buf[path_slice], last_val.unsqueeze(0)))
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = torch.tensor(np.copy(core.discount_cumsum(deltas, self.gamma * self.lam)), device=self.device)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = torch.tensor(np.copy(core.discount_cumsum(rews, self.gamma)[:-1]), device=self.device)
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # print(f"self.adv_buf: {self.adv_buf}")
        adv_mean, adv_std = torch.mean(self.adv_buf), torch.std(self.adv_buf) #mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: v for k,v in data.items()}



def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, device='cuda', 
        ref_env_fn=None, cmd_args=None):
    """
    Proximal Policy Optimization (by clipping), 
    with early stopping based on approximate KL
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Set up logger and save configuration
    # logger = logging.getLogger() #EpochLogger(**logger_kwargs)
    # logger.setLevel(logging.DEBUG)
    # logger.save_config(locals())

    # Random seed
    seed += 10000 #* proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    ref_env = ref_env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")
    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    print(f"ac: {ac}")
    # Sync params across processes
    # sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch) # / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, device=device)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)
    universal_pi_iter = 0
    universal_v_iter = 0
    wandb.define_metric("pi/step")
    wandb.define_metric("pi/*", step_metric="pi/step")
    wandb.define_metric("v/step")
    wandb.define_metric("v/*", step_metric="v/step")
    def update(epoch):
        nonlocal universal_pi_iter
        nonlocal universal_v_iter
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            universal_pi_iter = universal_pi_iter+1
            # print(f'universal_pi_iter={universal_pi_iter}')
            loss_pi, pi_info = compute_loss_pi(data)
            # print(f'pi_info_kl: {pi_info["kl"]}')
            # wandb.log({"loss_pi": loss_pi}, step=universal_pi_iter)
            # for i, val in enumerate(ac.pi.log_std):
            # print(f'universal_pi_iter={universal_pi_iter}, wandb logging: { dict(**{"pi/loss": loss_pi.item(), "pi/step": universal_pi_iter}, **{f"pi/log_std[{i}]": val.item() for i,val in enumerate(ac.pi.log_std)}) }')
            wandb.log({**{"pi/loss": loss_pi.item(), "pi/step": universal_pi_iter}, **{f"pi/log_std[{i}]": val.item() for i,val in enumerate(ac.pi.log_std)}})
            kl = np.mean(pi_info['kl'])
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            universal_v_iter = universal_v_iter+1
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            wandb.log({"v/loss_v": loss_v, "v/step": universal_v_iter})
        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(loss_pi.item() - pi_l_old),
        #              DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    o, info = o
    # Main loop: collect experience in env and update/log each epoch
    rewards = []
    ep_dist = {}
    best_reward = -np.inf
    universal_timestep = 0
    wandb.define_metric("reward/step")
    wandb.define_metric("reward/*", step_metric="reward/step")
    total_env_interacts = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        for t in range(local_steps_per_epoch):
            universal_timestep += 1
            start = time.time()
            try:
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(f"o: {o}")
                exit(1)
            next_o, r, d, _, info = env.step(a.cpu().numpy())
            total_env_interacts += 1
            ep_ret += r
            ep_len += 1
            for k, v in info['imitation_dist'].items():
                ep_dist[k] = ep_dist.get(k,0) + v
            # save and log
            buf.store(torch.as_tensor(o, device=device), torch.as_tensor(a, device=device), torch.as_tensor(r, device=device), torch.as_tensor(v, device=device), torch.as_tensor(logp, device=device))
            # logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o
            # print(f"next_o: {next_o}")

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
                else:
                    v = torch.tensor(0, device=device)
                buf.finish_path(v)
                # if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    # logger.store(EpRet=ep_ret, EpLen=ep_len)
                rewards.append(ep_ret / ep_len)
                wandb.log({**{f"reward/dist-{k}": v for k,v in ep_dist.items()}, **{"reward/avg reward vs number of episodes": ep_ret / ep_len, "reward/total reward vs number of episodes": ep_ret, "reward/episode length vs number of episodes": ep_len, "reward/step": total_env_interacts}})
                if ep_ret > best_reward:
                    best_reward = ep_ret
                    gif_path = f'gifs/run-{cmd_args.run}-interacts-{total_env_interacts}.gif'
                    save_gif(ac, env, gif_path)
                    ref_gif_path = f'gifs/exp-{cmd_args.exp_name}-run-{cmd_args.run}-interacts-{total_env_interacts}-ref.gif'
                    save_gif(ac, ref_env, ref_gif_path)
                    # imageio.mimsave(f'gifs/run-4-interacts-{total_env_interacts}.gif', frames)
                    wandb.log({"reward/video": wandb.Video(gif_path)})
                    wandb.log({"reward/ref-video": wandb.Video(ref_gif_path)})
                    print(f'saving new best reward model with reward={best_reward}')
                    torch.save({'ac': ac.state_dict(), 'ac_kwargs': ac_kwargs}, f'models/run-{cmd_args.run}-best-reward.pt')
                    
                    # state = torch.load('models/run-4-best-reward.pt')
                    # ac_load = core.MLPActorCritic(env.observation_space, env.action_space, **state['ac_kwargs'])
                    # ac_load.load_state_dict(state['ac'])
                    # for p1, p2 in zip(ac.parameters(), ac_load.parameters()):
                    #     if p1.data.ne(p2.data).sum() > 0:
                    #         print('models have UNEQUAL weights')
                    # print('models have EQUAL weights')

                    # avg_reward = simulate(ac, env)
                    # avg_reward_load = simulate(ac_load, env)
                    # print(f'Best model performance: avg reward={avg_reward}')
                # wandb.log({"total reward vs number of episodes": ep_ret}, step=len(rewards))
                # wandb.log({"episode length vs number of episodes": ep_len}, step=len(rewards))
                o, ep_ret, ep_len, ep_dist = env.reset(), 0, 0, {}
                o, info = o
            # print(f'epoch={epoch}, step={t}, time={time.time()-start}')
        print(f'epoch={epoch}, time={time.time()-epoch_start}')
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            # logger.save_state({'env': env}, None)
            torch.save({'ac': ac.state_dict(), 'ac_kwargs': ac_kwargs}, f'models/run-4-epoch-{epoch}.pt')

        # Perform PPO update!
        start_update = time.time()
        update(epoch)
        print(f'update time={time.time()-start_update}')

        # Log info about epoch
        # logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        # logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        # logger.log_tabular('Time', time.time()-start_time)
        # logger.dump_tabular()

def save_gif(ac, env, path):
    o, info = env.reset()
    frames = [env.render()]
    while True:
        a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
        o, r, terminated, truncated, info = env.step(a.cpu().numpy())
        frames.append(env.render())
        if terminated or truncated:
            imageio.mimsave(path, frames)
            break

def simulate(ac, env):
    o, info = env.reset()
    rewards = []
    lengths = []
    r = 0
    step = 0
    ep = 0
    while True:
        step += 1
        a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
        o, reward, terminated, truncated, info = env.step(a.cpu().numpy())
        r += reward
        if terminated or truncated:
            o, info = env.reset()
            ep += 1
            rewards.append(r)
            # print(f'episode={ep}, length={step}, total reward={r}')
            if ep > 100:
                # print('breaking')
                break
            r = 0
            step = 0
    return sum(rewards) / len(rewards)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Walker2d-v4')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    # torch.cuda.set_device(0)

    from gymnasium.envs.registration import register
    import imageio

    register(
        id='WalkerMimic-v4',
        entry_point='walker_mimic_env:Walker2dEnv',
        max_episode_steps=300,
    )

    # mpi_fork(args.cpu)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    wandb.init(project="rl-2d-walker", name=args.exp_name, reinit=False)
    ppo(lambda : gym.make(args.env, ref=False, render_mode="rgb_array"), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, device=args.device, 
        ref_env_fn=lambda : gym.make(args.env, ref=True, render_mode="rgb_array"), cmd_args=args)
    # state = torch.load(f'models/run-{args.run}-best-reward.pt')
    # env = gym.make(args.env, render_mode="rgb_array")
    # ac_load = core.MLPActorCritic(env.observation_space, env.action_space, **state['ac_kwargs'])
    # ac_load.load_state_dict(state['ac'])
    # save_gif(ac_load, env, '2d-walker-new.gif')

# deepmimic run
# python ppo.py --exp_name mimic-1 --env WalkerMimic-v4