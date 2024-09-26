import numpy as np
import torch as th
from torch import nn

import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

import time
from datetime import datetime
import os

import wandb

from PIL import Image

from replay_buffer import ReplayBuffer, EpisodesBuffer, ContinuousBuffer
from config import Config

from multiprocessing import Pipe, Process
import math


# https://github.com/oxwhirl/pymarl/blob/master/src/modules/agents/rnn_agent.py
class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()

        self.input_shape = input_shape
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# https://github.com/oxwhirl/pymarl/blob/master/src/components/epsilon_schedules.py
class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T /
                                                           self.exp_scaling)))
    pass


class MAC(nn.Module):
    def __init__(
        self,
        n_agents,
        input_shape,
        n_actions,
        hidden_dim=64,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=50_000,
    ):
        super().__init__()
        self.n_agents = n_agents

        # Consider the previous actions and agent's index
        self.input_shape = input_shape + n_actions + n_agents

        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.epsilon_anneal_time = epsilon_anneal_time

        self.agent = RNNAgent(self.input_shape, self.hidden_dim,
                              self.n_actions)

        self.scheduler = DecayThenFlatSchedule(
            self.epsilon_start,
            self.epsilon_finish,
            self.epsilon_anneal_time,
            decay='linear'
        )

    def select_actions(
        self,
        batch_observations,
        previous_actions,
        available_actions,
        t_ep,
        testing=True
    ):
        agent_outs = self.forward(batch_observations,
                                  previous_actions, available_actions)

        # https://github.com/oxwhirl/pymarl/blob/master/src/components/action_selectors.py
        masked_q_values = agent_outs.clone()
        masked_q_values[available_actions == 0.0] = -float('inf')

        q_values_actions = masked_q_values.max(dim=-1)[1]

        if testing:
            epsilon = 0.0
        else:
            if t_ep < self.epsilon_anneal_time:
                epsilon = self.scheduler.eval(t_ep)
            elif config.adaptive_epsilon:
                avail_actions_count = available_actions.sum(dim=-1)
                available_joint_actions_count = avail_actions_count.prod(dim=-1)
                available_joint_actions_count[available_joint_actions_count < 5] = 5
                base_epsilon = (available_joint_actions_count.log()).pow(0.5).mul(config.tanh_coef).tanh()
                if config.maxing_adaptive_epsilon:
                    base_epsilon[base_epsilon < config.min_epsilon] = config.min_epsilon
                if config.uniform_exploration:
                    epsilon = th.zeros_like(avail_actions_count, dtype=th.float32)
                    for i in range(len(epsilon)):
                        epsilon[i] = base_epsilon[i] * avail_actions_count[i] / config.env_n_actions
                else:
                    epsilon = base_epsilon
            else:
                epsilon = self.scheduler.eval(t_ep)

        random_numbers = th.rand_like(agent_outs[:, :, 0])
        explore = th.zeros_like(random_numbers, dtype=int)
        if type(epsilon) != float:
            for i in range(len(explore)):
                explore[i] = (random_numbers[i] < epsilon[i]).long()
        else:
            explore = (random_numbers < epsilon).long()

        explore_actions = Categorical(available_actions.float()).sample().long()
        picked_actions = explore * explore_actions + (1-explore) * masked_q_values.max(dim=-1)[1]

        return picked_actions, q_values_actions, epsilon

    def forward(self, batch_observations, previous_actions, available_actions):
        bs = batch_observations.shape[0]

        inputs = self.build_inputs(batch_observations, previous_actions)

        agent_outs, self.hidden_states = self.agent(
            inputs,
            self.hidden_states
        )

        self.hidden_states = self.hidden_states.reshape(bs, self.n_agents, -1)

        return agent_outs.reshape(bs, self.n_agents, self.n_actions)

    def build_inputs(self, batch_observations, previous_actions):
        bs = batch_observations.shape[0]

        inputs = [
            batch_observations,
            previous_actions,
            th.eye(self.n_agents, device=batch_observations.device).unsqueeze(0).expand(bs, -1, -1)
        ]

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs.reshape(-1, self.input_shape)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def reset_hidden(self, mask):
        if (mask == 0.0).any():
            mask = mask.view(-1, 1, 1).expand_as(self.hidden_states).to(device)
            self.hidden_states = self.hidden_states * mask


# https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py
class QMixer(nn.Module):
    def __init__(
        self,
        n_agents,
        state_shape,
        mixing_embed_dim=32,
        hypernet_layers=2,
        hypernet_embed=64,
    ):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed = hypernet_embed

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class OneHot:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32


# based on https://github.com/oxwhirl/pymarl/blob/master/src/runners/episode_runner.py
def run_one_episode(env_conns, mac, global_t=0, testing=True):
    global concat_time, p_sending_time, p_recieving_time, batch_appending_time, selecting_actions_time, running_episodes_time
    running_episodes_start = time.time()
    terminated_envs = set()
    for conn in env_conns:
        conn.send(('reset', None))
    envs_data = []
    i = len(env_conns)
    while i > 0:
        i -= 1
        if not conn.poll(20):
            print(f'\nrun_one_episode: wasn\'t able to get access to the envifonment number {i}\nRemoving this environment\n')
            env_conns.pop(i)
    for conn in env_conns:
        envs_data.append(conn.recv())
    obs = [
        th.tensor(d['obs']).unsqueeze(0)
        for d in envs_data
    ]
    state = [
        th.tensor(d['state'], dtype=th.float32).unsqueeze(0)
        for d in envs_data
    ]
    avail_actions = [
        th.tensor(d['avail_actions']).unsqueeze(0)
        for d in envs_data
    ]
    previous_actions = [
        th.zeros(config.env_n_agents, config.env_n_actions).unsqueeze(0)
        for _ in envs_data
    ]
    mac.init_hidden(batch_size=len(env_conns))
    episodes_return = [0 for _ in env_conns]
    env_t = global_t
    t = 0

    envs_info = [None] * len(env_conns)

    batch_data = [list() for _ in env_conns]

    while len(terminated_envs) < len(env_conns):
        if t > config.env_max_episode_length:
            print(f'run_one_episode: Something went wrong; current step {t} is larger than episode limit {config.env_max_episode_length}')
            return batch_data, envs_info
        concat_start = time.time()
        inp_obs = th.concat(obs).to(device)
        inp_previous_actions = th.concat(previous_actions).to(device)
        inp_avail_actions = th.concat(avail_actions).to(device)
        concat_time += time.time() - concat_start

        selecting_actions_start = time.time()
        actions, q_values_actions, epsilon = mac.select_actions(
            inp_obs,
            inp_previous_actions,
            inp_avail_actions,
            env_t,
            testing
        )
        selecting_actions_time += time.time() - selecting_actions_start
        for i, conn in enumerate(env_conns):
            if i in terminated_envs:
                continue
            p_sending_start = time.time()
            conn.send(('step', actions[i].cpu()))
            p_sending_time = time.time() - p_sending_start
        for i, conn in enumerate(env_conns):
            if i in terminated_envs:
                continue
            p_recieving_start = time.time()
            envs_data[i] = conn.recv()
            p_recieving_time += time.time() - p_recieving_start
            batch_appending_start = time.time()
            env_state = th.tensor(envs_data[i]['state'], dtype=th.float32).to(batch_device)
            env_avail_actions = th.tensor(envs_data[i]['avail_actions']).to(batch_device)
            env_obs = th.tensor(envs_data[i]['obs']).to(batch_device)
            terminated = envs_data[i]['terminated']
            if terminated == 1.0:
                terminated_envs.add(i)

            envs_info[i] = envs_data[i]['info']

            if config.scale_rewards:
                envs_data[i]['reward'] /= config.env_n_agents

            data = {
                'state': state[i][0],
                'avail_actions': avail_actions[i][0],
                'obs': obs[i][0],
                'reward': envs_data[i]['reward'],
                'actions': actions[i].to(batch_device),
                'actions_onehot': actions_onehot_encoder.transform(actions[i].unsqueeze(-1)).to(batch_device),
                'terminated': 1.0 if terminated != envs_data[i]['info'].get('episode_limit', False) else 0.0,
                # is required for masking
                'filled': 1.0,
            }
            obs[i] = env_obs.unsqueeze(0)
            avail_actions[i] = env_avail_actions.unsqueeze(0)
            state[i] = env_state.unsqueeze(0)
            if t > 0:
                previous_actions[i] = batch_data[i][-1]['actions_onehot'].unsqueeze(0)
            episodes_return[i] += data['reward']
            if type(epsilon) == float:
                # in case that epsilon is the same for all agents
                data.setdefault('epsilon', epsilon)
            else:
                data.setdefault('epsilon', epsilon[i])
            batch_data[i].append(data)
            batch_appending_time += time.time() - batch_appending_start

        t += 1

    concat_start = time.time()
    inp_obs = th.concat(obs).to(device)
    inp_previous_actions = th.concat(previous_actions).to(device)
    inp_avail_actions = th.concat(avail_actions).to(device)
    concat_time += time.time() - concat_start
    for i in range(len(env_conns)):
        batch_appending_start = time.time()

        if config.scale_rewards:
            envs_data[i]['reward'] /= config.env_n_agents

        data = {
            'state': state[i][0],
            'avail_actions': avail_actions[i][0],
            'obs': obs[i][0],
            'reward': 0.0,
            'actions': th.zeros_like(actions[i].to(batch_device)),
            'actions_onehot': actions_onehot_encoder.transform(th.zeros_like(actions[i].unsqueeze(-1))).to(batch_device),
            'terminated': 0.0,
            # is required for masking
            'filled': 0.0,
        }
        if type(epsilon) == float:
            # in case that epsilon is the same for all agents
            data.setdefault('epsilon', epsilon)
        else:
            data.setdefault('epsilon', epsilon[i])

        batch_data[i].append(data)
        batch_appending_time += time.time() - batch_appending_start

        envs_info[i].setdefault('episode_return', episodes_return[i])

    running_episodes_time += time.time() - running_episodes_start
    return batch_data, envs_info


def get_zero_elements(count, episode):
    elements = []
    zero_element = {}
    for k in float_tensors_keys:
        zero_element.setdefault(k, th.zeros(episode[0][k].shape).to(batch_device))
    zero_element.setdefault('avail_actions',
                            th.zeros(episode[0]['avail_actions'].shape,
                                     dtype=int).to(batch_device))
    zero_element.setdefault('reward', 0.0)
    zero_element.setdefault('terminated', 0.0)
    zero_element.setdefault('filled', 0.0)
    if config.adaptive_epsilon:
        zero_element.setdefault('epsilon', 0.0)

    for i in range(count):
        elements.append(get_batch_copy([[zero_element]])[0][0])
    return elements


def extend_batch(batch_episodes):
    # Adds empty steps to episodes so every episodes lengths are the
    # same
    max_len = max(len(episode) for episode in batch_episodes)

    zero_element = {}

    for k in float_tensors_keys:
        zero_element.setdefault(k, th.zeros(batch_episodes[0][0][k].shape).to(batch_device))
    zero_element.setdefault('avail_actions',
                            th.zeros(batch_episodes[0][0]['avail_actions'].shape,
                                     dtype=int).to(batch_device))
    zero_element.setdefault('reward', 0.0)
    zero_element.setdefault('terminated', 0.0)

    zero_element.setdefault('filled', 0.0)
    zero_element.setdefault('epsilon', 0.0)

    for episode in batch_episodes:
        for i in range(len(episode), max_len):
            episode.append(get_batch_copy([[zero_element]])[0][0])


def get_batch_step_data(batch_episodes, step_t):
    res = {k: [] for k in batch_episodes[0][0].keys()}

    for episode in batch_episodes:
        assert step_t < len(episode)
        for k, l in res.items():
            l.append(episode[step_t][k])
    for k in tensors_keys:
        res[k] = th.stack(res[k])
    return res


def get_batch_copy(batch_episodes):
    res = []
    for episode_data in batch_episodes:
        ep_data_copy = []
        for step_data in episode_data:
            data = {}
            for k, v in step_data.items():
                if k in tensors_keys:
                    data.setdefault(k, v.clone().detach())
                else:
                    data.setdefault(k, v)
            ep_data_copy.append(data)
        res.append(ep_data_copy)

    return res


def get_batch_attr(batch_episodes, attr):
    if type(batch_episodes[0][0][attr]) == th.Tensor:
        return th.stack([th.stack([step[attr] for step in episode]) for episode in batch_episodes])
    return th.tensor([[step[attr] for step in episode] for episode in batch_episodes])


def init_hidden_on_sample(mac, target_mac, episodes_sample):
    max_ep_length = len(episodes_sample[0])
    batch_size = len(episodes_sample)

    mac.init_hidden(batch_size)
    for t in range(max_ep_length):
        data = get_batch_step_data(episodes_sample, t)
        if t > 0:
            previous_actions = get_batch_step_data(episodes_sample, t-1)['actions_onehot'].to(device)
            is_same_episode = th.tensor(get_batch_step_data(episodes_sample, t-1)['filled'], requires_grad=False)
            mac.reset_hidden(is_same_episode)
        else:
            previous_actions = th.zeros(get_batch_step_data(episodes_sample, t)['actions_onehot'].shape).to(device)
        mac.forward(data['obs'].to(device), previous_actions, data['avail_actions'].to(device))

    target_mac.init_hidden(batch_size)
    for t in range(max_ep_length):
        data = get_batch_step_data(episodes_sample, t)
        if t > 0:
            previous_actions = get_batch_step_data(episodes_sample, t-1)['actions_onehot'].to(device)
            is_same_episode = th.tensor(get_batch_step_data(episodes_sample, t-1)['filled'], requires_grad=False)
            target_mac.reset_hidden(is_same_episode)
        else:
            previous_actions = th.zeros(get_batch_step_data(episodes_sample, t)['actions_onehot'].shape).to(device)
        target_mac.forward(data['obs'].to(device), previous_actions, data['avail_actions'].to(device))


def train(
        mac,
        target_mac,
        mixer,
        target_mixer,
        batch_episodes,
        optimizer,
        should_init_hidden=True,
        start_action=None,
):
    global train_attrs_time
    train_attrs_start = time.time()
    obs = get_batch_attr(batch_episodes, 'obs')[:, :-1].unsqueeze(-1).to(device)
    rewards = get_batch_attr(batch_episodes, 'reward')[:, :-1].unsqueeze(-1).to(device)

    actions = get_batch_attr(batch_episodes, 'actions')[:, :-1].long().unsqueeze(-1).to(device)
    terminated = get_batch_attr(batch_episodes, 'terminated')[:, :-1].unsqueeze(-1).to(device)
    avail_actions = get_batch_attr(batch_episodes, 'avail_actions').to(device)
    states = get_batch_attr(batch_episodes, 'state').to(device)
    mask = get_batch_attr(batch_episodes, 'filled').unsqueeze(-1)[:, :-1].to(device)
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

    train_attrs_time += time.time() - train_attrs_start

    max_ep_length = len(batch_episodes[0])
    mac_out = []
    batch_size = len(batch_episodes)
    n_agents = actions.shape[-1]

    loss = 0.0
    # qmix
    if should_init_hidden:
        mac.init_hidden(batch_size)
    for t in range(max_ep_length):
        data = get_batch_step_data(batch_episodes, t)
        if t > 0:
            previous_actions = get_batch_step_data(batch_episodes, t-1)['actions_onehot'].to(device)
            is_same_episode = th.tensor(get_batch_step_data(batch_episodes, t-1)['filled'], requires_grad=False)
            mac.reset_hidden(is_same_episode)
        else:
            if start_action is not None:
                previous_actions = start_action.to(device)
            else:
                previous_actions = th.zeros(get_batch_step_data(batch_episodes, t)['actions_onehot'].shape).to(device)
        agent_outs = mac.forward(data['obs'].to(device), previous_actions, data['avail_actions'].to(device))
        mac_out.append(agent_outs)

    mac_out = th.stack(mac_out, dim=1)
    chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

    target_mac_out = []
    if should_init_hidden:
        target_mac.init_hidden(len(batch_episodes))
    for t in range(max_ep_length):
        data = get_batch_step_data(batch_episodes, t)
        if t > 0:
            previous_actions = get_batch_step_data(batch_episodes, t-1)['actions_onehot'].to(device)
            is_same_episode = th.tensor(get_batch_step_data(batch_episodes, t-1)['filled'], requires_grad=False)
            target_mac.reset_hidden(is_same_episode)
        else:
            if start_action is not None:
                previous_actions = start_action.to(device)
            else:
                previous_actions = th.zeros(get_batch_step_data(batch_episodes, t)['actions_onehot'].shape).to(device)
        target_agent_outs = target_mac.forward(data['obs'].to(device), previous_actions, data['avail_actions'].to(device))
        target_mac_out.append(target_agent_outs)

    # We don't need the first timesteps Q-Value estimate for calculating targets
    target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

    # Mask out unavailable actions
    target_mac_out[avail_actions[:, 1:] == 0] = -9999999

    # Get actions that maximise live Q (for double q-learning)
    mac_out_detach = mac_out.clone().detach()
    mac_out_detach[avail_actions == 0] = -9999999

    cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
    target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

    # Mix
    chosen_action_qvals = mixer(chosen_action_qvals, states[:, :-1])
    target_max_qvals = target_mixer(target_max_qvals, states[:, 1:])
    targets = rewards + config.gamma * (1 - terminated) * target_max_qvals

    td_error = (chosen_action_qvals - targets.detach())

    mask = mask.expand_as(td_error)
    masked_td_error = td_error * mask

    loss = (masked_td_error ** 2).sum() / mask.sum()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = th.nn.utils.clip_grad_norm_(params, config.grad_norm_clip)
    optimizer.step()
    loss = loss.item()
    return loss, grad_norm.item()


def save_model(model_dir):
    os.makedirs(model_dir)
    if model_dir[-1] != '/':
        model_dir += '/'
    th.save(mac.state_dict(), f'{model_dir}mac.pth')
    th.save(mixer.state_dict(), f'{model_dir}mixer.pth')
    if not config.omit_wandb:
        wandb.save(f'{model_dir}*')


def load_model(model_dir, env):
    global mac, exploration_network, exploration_networks, vae, mixer
    mac = MAC(
        config.env_n_agents,
        config.env_obs_size,
        config.env_n_actions,
        config.rnn_hidden_dim,
        epsilon_start=config.epsilon_start,
        epsilon_finish=config.epsilon_finish,
        epsilon_anneal_time=config.epsilon_anneal_time,
    ).to(device)
    mac.load_state_dict(th.load(f'{model_dir}mac.pth'))

    mixer = QMixer(config.env_n_agents, config.env_state_size).to(device)
    mixer.load_state_dict(th.load(f'{model_dir}mixer.pth'))


# https://github.com/oxwhirl/pymarl/blob/master/src/runners/parallel_runner.py
def env_worker(remote, env_fn):
    # Make environment
    env = env_fn()
    was_initialized = False
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            try:
                env.reset()
            except:
                print('Exception: failed to reset an environment. Trying to reset it once more time.')
                time.sleep(1)
                env.reset()
                print('Reset successfully!')
            if not was_initialized:
                env._episode_count = 1
                was_initialized = True

            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


train_attrs_time = 0
concat_time = 0
p_sending_time = 0
p_recieving_time = 0
batch_appending_time = 0
selecting_actions_time = 0
running_episodes_time = 0


def main():
    global mac, mixer, target_mac, target_mixer, actions_onehot_encoder, optimizer, writer, t_env, float_tensors_keys, tensors_keys, params

    actions_onehot_encoder = OneHot(config.env_n_actions)

    mac = MAC(
        config.env_n_agents,
        config.env_obs_size,
        config.env_n_actions,
        hidden_dim=config.rnn_hidden_dim,
        epsilon_start=config.epsilon_start,
        epsilon_finish=config.epsilon_finish,
        epsilon_anneal_time=config.epsilon_anneal_time,
    ).to(device)

    target_mac = MAC(
        config.env_n_agents,
        config.env_obs_size,
        config.env_n_actions,
        hidden_dim=config.rnn_hidden_dim,
    ).to(device)
    target_mac.load_state_dict(mac.state_dict())

    mixer = QMixer(config.env_n_agents, config.env_state_size).to(device)

    target_mixer = QMixer(config.env_n_agents, config.env_state_size).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    params = list(mac.parameters())
    params += list(mixer.parameters())

    optimizer = Adam(params=params, lr=config.lr)

    float_tensors_keys = ['state', 'obs', 'actions', 'actions_onehot']
    tensors_keys = ['state', 'avail_actions', 'obs', 'actions', 'actions_onehot']

    group_name = config.map_name
    test_name = f'{group_name}_{len(os.listdir("tensorboard"))}'

    if not config.omit_wandb:
        wandb.init(
            project=config.wandb_project,
            group=group_name,
            sync_tensorboard=True,
            config=config.dict(),
        )
    writer = SummaryWriter(f'tensorboard/{test_name}')

    # based on https://github.com/oxwhirl/pymarl/blob/master/src/run.py
    t_env = 0
    last_test_T = -config.test_interval - 1
    model_saves_count = 0

    start_time = time.time()

    if config.buffer_by_episodes:
        replay_buffer = EpisodesBuffer(
            config.replay_buffer_episodes
        )
    elif config.continuous_buffer:
        replay_buffer = ContinuousBuffer(
            config.replay_buffer_steps,
            config.buffer_sequence_size
        )
    else:
        replay_buffer = ReplayBuffer(
            config.replay_buffer_steps,
            config.buffer_sequence_size,
            config.burn_in
        )

    wins_count = 0
    last_res = 0
    last_ep_count = 0
    episodes_count = 0
    dead_enemies = 0
    dead_allies = 0
    rewards = 0

    loss = 0
    grad_norm = 0
    trains_count = 0

    last_target_update_step = 0

    steps_to_train = None

    test_winrate = []

    sampling_time = 0
    buffer_inserting_time = 0
    train_time = 0
    train_prepare_time = 0
    test_time = 0

    if config.buffer_by_episodes:
        steps_per_batch = config.batch_size
    else:
        steps_per_batch = config.batch_size * config.buffer_sequence_size
    while t_env < config.t_max:
        if (t_env - model_saves_count*config.save_model_interval) >= 0:
            print('Saving the model')
            save_model(f'models/{test_name}/{model_saves_count}/')
            model_saves_count += 1
        with th.no_grad():
            sample_start_time = time.time()
            episode_batch, env_info = run_one_episode(parent_conns, mac, t_env, testing=False)
            sampling_time += time.time() - sample_start_time
            buffer_inserting_start = time.time()

        # removing episodes at which environment has restarted
        i = len(episode_batch)
        while i > 0:
            i -= 1
            try:
                if config.env_name == 'smac':
                    wins_count += env_info[i]['battle_won']
                    dead_enemies += env_info[i]['dead_enemies']
                    dead_allies += env_info[i]['dead_allies']
                rewards += env_info[i]['episode_return']
                episodes_count += 1
            except Exception as e:
                print(f"Unable to get env info: exception: '{e}'")
                env_info.pop(i)
                episode_batch.pop(i)

        if len(episode_batch) == 0:
            print('All environments restarted at the same time')
            continue

        if config.env_name in ['pogema', 'pogema-follower']:
            writer.add_scalar('avg_throughput', env_info[0]['avg_throughput'], t_env)

        if steps_to_train is not None:
            if config.buffer_by_episodes:
                steps_to_train += len(episode_batch) * config.train_frequency
            else:
                for ep in episode_batch:
                    steps_to_train += len(ep) * config.train_frequency

        mean_avail_actions_count = np.array([
            get_batch_attr([ep], 'avail_actions')[0].sum(axis=-1).float().mean().item()
            for ep in episode_batch
        ]).mean()
        writer.add_scalar('mean_avail_actions_count', mean_avail_actions_count, t_env)

        if config.adaptive_epsilon:
            adaptive_epsilon = get_batch_attr([episode_batch[0]], 'epsilon')[0].mean().item()
            writer.add_scalar('epsilon', adaptive_epsilon, t_env)

        for ep in episode_batch:
            t_env += len(ep)
        if config.burn_in and not config.continuous_buffer:
            zero_elements = [
                get_zero_elements(config.buffer_sequence_size // 2, ep)
                for ep in episode_batch
            ]
            for i in range(len(zero_elements)):
                zero_elements[i].extend(episode_batch[i])
                episode_batch[i] = zero_elements[i]
        for ep in episode_batch:
            replay_buffer.add(ep)
        buffer_inserting_time += time.time() - buffer_inserting_start
        if replay_buffer.can_sample(config.batch_size):
            if steps_to_train is None:
                steps_to_train = steps_per_batch
            while steps_to_train >= steps_per_batch:
                steps_to_train -= steps_per_batch
                train_prepare_start = time.time()
                episode_sample = replay_buffer.sample(config.batch_size)

                extend_batch(episode_sample)

                if config.burn_in:
                    burn_in_sample = np.array([None] * config.batch_size, dtype=object)
                    train_sample = np.array([None] * config.batch_size, dtype=object)
                    for i in range(config.batch_size):
                        burn_in_sample[i] = episode_sample[i][:config.buffer_sequence_size // 2]
                        train_sample[i] = episode_sample[i][config.buffer_sequence_size // 2:]
                    train_prepare_time += time.time() - train_prepare_start
                    train_start = time.time()
                    with th.no_grad():
                        init_hidden_on_sample(mac, target_mac, burn_in_sample)
                    new_loss, new_grad_norm = train(
                        mac,
                        target_mac,
                        mixer,
                        target_mixer,
                        train_sample,
                        optimizer,
                        should_init_hidden=False,
                        start_action=get_batch_step_data(episode_sample, config.buffer_sequence_size//2 - 1)['actions_onehot']
                    )
                    train_time += time.time() - train_start
                else:
                    train_start = time.time()
                    new_loss, new_grad_norm = train(
                        mac,
                        target_mac,
                        mixer,
                        target_mixer,
                        episode_sample,
                        optimizer
                    )
                    train_time += time.time() - train_start
                loss += new_loss
                grad_norm += new_grad_norm
                trains_count += 1

                if not config.buffer_by_episodes:
                    if (t_env - last_target_update_step) / config.target_update_interval >= 1.0:
                        print('Updating target mac/mixer params')
                        target_mac.load_state_dict(mac.state_dict())
                        target_mixer.load_state_dict(mixer.state_dict())
                        last_target_update_step = t_env
                else:
                    if (episodes_count - last_target_update_step) / config.target_update_interval_episodes >= 1.0:
                        print('Updating target mac/mixer params')
                        target_mac.load_state_dict(mac.state_dict())
                        target_mixer.load_state_dict(mixer.state_dict())
                        last_target_update_step = episodes_count

        if (t_env - last_test_T) / config.test_interval >= 1.0:
            print()
            print(f'Current time: {datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')
            td = time.time() - start_time
            hours = int(td // 3600)
            minutes = int((td % 3600) // 60)
            seconds = int(td % 60)
            print(f'Time passed: {hours}:{minutes}:{seconds}')
            print(f'sampling_time: {sampling_time}')
            print(f'buffer_inserting_time: {buffer_inserting_time}')
            print(f'train_time: {train_time}')
            print(f'train_prepare_time: {train_prepare_time}')
            print(f'train_attrs_time: {train_attrs_time}')
            print(f'concat_time: {concat_time}')
            print(f'p_sending_time: {p_sending_time}')
            print(f'p_recieving_time: {p_recieving_time}')
            print(f'batch_appending_time: {batch_appending_time}')
            print(f'selecting_actions_time: {selecting_actions_time}')
            print(f'running_episodes_time: {running_episodes_time}')
            print(f't_env: {t_env} / {config.t_max}')
            print(f'Episodes passed: {episodes_count}')

            last_test_T = t_env

            test_counts = 0
            test_wins = 0
            test_dead_enemies = 0
            test_dead_allies = 0
            test_rewards = 0

            test_start = time.time()
            while test_counts < config.test_episodes_count:
                with th.no_grad():
                    episode_batch, env_info = run_one_episode(parent_conns, mac, t_env, testing=True)
                for i in range(len(env_info)):
                    # env might restart, in that case necessary info won't be returned
                    try:
                        if config.env_name == 'smac':
                            test_wins += env_info[i]['battle_won']
                            test_dead_enemies += env_info[i]['dead_enemies']
                            test_dead_allies += env_info[i]['dead_allies']
                        test_rewards += env_info[i]['episode_return']
                        test_counts += 1
                        test_mean_avail_actions_count = get_batch_attr([episode_batch[i]], 'avail_actions')[0].sum(axis=-1).float().mean().item()
                    except:
                        continue
            test_time += time.time() - test_start
            print(f'test_time: {test_time}')
            if config.env_name == 'smac':
                test_battle_won_mean = test_wins / test_counts
                test_winrate.append(test_battle_won_mean)
                test_dead_allies_mean = test_dead_allies / test_counts
                test_dead_enemies_mean = test_dead_enemies / test_counts
            test_return_mean = test_rewards / test_counts

            current_episodes = episodes_count - last_ep_count
            if config.env_name == 'smac':
                current_wins = wins_count - last_res
                current_winrate = (wins_count - last_res)/(episodes_count - last_ep_count)
                mean_dead_enemies = dead_enemies / current_episodes
                mean_dead_allies = dead_allies / current_episodes
            mean_rewards = rewards / current_episodes

            if config.env_name == 'smac':
                print(f'New winrate: {current_wins}/{current_episodes} = {round(current_winrate, 4)}')
                print(f'mean dead enemies: {round(mean_dead_enemies, 4)}')
                print(f'mean dead allies: {round(mean_dead_allies, 4)}')
            print(f'mean return: {round(mean_rewards, 4)}')
            if trains_count > 0:
                print(f'mean loss: {round(loss / trains_count, 4)}')
                print(f'mean grad norm: {round(grad_norm / trains_count, 4)}')

            if config.env_name == 'smac':
                print(f'Test winrate: {test_wins}/{test_counts} = {round(test_battle_won_mean, 4)}')
                print(f'Test mean dead enemies: {round(test_dead_enemies_mean, 4)}')
                print(f'Test mean dead allies: {round(test_dead_allies_mean, 4)}')
            print(f'Test mean return: {round(test_return_mean, 4)}')

            if config.env_name == 'smac':
                writer.add_scalar('battle_won_mean', current_winrate, t_env)
                writer.add_scalar('dead_enemies_mean', mean_dead_enemies, t_env)
                writer.add_scalar('dead_allies_mean', mean_dead_allies, t_env)
            writer.add_scalar('return_mean', mean_rewards, t_env)
            if trains_count > 0:
                writer.add_scalar('loss', loss / trains_count, t_env)
                writer.add_scalar('grad_norm', grad_norm / trains_count, t_env)

            if config.env_name == 'smac':
                writer.add_scalar('test_battle_won_mean', test_battle_won_mean, t_env)
                writer.add_scalar('test_dead_allies_mean', test_dead_allies_mean, t_env)
                writer.add_scalar('test_dead_enemies_mean', test_dead_enemies_mean, t_env)
            writer.add_scalar('test_return_mean', test_return_mean, t_env)

            writer.add_scalar('test_mean_avail_actions_count', test_mean_avail_actions_count, t_env)
            writer.add_scalar('episode', episodes_count, t_env)

            dead_enemies = 0
            dead_allies = 0
            rewards = 0
            loss = 0
            grad_norm = 0
            trains_count = 0

            last_res = wins_count
            last_ep_count = episodes_count

    # in some cases env might restart during testing and it won't
    # return test results
    eval_counts = 0
    eval_wins = 0
    eval_returns = 0

    while eval_counts < config.final_eval_episodes_count:
        _, env_info = run_one_episode(parent_conns, mac, t_env, testing=True)
        for i in range(len(env_info)):
            try:
                if config.env_name in ['pogema', 'pogema-follower']:
                    eval_returns += env_info[i]['episode_return']
                else:
                    eval_wins += env_info[i]['battle_won']
                eval_counts += 1
            except:
                continue

    print()
    if config.env_name == 'pogema':
        eval_return = eval_returns / eval_counts
        print(f'Eval return: {eval_returns}/{eval_counts} = {eval_return}')
        writer.add_scalar('eval_return', eval_return)
        return eval_return
    else:
        eval_winrate = eval_wins / eval_counts
        print(f'Eval winrate: {eval_wins}/{eval_counts} = {eval_winrate}')
        writer.add_scalar('eval_winrate', eval_winrate)
        return eval_winrate


if __name__ == '__main__':
    config = Config.load_and_parse_args()

    if config.env_name == 'smac':
        from smac.env import StarCraft2Env
        env_fn = lambda: StarCraft2Env(config.map_name)
    elif config.env_name == 'pogema':
        from pogema import pogema_v0, GridConfig
        size = 16
        num_agents = 16
        max_steps = 512
        env_fn = lambda: pogema_v0(GridConfig(integration="PyMARL",
                                              obs_radius=5,
                                              size=size,
                                              max_episode_steps=max_steps,
                                              num_agents=num_agents,
                                              on_target='restart'))
        config.map_name = f'pogema_size={size}_num_agents={num_agents}_max_steps={max_steps}'
    elif config.env_name == 'pogema-follower':
        from pogema import GridConfig
        from follower_env import FollowerPyMARL
        size = 16
        num_agents = 16
        max_steps = 512
        env_fn = lambda: FollowerPyMARL(GridConfig(integration="PyMARL",
                                                   obs_radius=5,
                                                   size=size,
                                                   max_episode_steps=max_steps,
                                                   num_agents=num_agents,
                                                   observation_type='POMAPF',
                                                   on_target='restart'))
        config.map_name = f'pogema_follower_size={size}_num_agents={num_agents}_max_steps={max_steps}'
    else:
        raise RuntimeError(f'Unknown environment: {config.env_name}')

    for folder in ['tensorboard', 'wandb', 'models', 'renders']:
        if not os.path.exists(folder):
            os.mkdir(folder)

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_device = 'cpu'

    parent_conns, worker_conns = zip(*[Pipe() for _ in range(config.envs_count)])

    processes = [
        Process(
            target=env_worker,
            args=(worker_conn, env_fn)
        )
        for worker_conn in worker_conns
    ]
    for process in processes:
        process.daemon = True
        process.start()

    # checking if sc2 envs are able to start; if not - just remove them from list
    parent_conns = list(parent_conns)
    worker_conns = list(worker_conns)
    for conn in parent_conns:
        conn.send(('reset', None))
    i = len(parent_conns)
    while i > 0:
        i -= 1
        should_remove = True
        try:
            if parent_conns[i].poll(timeout=20):
                parent_conns[i].recv()
                should_remove = False
        except:
            pass
        if should_remove:
            print(f'removing environment number {i}')
            parent_conns.pop(i)
            worker_conns.pop(i)
            processes.pop(i)
            config.envs_count -= 1

    if config.envs_count == 0:
        raise RuntimeError("Wasn't able to start any of SC2 environments")

    print('Successfully started environments')

    parent_conns[0].send(('get_env_info', None))
    env_info = parent_conns[0].recv()
    config.env_n_agents = env_info['n_agents']
    config.env_n_actions = env_info['n_actions']
    config.env_obs_size = env_info['obs_shape']
    config.env_state_size = env_info['state_shape']
    config.env_max_episode_length = env_info['episode_limit']

    try:
        if config.sweep:
            winrates = []
            for _ in range(config.sweep_runs_count):
                winrate = main()
                winrates.append(winrate)
            if not config.omit_wandb:
                wandb.log({'winrate': np.mean(winrates)})
        else:
            main()
    except Exception as e:
        print(f'Got an exception in main: {e}')
        raise e
    finally:
        for conn in parent_conns:
            conn.send(('close', None))
        for process in processes:
            try:
                process.terminate()
            except Exception as e:
                print(f'exception while terminating a process: {e}')
            try:
                time.sleep(0.5)
                process.close()
            except Exception as e:
                print(f'exception while closing a process: {e}')
