from pydantic import BaseModel
import argparse
import os
import json


class Config(BaseModel):
    sweep: bool = False
    sweep_runs_count: int = 3
    wandb_project: str = 'nil'
    run_name: str = 'nil'

    env_name: str = 'smac'
    scale_rewards: bool = False

    envs_count: int = 8

    # Are set after an environment initialization
    env_n_agents: int = -1
    env_n_actions: int = -1
    env_obs_size: int = -1
    env_state_size: int = -1
    env_max_episode_length: int = -1

    omit_wandb: bool = False
    burn_in: bool = False
    adaptive_epsilon: bool = False
    uniform_exploration: bool = False

    map_name: str = 'corridor'

    grad_norm_clip: float = 10.0
    gamma: float = 0.99
    lr: float = 1e-3

    rnn_hidden_dim: int = 64

    t_max: int = 2_050_000
    epsilon_anneal_time: int = 50_000
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    min_epsilon: float = 0.05
    maxing_adaptive_epsilon: bool = False

    buffer_by_episodes: bool = False
    continuous_buffer: bool = False
    replay_buffer_episodes: int = 5000  # only used with buffer_by_episodes=True
    replay_buffer_steps: int = 400_000
    batch_size: int = 32
    buffer_sequence_size: int = 32
    train_frequency: float = 16.0

    test_interval: int = 10000
    test_episodes_count: int = 32
    target_update_interval: int = 20_000
    target_update_interval_episodes: int = 200 # only used with buffer_by_episodes=True
    update_alpha_interval: int = 5
    save_model_interval: int = 100_000
    final_eval_episodes_count: int = 512

    # Adaptive epsilon
    tanh_coef: float = 0.5

    def save(self):
        with open('config.json', 'w') as f:
            f.write(self.model_dump_json(indent=4))

    @staticmethod
    def load():
        with open('config.json', 'r') as f:
            d = json.loads(f.read())
            config = Config.parse_obj(d)
            return config

    @staticmethod
    def load_and_parse_args():
        if os.path.exists('config.json'):
            config = Config.load()
        else:
            config = Config()

        # translate pydantic types representation to that of argparse
        types = {
            'boolean': bool,
            'string': str,
            'number': float,
            'integer': int,
        }

        parser = argparse.ArgumentParser()
        for prop_name, prop in Config.schema()['properties'].items():
            _type = types[prop['type']]
            name = f'--{prop_name}'
            if _type == bool:
                parser.add_argument(name, action='store_true')
            else:
                parser.add_argument(name, type=_type)

        args, unknown = parser.parse_known_args()
        if len(unknown) > 0:
            raise RuntimeError(f'argparse: unrecognized arguments {unknown}')

        args_dict = vars(args)
        for prop_name, prop in Config.schema()['properties'].items():
            if prop['type'] == 'boolean':
                if not args_dict[prop_name]:
                    continue
            if args_dict[prop_name] is not None:
                setattr(config, prop_name, args_dict[prop_name])

        return config

    def update_from_dict(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


if __name__ == '__main__':
    # Just updates default config
    config = Config()
    config.save()
