import gymnasium
from gymnasium import ObservationWrapper
from pogema import GridConfig
from pogema.envs import _make_pogema
from pogema.integrations.pymarl import PyMarlPogema
from pydantic import BaseModel

from follower.preprocessing import FollowerWrapper, ConcatPositionalFeatures


class PlannerConfig(BaseModel):
    use_precalc_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True


class PreprocessorConfig(PlannerConfig):
    intrinsic_target_reward: float = 0.01


class ProvideGlobalObstacles(gymnasium.Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()


class FixObservation(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space['obs']

    def observation(self, observations):
        result = []
        for obs in observations:
            result.append(obs['obs'])
        return result


def make_follower_pogema(grid_config):
    env = _make_pogema(grid_config)
    env = ProvideGlobalObstacles(env)
    env = FollowerWrapper(env=env, config=PreprocessorConfig())
    env = ConcatPositionalFeatures(env)
    env = FixObservation(env)
    return env


# noinspection PyMissingConstructor
class FollowerPyMARL(PyMarlPogema):
    def __init__(self, grid_config, mh_distance=False):
        gc = grid_config
        self._grid_config: GridConfig = gc

        self.env = make_follower_pogema(grid_config=gc)
        self._mh_distance = mh_distance
        self._observations, _ = self.env.reset()
        self.max_episode_steps = gc.max_episode_steps
        self.episode_limit = gc.max_episode_steps
        self.n_agents = self.env.get_num_agents()

        self.spec = None

    def step(self, actions):
        self._observations, rewards, terminated, truncated, infos = self.env.step(actions)
        info = {}
        done = all(terminated) or all(truncated)
        if done:
            for key, value in infos[0]['metrics'].items():
                info[key] = value
            info.setdefault('episode_limit', 1.0 if all(truncated) else 0.0)

        return sum(rewards), done, info


def main():
    env = FollowerPyMARL(GridConfig(obs_radius=5,
                              size=16,
                              max_episode_steps=128,
                              integration='PyMARL',
                              num_agents=2,
                              observation_type='POMAPF',
                              on_target='restart'))
    obs = env.reset()
    env.render()
    env.step([3, 2])
    env.render()


if __name__ == '__main__':
    main()
