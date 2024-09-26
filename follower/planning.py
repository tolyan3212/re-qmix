from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner

from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PlannerConfig(BaseModel):
    use_precalc_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True


INF = 1000000000


class Planner:
    def __init__(self, cfg: PlannerConfig):

        gc: GridConfig = GridConfig()

        self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}
        self.steps = 0
        self.planner = None
        self.previous_positions = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None

    def update(self, obs):
        num_agents = len(obs)
        obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.planner = [planner(self.obstacles, self.cfg.use_precalc_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(num_agents)]
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])
            if self.cfg.use_precalc_cost:
                pen_calc = planner(self.obstacles, self.cfg.use_precalc_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost)
                penalties = pen_calc.precompute_penalty_matrix(obs_radius)
                for p in self.planner:
                    p.set_penalties(penalties)
        if self.previous_positions is None:
            self.previous_positions = [[] for _ in range(num_agents)]

        action = []
        for k in range(num_agents):
            self.previous_positions[k].append(obs[k]['xy'])
            if obs[k]['xy'] == obs[k]['target_xy']:
                action.append(None)
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            obs[k]['agents'][obs_radius][obs_radius] = 1
            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
            path = self.planner[k].get_next_node()
            if path is not None and path[1][0] < INF:
                action.append(self.actions[(path[1][0] - path[0][0], path[1][1] - path[0][1])])
            else:
                action.append(None)
        self.steps += 1
        return action

    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path())
        return results


class ResettablePlanner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._agent = None

    def update(self, observations):
        return self._agent.update(observations)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = Planner(self._cfg)