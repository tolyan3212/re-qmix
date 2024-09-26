import json
from argparse import Namespace

import numpy as np
import yaml
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg

from eval_toolbox import toolbox_registry
from eval_toolbox.evaluator import evaluation

from sample_factory.train import make_runner
from sample_factory.utils.utils import experiment_dir
from sample_factory.algo.utils.misc import ExperimentStatus

import wandb

from follower.training_config import Experiment

from sample_factory.utils.utils import log

from follower.register_env import register_custom_components
from utils.register_eval import register_all
from follower.register_training_utils import register_custom_model, register_msg_handlers


def calculate_metric(evaluation_config, path_to_weights, target_metric='avg_throughput', ):
    register_all()

    log.debug('path_to_weights: ' + path_to_weights)
    name = 'Follower'
    if path_to_weights is not None:
        evaluation_config['algorithms'][name]['path_to_weights'] = path_to_weights
    algo = toolbox_registry.make(name, **evaluation_config['algorithms'][name])
    model_parameters = algo.get_model_parameters()
    evaluation_results = evaluation(evaluation_config)
    results = []
    for eval_run in evaluation_results:
        results.append(eval_run['metrics'][target_metric])

    return {f'eval/{target_metric}': np.mean(results), f'eval/model_parameters': model_parameters}


def create_sf_config(exp: Experiment):
    # creating sample_factory config
    custom_argv = [f'--env={exp.env}']
    # print(custom_argv), exit(0)
    parser, partial_cfg = parse_sf_args(argv=custom_argv, evaluation=False)
    parser.set_defaults(**exp.dict())
    final_cfg = parse_full_cfg(parser, argv=custom_argv)
    return final_cfg


def run(config=None):
    register_custom_model()

    if config is None:
        import argparse

        parser = argparse.ArgumentParser(description='Process training config.')

        parser.add_argument('--config_path', type=str, action="store", default='train-debug.yaml',
                            help='path to yaml file with single run configuration', required=False)

        parser.add_argument('--raw_config', type=str, action='store',
                            help='raw json config', required=False)

        parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                            help='Run wandb in thread mode. Usefull for some setups.', required=False)

        params = parser.parse_args()
        if params.raw_config:
            params.raw_config = params.raw_config.replace("\'", "\"")
            config = json.loads(params.raw_config)
        else:
            if params.config_path is None:
                raise ValueError("You should specify --config_path or --raw_config argument!")
            with open(params.config_path, "r") as f:
                config = yaml.safe_load(f)
    else:
        params = Namespace(**config)
        params.wandb_thread_mode = False

    exp = Experiment(**config)
    flat_config = Namespace(**exp.dict())
    env_name = exp.environment.env
    log.debug(f'env_name = {env_name}')
    register_custom_components(env_name)

    log.info(flat_config)

    if exp.train_for_env_steps == 1_000_000:
        exp.use_wandb = False

    if exp.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project='Learn-to-Follow', config=exp.dict(), save_code=False, sync_tensorboard=True,
                   anonymous="allow", job_type=exp.environment.env, group='train')

    flat_config, runner = make_runner(create_sf_config(exp))
    register_msg_handlers(flat_config, runner)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    with open('experiments/validation-mazes.yaml') as f:
        evaluation_config = yaml.safe_load(f)
    eval_metrics = calculate_metric(evaluation_config=evaluation_config,
                                    path_to_weights=experiment_dir(cfg=flat_config))
    if exp.use_wandb:
        import shutil
        path = experiment_dir(cfg=flat_config)
        zip_name = str(path)
        shutil.make_archive(zip_name, 'zip', path)
        wandb.save(zip_name + '.zip')

        wandb.log(eval_metrics)
        wandb.finish()

    return status


def main():
    print(calculate_metric(path_to_weights='experiments/follower-40k'))


if __name__ == '__main__':
    main()
