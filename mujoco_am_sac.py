import argparse
import os

import tensorflow as tf
import numpy as np
import joblib
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from rllab import config
from sac.misc import tf_utils
from sac.algos import SAC
from sac.envs import SimpleMazeAntEnv,RandomGoalAntEnv
from sac.environments.pusher import PusherEnv
from sac.envs import (
    GymEnv,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv,
    CrossMazeAntEnv,
)
from sac.envs import HalfCheetahHurdleEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import  UniformPolicy,GaussianPtrPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants
import pickle
ENVIRONMENTS = {
    'ant-cross-maze': {
        'default': CrossMazeAntEnv
    },
    'ant-random-goal': {
        'default': RandomGoalAntEnv
    },
    'cheetah-hurdle': {
        'default': HalfCheetahHurdleEnv
    },
    'pusher': {
        'default': PusherEnv
    }
}

DEFAULT_DOMAIN = DEFAULT_ENV = 'ant-cross-maze'
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default='ant-cross-maze')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian','gaussian_ptr'),
                        default='gaussian_ptr')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args

def load_low_level_policy(policy_path=None,name=None):
    with tf_utils.get_default_session().as_default():
        with tf.variable_scope(name, reuse=False):
            snapshot = joblib.load(policy_path)

    policy = snapshot["policy"]

    return policy


args = parse_args()
def run_experiment(variant):
    domain =None
    goal_size=None
    sub_level_policies_paths=[]
    if args.domain=='ant-cross-maze':
        domain=CrossMazeAntEnv
        goal_size=2
        sub_level_policies_paths.append("primitive-policies/ant/fwrd/fwrd.pkl")
        sub_level_policies_paths.append("primitive-policies/ant/bwrd/bwrd.pkl")
        sub_level_policies_paths.append("primitive-policies/ant/uwrd/uwrd.pkl")
        sub_level_policies_paths.append("primitive-policies/ant/dwrd/dwrd.pkl")
    elif args.domain=='ant-random-goal':
        domain=RandomGoalAntEnv
        goal_size=2
        sub_level_policies_paths.append("primitive-policies/ant/fwrd/fwrd.pkl")
        sub_level_policies_paths.append("primitive-policies/ant/bwrd/bwrd.pkl")
        sub_level_policies_paths.append("primitive-policies/ant/uwrd/uwrd.pkl")
        sub_level_policies_paths.append("primitive-policies/ant/dwrd/dwrd.pkl")
    elif args.domain=='cheetah-hurdle':
        domain=HalfCheetahHurdleEnv
        goal_size=2
        sub_level_policies_paths.append("primitive-policies/hc/fwd/fwd.pkl")
        sub_level_policies_paths.append("primitive-policies/hc/jp-longz/jump.pkl")
    elif args.domain=='pusher':
        domain=PusherEnv
        goal_size=0
        sub_level_policies_paths.append("primitive-policies/pusher/bottom/bottom.pkl")
        sub_level_policies_paths.append("primitive-policies/pusher/left/left.pkl")




    env = normalize(domain())#CrossMazeAntEnv())

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=1e6,
        seq_len=len(sub_level_policies_paths),
    )

    sampler = SimpleSampler(
        max_path_length=1000,
        min_pool_size=1000,
        batch_size=256

    )



    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=5e3,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
        sampler=sampler
    )


    M = 128
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)
    policy=GaussianPtrPolicy(env_spec=env.spec,hidden_layer_sizes=(M,M),reparameterize=True,reg=1e-3,)

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        g=goal_size,
        policy=policy,
        sub_level_policies_paths=sub_level_policies_paths,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        lr=3e-4,
        scale_reward=5,
        discount=0.99,
        tau=0.005,
        reparameterize=True,
        target_update_interval=1,
        action_prior='uniform',
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    algorithm.train()


def launch_experiments(args):

    num_experiments =5
    print('Launching {} experiments.'.format(num_experiments))
    i=0
    if i==0:
        print("Experiment: {}/{}".format(i, num_experiments))
        experiment_prefix = 'ant/cross-maze' + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix='ant/cross-maze', exp_name=args.exp_name, i=0)

        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode='gap',
            snapshot_gap=1000,
            sync_s3_pkl=True,
        )


def main():
    #args = parse_args()
    launch_experiments(args)


if __name__ == '__main__':
    main()
