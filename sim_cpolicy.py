import argparse
import joblib
from rllab.misc import tensor_utils
import time
from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from sac.envs import CrossMazeAntEnv, RandomGoalAntEnv,HalfCheetahHurdleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc import tensor_utils
from sac.misc import tf_utils

def rollout(env, policy,sub_level_policies,path_length=1000, render=True, speedup=10, g=2):
	observation = env.reset()
	policy.reset()

	t = 0
	obs = observation
	for t in range(path_length):


		sub_level_actions=[]
		if g!=0:
			obs=observation[:-g]
		else:
			obs=observation
		for i in range(0,len(sub_level_policies)):
			action, _ = sub_level_policies[i].get_action(obs)
			sub_level_actions.append(action.reshape(1,-1))
		sub_level_actions=np.stack(sub_level_actions,axis=0)
		sub_level_actions=np.transpose(sub_level_actions,(1,0,2))

		action, agent_info = policy.get_action(observation,sub_level_actions)
		next_obs, reward, terminal, env_info = env.step(action)


		observation = next_obs

		if render:
			env.render()
			time_step = 0.05
			time.sleep(time_step / speedup)

		if terminal:
			break


	return 0


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('file', type=str, help='Path to the snapshot file.')
	parser.add_argument('--max-path-length', '-l', type=int, default=1000)
	parser.add_argument('--speedup', '-s', type=float, default=10)
	parser.add_argument('--domain',type=str,default='ant-cross-maze')
	parser.add_argument('--deterministic', '-d', dest='deterministic',
		                action='store_true')
	parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
		                action='store_false')
	parser.add_argument('--policy_h', type=int)
	parser.set_defaults(deterministic=True)

	args = parser.parse_args()

	return args

def load_low_level_policy(policy_path=None,name=None):
	with tf_utils.get_default_session().as_default():
		with tf.variable_scope(name, reuse=False):
			snapshot = joblib.load(policy_path)

	policy = snapshot["policy"]
	return policy


def simulate_policy_ant(args):
	sub_level_policies=[]
	with tf.Session() as sess:
		with tf.variable_scope("fwrd", reuse=False):
			fwrd = joblib.load("primitive-policies/ant/fwrd/fwrd.pkl")
		with tf.variable_scope("bwrd", reuse=False):
			bwrd = joblib.load("primitive-policies/ant/bwrd/bwrd.pkl")
		with tf.variable_scope("uwrd", reuse=False):
			uwrd = joblib.load("primitive-policies/ant/uwrd/uwrd.pkl")
		with tf.variable_scope("dwrd", reuse=False):
			dwrd = joblib.load("primitive-policies/ant/dwrd/dwrd.pkl")
		sub_level_policies.append(fwrd["policy"])
		sub_level_policies.append(bwrd["policy"])
		sub_level_policies.append(uwrd["policy"])
		sub_level_policies.append(dwrd["policy"])
		data = joblib.load(args.file)
		if 'algo' in data.keys():
			policy = data['algo'].policy
			env = data['algo'].env
		else:
			policy = data['policy']
			env = data['env']
		with policy.deterministic(args.deterministic):
			while True:
				path = rollout(env, policy,sub_level_policies,path_length=args.max_path_length,g=2)

def simulate_policy_pusher(args):
	sub_level_policies=[]
	with tf.Session() as sess:
		with tf.variable_scope("bottom", reuse=False):
			btm = joblib.load("primitive-policies/pusher/bottom/bottom.pkl")
		with tf.variable_scope("jump", reuse=False):
			lft = joblib.load("primitive-policies/pusher/left/left.pkl")
		sub_level_policies.append(btm["policy"])
		sub_level_policies.append(lft["policy"])
		data = joblib.load(args.file)
		if 'algo' in data.keys():
			policy = data['algo'].policy
			env = data['algo'].env
		else:
			policy = data['policy']
			env =data['env']
		with policy.deterministic(args.deterministic):
			while True:
				path = rollout(env, policy,sub_level_policies,path_length=args.max_path_length,g=0)

def simulate_policy_hch(args):
	sub_level_policies=[]
	with tf.Session() as sess:
		with tf.variable_scope("fwrd", reuse=False):
			fwrd = joblib.load("primitive-policies/hc/fwd/fwd.pkl")
		with tf.variable_scope("jump", reuse=False):
			jmp = joblib.load("primitive-policies/hc/jp-longz/jump.pkl")
		sub_level_policies.append(fwrd["policy"])
		sub_level_policies.append(jmp["policy"])
		data = joblib.load(args.file)
		if 'algo' in data.keys():
			policy = data['algo'].policy
			env = data['algo'].env
		else:
			policy = data['policy']
			env = normalize(HalfCheetahHurdleEnv()) #data['env']
		with policy.deterministic(args.deterministic):
			while True:
				path = rollout(env, policy,sub_level_policies,path_length=args.max_path_length, g=2)

if __name__ == "__main__":
	args = parse_args()
	if args.domain=='ant-cross-maze' or args.domain=='ant-random-goal':
		simulate_policy_ant(args)
	if args.domain=='cheetah-hurdle':
		simulate_policy_hch(args)
	if args.domain=='pusher':
		simulate_policy_pusher(args)
