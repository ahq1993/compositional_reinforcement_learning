"""Implements a ant which is sparsely rewarded for reaching a goal"""
#from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
#from gym.envs.mujoco.mujoco_env import MujocoEnv


from rllab.core.serializable import Serializable
from sac.misc.utils import PROJECT_PATH
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.envs.base import Step
from gym import utils
import os
import numpy as np

MODELS_PATH = os.path.abspath(os.path.join(PROJECT_PATH, 'sac/mujoco_models'))

class HalfCheetahHurdleEnv(HalfCheetahEnv):
	def __init__(self):
		self.exteroceptive_observation =[12.0,0,0.5]
		self.hurdles_xpos=[-15.,-13.,-9.,-5.,-1.,3.,7.,11.,15.]#,19.,23.,27.]
		path = os.path.join(MODELS_PATH, 'half_cheetah_hurdle.xml')
		MujocoEnv.__init__(self,file_path=path)
		#MujocoEnv.__init__(self)
		Serializable.quick_init(self, locals())

	def get_current_obs(self):
		proprioceptive_observation = super().get_current_obs()
		x_pos1 =self.get_body_com('ffoot')[0]#self.model.data.qpos.flat[:1]
		x_pos2 =self.get_body_com('bfoot')[0]#self.model.data.qpos.flat[:1]
		matches = [x for x in self.hurdles_xpos if x >= x_pos2]
		next_hurdle_x_pos = [matches[0]]
		ff_dist_frm_next_hurdle=[np.linalg.norm(matches[0] - x_pos1)]
		bf_dist_frm_next_hurdle=[np.linalg.norm(matches[0] - x_pos2)]
		observation =np.concatenate([proprioceptive_observation,next_hurdle_x_pos,bf_dist_frm_next_hurdle]).reshape(-1)
		return observation

	def isincollision(self):
		hurdle_size=[0.05,1.0,0.03]
		x_pos =self.get_body_com('ffoot')[0]#self.model.data.qpos.flat[:1]
		matches = [x for x in self.hurdles_xpos if x >= x_pos]
		if len(matches)==0:
			return False
		hurdle_pos =[matches[0],0.0,0.20]
		#names=['fthigh','bthigh']
		#names=['torso','bthigh','bshin','bfoot']
		names=['ffoot']
		xyz_pos=[]
		for i in range(0,len(names)):
			xyz_pos.append(self.get_body_com(names[i]))
		for i in range(0,len(names)):
			#xyz_position = self.get_body_com(names[i])
			cf=True
			for j in range(0,1):
				if abs(hurdle_pos[j]-xyz_pos[i][j])>1.5*hurdle_size[j]:
					cf=False
					break
			if cf:
				return True
		return False

	def get_hurdle_reward(self):
		hurdle_size=[0.05,1.0,0.03]
		x_pos =self.get_body_com('bfoot')[0]#self.model.data.qpos.flat[:1]
		matches = [x for x in self.hurdles_xpos if x >= x_pos]
		hurdle_reward =-1.0*len(matches)

		return hurdle_reward

	def step(self, action):
		xyz_pos_before = self.get_body_com('bshin')
		self.forward_dynamics(action)
		xyz_pos_after = self.get_body_com('bshin')
		xyz_position = self.get_body_com('torso')
		jump_reward = np.abs(self.get_body_comvel("torso")[2])
		run_reward = self.get_body_comvel("torso")[0]
		next_obs= self.get_current_obs()
		if self.isincollision():# or (xyz_pos_after[0]-xyz_pos_before[0])<-0.01:#dist_from_hurdle < 1 and dist_from_hurdle > 0.3 and z_after<0.05:(xyz_pos_after[0]-xyz_pos_before[0])<-0.01: #
			collision_penality=-2.0
			#print("collision")
		else:
			collision_penality=0.0
			#print("not collisions")
		hurdle_reward = self.get_hurdle_reward()
		#print(hurdle_reward)
		done = False
		goal_reward=0
		goal_distance =np.linalg.norm(xyz_position - self.exteroceptive_observation)
		if (goal_distance)<1.0:
			done=True
			goal_reward=1000
		else:
			done=False

		reward=-1e-1*goal_distance+hurdle_reward+goal_reward+run_reward+3e-1*jump_reward+collision_penality#1e-1*goal_distance+run_reward+jump_reward+collision_penality
		info = {'goal_distance': goal_distance}
		return Step(next_obs, reward, done, **info)
