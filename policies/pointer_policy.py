""" Gaussian pointer policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from sac.distributions import Normal 
from sac.policies import NNPolicy2
from sac.misc import tf_utils
from rllab.misc import tensor_utils
import time
dis=tf.contrib.distributions
EPS = 1e-6

def linear(X, dout, name, bias=True):
	with tf.variable_scope(name):
		dX = int(X.get_shape()[-1])
		W = tf.get_variable('W', shape=(dX, dout))
		if bias:
			b = tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32)))
		else:
			b = 0
	return tf.matmul(X, W)+b

def relu_layer(X, dout, name):
	return tf.nn.relu(linear(X, dout, name))

def decoder(x, layers=2, d_hidden=32):
	out = x
	for i in range(layers):
		out = relu_layer(out, dout=d_hidden, name='l%d'%i)
	return out

class GaussianPtrPolicy(NNPolicy2, Serializable):
	def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3, squash=True, reparameterize=False, name='gauss_ptrpolicy'):


		Serializable.quick_init(self, locals())

		self._hidden_layers = hidden_layer_sizes
		self._Da = env_spec.action_space.flat_dim
		self._Ds = env_spec.observation_space.flat_dim
		self._is_deterministic = False
		self._squash = squash
		self._reparameterize = reparameterize
		self._reg = reg
		self.name=name

		self._scope_name = (tf.get_variable_scope().name + "/" + name).lstrip("/")
		self.n_hiddens=hidden_layer_sizes[0]
		self.initializer=tf.contrib.layers.xavier_initializer()
		self.build()

		super(NNPolicy2, self).__init__(env_spec)

	def actions_for(self, observations, sub_level_actions,name=None, reuse=tf.AUTO_REUSE,with_log_pis=False, regularize=False):
		name = name or self.name
		#feed_dict = {self._observations_ph: observations,self.sub_level_actions: sub_level_actions}
		#raw_actions,log_p_t=tf_utils.get_default_session().run([self.distribution.x_t,self.distribution.log_p_t], feed_dict=feed_dict)

		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			lstm_fw_cell=tf.nn.rnn_cell.LSTMCell(self.n_hiddens,initializer=self.initializer)
			lstm_bw_cell=tf.nn.rnn_cell.LSTMCell(self.n_hiddens,initializer=self.initializer)
			action_encodings,last_hidden=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,sub_level_actions,dtype=tf.float32)#ae=(2,B,seq,N_hidden)

			f_encoding=tf.transpose(action_encodings[0],perm=[1,0,2])
			b_encoding=tf.transpose(action_encodings[1],perm=[1,0,2])
			input_decoder=tf.concat([observations,f_encoding[-1],b_encoding[-1]],axis=1)
			decoder_output=decoder(input_decoder,layers=2,d_hidden=self.n_hiddens)


			Wref_f=tf.get_variable("Wref_f",[1,self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			Wref_b=tf.get_variable("Wref_b",[1,self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			Wd=tf.get_variable("Wd",[self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			v=tf.get_variable("v",[self.n_hiddens], initializer=self.initializer)

			We_f=tf.nn.conv1d(action_encodings[0],Wref_f,1,"VALID", name="We_f")
			We_b=tf.nn.conv1d(action_encodings[1],Wref_b,1,"VALID", name="We_b")
			Wd=tf.expand_dims(tf.matmul(decoder_output,Wd, name="Wd"),1)			
			scores=tf.reduce_sum(v*tf.tanh(We_f+We_b+Wd),[-1],name="scores") #BatchXseq_len

			sm_scores=tf.nn.softmax(scores/0.5,name="sm_scores") #BatchXseq_len
			scores_index=tf.argmax(sm_scores,axis=1)
			#one_hot=tf.one_hot(scores_index,tf.shape(sub_level_actions)[1])	

		actions =tf.reduce_sum(tf.multiply(sub_level_actions,tf.expand_dims(sm_scores,2)),1)
		raw_actions =tf.reduce_sum(tf.multiply(sub_level_actions,tf.expand_dims(sm_scores,2)),1)
		# TODO: should always return same shape out
		# Figure out how to make the interface for `log_pis` cleaner
		if with_log_pis:
			# TODO.code_consolidation: should come from log_pis_for
			return actions,sm_scores,self._squash_correction(raw_actions)

		return actions

	def build(self):
		self._observations_ph= tf.placeholder(tf.float32,(None,self._Ds),name='observations',)
		self.sub_level_actions= tf.placeholder(tf.float32,(None,None,self._Da),name='sub_level_actions',)
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

			self.lstm_fw_cell=tf.nn.rnn_cell.LSTMCell(self.n_hiddens,initializer=self.initializer)
			self.lstm_bw_cell=tf.nn.rnn_cell.LSTMCell(self.n_hiddens,initializer=self.initializer)
			self.action_encodings,self.last_hidden=tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell,self.lstm_bw_cell,self.sub_level_actions,dtype=tf.float32)#ae=(2,B,seq,N_hidden)

			self.f_encoding=tf.transpose(self.action_encodings[0],perm=[1,0,2])
			self.b_encoding=tf.transpose(self.action_encodings[1],perm=[1,0,2])
			self.input_decoder=tf.concat([self._observations_ph,self.f_encoding[-1],self.b_encoding[-1]],axis=1)
			self.decoder_output=decoder(self.input_decoder,layers=2,d_hidden=self.n_hiddens)

			self.Wref_f=tf.get_variable("Wref_f",[1,self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			self.Wref_b=tf.get_variable("Wref_b",[1,self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			self.Wd=tf.get_variable("Wd",[self.n_hiddens,self.n_hiddens], initializer=self.initializer)
			self.v=tf.get_variable("v",[self.n_hiddens], initializer=self.initializer)

			self.We_f=tf.nn.conv1d(self.action_encodings[0],self.Wref_f,1,"VALID", name="We_f")
			self.We_b=tf.nn.conv1d(self.action_encodings[1],self.Wref_b,1,"VALID", name="We_b")
			self.Wd=tf.expand_dims(tf.matmul(self.decoder_output,self.Wd, name="Wd"),1)#linear(decoder_output, dout=hidden_size, name="Wd")			
			self.scores=tf.reduce_sum(self.v*tf.tanh(self.We_f+self.We_b+self.Wd),[-1],name="scores") #BatchXseq_len

			self.sm_scores=tf.nn.softmax(self.scores/0.5,name="sm_scores") #BatchXseq_len
			self.scores_index=tf.argmax(self.sm_scores,axis=1)	
			self.one_hot=tf.one_hot(self.scores_index,tf.shape(self.sub_level_actions)[1])
		self._actions =tf.reduce_sum(tf.multiply(self.sub_level_actions,tf.expand_dims(self.sm_scores,2)),1)
	@overrides
	def get_actions(self, observations,sub_level_actions):
		"""Sample actions based on the observations.

		If `self._is_deterministic` is True, returns the mean action for the 
		observations. If False, return stochastically sampled action.

		TODO.code_consolidation: This should be somewhat similar with
		`LatentSpacePolicy.get_actions`.
		"""
		feed_dict = {self._observations_ph: observations,self.sub_level_actions: sub_level_actions}
		if self._is_deterministic: # Handle the deterministic case separately
			mu = tf.get_default_session().run(self._actions, feed_dict)  # 1 x Da

			return mu
		return super(GaussianPtrPolicy, self).get_actions(observations,sub_level_actions) 

	def _squash_correction(self, actions):
		if not self._squash: return 0
		return tf.reduce_sum(tf.log(1 - actions ** 2 + EPS), axis=1)

	@contextmanager
	def deterministic(self, set_deterministic=True, latent=None):
		"""Context manager for changing the determinism of the policy.

		See `self.get_action` for further information about the effect of
		self._is_deterministic.

		Args:
			set_deterministic (`bool`): Value to set the self._is_deterministic
				to during the context. The value will be reset back to the
				previous value when the context exits.
			latent (`Number`): Value to set the latent variable to over the
				deterministic context.
		"""
		was_deterministic = self._is_deterministic

		self._is_deterministic = set_deterministic

		yield

		self._is_deterministic = was_deterministic


