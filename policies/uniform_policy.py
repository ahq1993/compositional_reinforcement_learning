from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sac.policies.base import Policy2

import numpy as np


class UniformPolicy(Policy2, Serializable):
    """
    Fixed policy that randomly samples actions uniformly at random.

    Used for an initial exploration period instead of an undertrained policy.
    """
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        self._Da = env_spec.action_space.flat_dim

        super(UniformPolicy, self).__init__(env_spec)

    # Assumes action spaces are normalized to be the interval [-1, 1]
    @overrides
    def get_action(self, observation,sub_level_actions):
        return np.random.uniform(-1., 1., self._Da), None 
    '''@overrides 
    def get_action(self, observation,sub_level_actions):
        probs=np.random.uniform(0.0, 1., 4)
        probs=np.argmax(probs)#probs/sum(probs)
        #probs=np.array([1,0,0,0],dtype=np.float32)
        #probs=np.random.shuffle(probs)
        #actions_mean=np.sum(np.multiply(sub_level_actions[0],np.expand_dims(probs,2)),1)
        return sub_level_actions[0][0][probs], None 
        #return np.random.uniform(-1., 1., self._Da), None''' 

    @overrides
    def get_actions(self, observations,sub_level_actions):
        pass 

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        pass 

