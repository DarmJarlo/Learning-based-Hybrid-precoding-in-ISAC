
'''
Design approach:
1.give a initial precoding matrix corresponding to the initial state
2.state:defined as previous 10 times CRB_d, CRB_angle,sumrate, estiamted_velocity,estimated_angle
3.reward: defined as CRB_d,CRB_angle,sumrate
4. action:so far a very large action space.
consider analog part exp**(j*theta) theta from 0 to 2*pi

digital part a*exp**(j*theta) a from 0-1  theta from 0 to 2*pi
as the precoding matrix has around 400 elements,the action space can be over 10000
'''

import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as kls
from tensorflow.python.keras.backend import dtype, shape
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp

class RSU:
    def __init__(self):
        self.DistanceToLane = 500
        self.DistanceToVehicle = []
        self.precoding_matrix = np.zeros([8,8])#precoding_matrix : Nt*K
        self.location = (0,0)
        self.obs = np.zeros([])

    def init_NNs(self, action_space):
        self.network = CombinedNN(action_space)
        self.network.compile(self.optimizer)
    def observation(self,):
    def act(self, obs, action_space):

    def append_last_ten_obs(self, obs):
        self.last_two_obs.pop(0)
        self.last_two_obs.append(obs)