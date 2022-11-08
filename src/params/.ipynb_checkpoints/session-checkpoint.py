import numpy as np
import sys
sys.path.insert(0, 'utils')
from params import plasticityParams, bumpParams, W_ring_attractor, ringParams, inputWeights, visual_input
from vel_calc import vel_calc

class session:
    '''## Details of the simulation session'''
    def __init__(self, learning_rule, sim_cond, add_noise, use_2D_input, pos_data, n_input_elevation = 1, simulation_dt = 0.01, membrane_saturation = 1000, membrane_threshold = 0):
        self.dt = simulation_dt
        self.learning_rule = learning_rule
        self.sim_cond = sim_cond
        self.add_noise = add_noise
        self.bumpParams = bumpParams()
        self.W_ring_attractor = W_ring_attractor(self)
        self.membrane_saturation = membrane_saturation
        self.membrane_threshold = membrane_threshold
        self.ringParams = ringParams(n_input_elevation = n_input_elevation)
        self.W_input = inputWeights(self.bumpParams, self.ringParams, 'von Mises weight') # Initialize the synaptic weight matrix W
        self.wedge_input = np.random.rand(self.bumpParams.n_wedge_neurons,1) * self.bumpParams.A # Randomly initialize the compass neurons (= wedge neurons) activity level
        [self.t, self.vel, self.xpos_radian] = vel_calc(pos_data, simulation_dt)
        if sim_cond == 'no input':
            self.vel = np.zeros(len(self.t))
        self.plasticityParams = plasticityParams(learning_rule, use_2D_input, self.vel)
        self.sim_visual_neurons = visual_input(self)
        self.wedge_current_injection = np.zeros((self.bumpParams.n_wedge_neurons, len(self.t))) # can be modified later if optogenic input is desired