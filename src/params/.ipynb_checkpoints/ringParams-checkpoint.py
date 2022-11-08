class ringParams:
    '''## Bulb ring neurons (input nodes)'''
    def __init__(self, n_input_elevation = 1):
        self.n_input_elevation = n_input_elevation
        self.n_input_azimuth = 32
        self.input_weight_id = 3 # initial synaptic weights
        self.n_input_nodes = self.n_input_elevation * self.n_input_azimuth