import numpy as np

def inputWeights(bumpParams, ringParams, input_weight_type):

    W_max = 0.33;
    
    if input_weight_type == 'zero weight':
        W_input = np.zeros((bumpParams.n_wedge_neurons, ringParams.n_input_nodes))
              
    if input_weight_type == 'von Mises weight':
        d = 2*np.pi/ringParams.n_input_nodes
        kappa = 3        
        W_input = np.zeros((bumpParams.n_wedge_neurons, ringParams.n_input_nodes))
        for i in range(W_input.shape[0]):
            W_input[i,:] = circularPdfVonMises(np.arange(0,2*np.pi-0.001,2*np.pi/ringParams.n_input_nodes),
                                               2*np.pi/bumpParams.n_wedge_neurons*i,
                                               kappa,
                                               'radian')  
        W_input = W_input/np.max(W_input)*W_max
             
    if input_weight_type ==  'random weight':
        W_input = np.random.rand(bumpParams.n_wedge_neurons, ringParams.n_input_nodes)*W_max
                           
    return W_input
