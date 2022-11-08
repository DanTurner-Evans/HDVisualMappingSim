import numpy as np
from mat4py import loadmat
import cv2
import sys
sys.path.insert(0, 'utils')
from circularPdfVonMises import circularPdfVonMises
from vel_calc import vel_calc
from moving_avg import moving_avg

class session:
    '''## Details of the simulation session'''
    def __init__(self, learning_rule, sim_cond, input_weight, add_noise, pos_data, n_input_elevation = 1, simulation_dt = 0.01, membrane_saturation = 1000, membrane_threshold = 0):
        self.dt = simulation_dt
        self.learning_rule = learning_rule
        self.sim_cond = sim_cond
        self.add_noise = add_noise
        self.bumpParams = bumpParams()
        self.W_ring_attractor = W_ring_attractor(self)
        self.membrane_saturation = membrane_saturation
        self.membrane_threshold = membrane_threshold
        self.ringParams = ringParams(n_input_elevation = n_input_elevation)
        self.W_input = inputWeights(self.bumpParams, self.ringParams, input_weight) # Initialize the synaptic weight matrix W
        self.wedge_input = np.random.rand(self.bumpParams.n_wedge_neurons,1) * self.bumpParams.A # Randomly initialize the compass neurons (= wedge neurons) activity level
        [self.t, self.vel, self.xpos_radian] = vel_calc(pos_data, simulation_dt, add_noise)
        if sim_cond == 'no input':
            self.vel = np.zeros(len(self.t))
        if ('2D' in sim_cond):
            self.plasticityParams = plasticityParams(learning_rule, True, self.vel)
        else:
            self.plasticityParams = plasticityParams(learning_rule, False, self.vel)
        self.sim_visual_neurons = visual_input(self)
        self.wedge_current_injection = np.zeros((self.bumpParams.n_wedge_neurons, len(self.t))) # can be modified later if optogenic input is desired
        
        
        

class bumpParams:
    ''' E-PG neurons (compass neurons), ring attractor
    The ring attractor parameters for simulation 
    tau and D affect the bump speed.
    Larger D or largaer tau mean slower flow
    beta_discrete has minimal to no impact on the speed of the bump.
    '''
    def __init__(self):
        self.n_wedge_neurons = 32
        self.tau_wedge = 0.05*np.ones(self.n_wedge_neurons)
        self.D_cont = 0.2
        self.beta_cont = 10
        self.bump_width = np.pi/2*1.2 # bump_width: tail to tail, For most simulations. Half width is around pi/2.
        [self.D, self.alpha_, self. beta_discrete, self.A] = self.DiscreteRingSolution()
      
    # Convert continuous parameters into discrete parameters and check the stability of parameters
    def ContRingParamToDiscrete(self):
        m = self.n_wedge_neurons/2/np.pi*self.bump_width # Number of active wedges
        c2d_scalar = 2*np.pi/self.n_wedge_neurons
        D = self.D_cont  /  c2d_scalar**2
        alpha_ = (np.sin(np.pi/(m-0.5+2))*2)**2*D+1 #-0.5 is to ensure that the m is between m-0.5 and m+0.5
        beta_discrete = self.beta_cont  *  c2d_scalar

        return [c2d_scalar, D, alpha_, beta_discrete]
    
    # Find the solution for the discrete solutions
    # See the Supplimentary Materials of Kim et al. 2017 Science paper for the
    # equation numbers cited in this script.
    def DiscreteRingSolution(self):
        [c2d_scalar, D, alpha_, beta_discrete] = bumpParams.ContRingParamToDiscrete(self)

        omega = np.arcsin( np.sqrt((alpha_-1)/D) / 2 ) *2 # eq.14
        M_range = [2*np.pi/omega - 2,  2*np.pi/omega - 1] # eq 18
        M = np.ceil(M_range[0]).astype(int) # discrete variable
        phi = np.arctan( np.sin( (M+1)*omega )  /
                      ( np.cos( (M+1)*omega ) - 1    ) ) # eq 19
        tmp = (1-alpha_)*np.sin(phi) + beta_discrete * (
            np.sin(omega)*np.cos(phi)/(1-np.cos(omega)) + (M+1)*np.sin(phi) )
        beta_min =  (alpha_-1)*np.sin(phi)  / (
            np.sin(omega)*np.cos(phi)/(1-np.cos(omega)) + (M+1)*np.sin(phi) )
        beta_min_cont = beta_min / c2d_scalar
        A = 1/tmp  # eq 21 : Amplitude
        S = A* ( np.sin(omega)*np.cos(phi)/
                (1-np.cos(omega)) + (M+1)*np.sin(phi)   ) # eq 20 : Total activity

        return [D, alpha_, beta_discrete, A]
    
    

class ringParams:
    ''' Bulb ring neurons (input nodes)'''
    def __init__(self, n_input_elevation = 1):
        self.n_input_elevation = n_input_elevation
        self.n_input_azimuth = 32
        self.input_weight_id = 3 # initial synaptic weights
        self.n_input_nodes = self.n_input_elevation * self.n_input_azimuth
        
        
        
class plasticityParams:
    '''Plasticity constants'''
    
    def __init__(self, learning_rule, use_2D_input, vel):
        if (learning_rule == 'No learning'):
            self.W_max = 0
            self.epsilon_W_input = []
        elif (learning_rule == 'SOM inhib, Post-synaptically gated, input profile'):
            epsilon_W_input = 0.5
            if use_2D_input:
                epsilon_W_input = 0.75

            self.epsilon_W_input = epsilon_W_input
            self.W_max = 0.33
        elif (learning_rule == 'Hebb inhib, Pre-synaptically gated, wedge profile'):
            self.epsilon_W_input = 0.5
            self.W_max = 0.33
        else:
            ValueError('no such plasticity rule')
        self.adjusted_vel = self.adjust_vel(vel)
        
    def adjust_vel(self, vel):
        '''The learning rate is assumed to be velocity dependent.'''
        adjusted_vel = [i**2 for i in vel]
        sss = np.mean(adjusted_vel)+1.5*np.std(adjusted_vel);
        adjusted_vel = adjusted_vel/sss
        return adjusted_vel



def W_ring_attractor(session):
    '''Weight matrix for the Ring Attractor (local model)'''
    W_ring_attractor = -session.bumpParams.beta_discrete*np.ones((session.bumpParams.n_wedge_neurons, 
                                                                 session.bumpParams.n_wedge_neurons))
    for i in range(np.shape(W_ring_attractor)[0]):
        W_ring_attractor[i,i] = session.bumpParams.alpha_-2*session.bumpParams.D-session.bumpParams.beta_discrete
        ind = [(i-1) % session.bumpParams.n_wedge_neurons,
               (i+1) % session.bumpParams.n_wedge_neurons]
        W_ring_attractor[i, ind] = session.bumpParams.D - session.bumpParams.beta_discrete
        
    return W_ring_attractor



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



def vonMises_input(session):
    
    # Set the maximum input strength
    mamp = 0.35
    if (session.ringParams.n_input_elevation>2):
        mamp = mamp*0.7
    if 'Pre-synaptic' in session.learning_rule:
        mamp = mamp/3
        
    print('max amp of input=' + str(mamp))
    
    # Set the width of the circular pdf
    if 'narrow' in session.sim_cond:
        kappa = 15
    else:
        kappa = 1
    
    # Get the base shape of the input (a von Mises distribution)
    d = 2*np.pi/session.ringParams.n_input_azimuth
    a = circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                            np.pi, kappa, 'radian')
    a = (a-min(a))/(max(a)-min(a))*mamp
    
    return [mamp, kappa, a]



def visual_input(session):
    
    # Get the number of neurons around the azimuth
    n_input_azimuth = session.ringParams.n_input_azimuth
    n_input_elevation = session.ringParams.n_input_elevation
    d = 2*np.pi/n_input_azimuth;
    
    # Get the visual condition
    sim_cond = session.sim_cond
    
    # Get the time points
    sim_t = session.t
    
    # Get the fly's x position
    xpos_radian = session.xpos_radian
    
    # Initialize the visual and wedge neuron activity
    sim_visual_neurons = np.zeros((n_input_azimuth, np.size(sim_t)))
    [mamp, kappa, a] = vonMises_input(session)
    
    # Determine how to shift the visual input based on the animal's movements
    mcshift = np.ceil((xpos_radian % (2*np.pi))/(2*np.pi)*np.shape(sim_visual_neurons)[0]).astype(int)
    
    if ((sim_cond == 'no input') |
        (sim_cond == 'no visual input and no current injection') |
        (sim_cond == 'narrow, probe, 360d span')
       ):
        pass
    
    # Create a 1D stimuli with a gaussian or complex scene that follows natural turning
    elif ((sim_cond == 'natural turning, gaussian, narrow') |
        (sim_cond == 'natural turning, complex scene, narrow') |
        (sim_cond == 'natural turning, two gaussians, narrow')
       ):
        
        if ('complex scene' in sim_cond):
            atm = np.random.rand(15)
            a = circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[0], kappa*atm[1]*3, 'radian')/mamp*atm[2]*0.7 + \
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[3], kappa*atm[4]*3, 'radian')/mamp*atm[5]*0.7 + \
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[6], kappa*atm[7]*3, 'radian')/mamp*atm[8]*0.7 + \
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[9], kappa*atm[10]*3, 'radian')/mamp*atm[11]*0.7 + \
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[12], kappa*atm[13]*3, 'radian')/mamp*atm[14]*0.7        
            a = (a-np.amin(a))/(np.amax(a)-np.amin(a))*mamp
        
        for i in range(len(sim_t)-1,-1,-1):
            sim_visual_neurons[:,i] = np.roll(a,mcshift[i])
                          
        if ('two gaussians' in sim_cond):
            sn = sim_visual_neurons.shape[0]/2
            a = sim_visual_neurons + np.roll(sim_visual_neurons,int(sn),axis=0)
            sim_visual_neurons = (a-np.amin(a))/(np.amax(a)-np.amin(a))*mamp
            
    # Create 2D visual stimuli with one or two gaussians
    elif ((sim_cond == 'natural turning, gaussian, narrow, 2D') |
        (sim_cond == 'natural turning, two gaussians, narrow, 2D')):
        
        for i in range(len(sim_t)-1,-1,-1):
            sim_visual_neurons[:,i] = np.roll(a,mcshift[i])        
        
        if ('two gaussians' in sim_cond):
            sn = sim_visual_neurons.shape[0]/2
            a = sim_visual_neurons + np.roll(sim_visual_neurons,int(sn),axis=0)
            sim_visual_neurons = (a-np.amin(a))/(np.amax(a)-np.amin(a))*mamp        
        
        # Modify the array structure so that it is formatted appropriately for the differential equation solver
        tmp = np.zeros((n_input_azimuth*n_input_elevation, sim_visual_neurons.shape[1]))
        for ri in range(n_input_azimuth):
            tmp[[ri*n_input_elevation + n for n in range(n_input_elevation)],:] = np.tile(sim_visual_neurons[ri,:],(n_input_elevation,1));
        sim_visual_neurons = tmp/n_input_elevation
                          
    elif ((sim_cond == 'natural turning, 4 objects, same elevation, 2D') |
          (sim_cond == 'natural turning, 4 objects, diff elevation, 2D') |
          (sim_cond == 'natural turning, 4 objects, diff elevation - arrangement 2, 2D') |
          (sim_cond == 'natural turning, natural scene 1, 2D') |
          (sim_cond == 'natural turning, natural scene 2, 2D')):
        
        if ((n_input_elevation == 8) & (n_input_azimuth==32)):
            tmp = np.zeros((n_input_elevation, n_input_azimuth))
            if 'same elevation' in sim_cond:
                tmp[3:4, 2:5] = 1
                tmp[3:4, 10:13] = 1
                tmp[3:4, 18:21] = 1
                tmp[3:4, 26:29] = 1
            elif 'diff elevation, 2D' in sim_cond:
                tmp[0:1, 3:5] = 1
                tmp[2:3, 11:13] = 1
                tmp[4:5, 19:21] = 1
                tmp[6:7, 27:29] = 1
            elif 'arrangement 2' in sim_cond:
                tmp[6:7, 3:5] = 1
                tmp[4:5, 11:13] = 1
                tmp[2:3, 19:21] = 1
                tmp[0:1, 27:29] = 1
            elif 'natural scene 1' in sim_cond:
                imgs = np.array(loadmat('natural_scenes.mat')['imgs'])
                tmp = imgs[0,0:14:2,:][:, [round(2.75*i) for i in range(31)]]
            elif 'natural scene 2' in sim_cond:
                imgs = np.array(loadmat('natural_scenes.mat')['imgs'])
                tmp = imgs[1,0:14:2,:][:, [round(2.75*i) for i in range(31)]]

            tmpz = np.zeros(tmp.shape);
            tmp2 = np.concatenate((np.concatenate((tmpz, tmpz, tmpz)),
                                   np.concatenate((tmp, tmp, tmp)),
                                   np.concatenate((tmpz, tmpz, tmpz))),axis=1)
            tmp2 = cv2.GaussianBlur(tmp2, (0, 0), 2, 2, borderType=cv2.BORDER_REPLICATE)
            tmp = tmp2[[n_input_elevation+n for n in range(n_input_elevation)],:][:,[n_input_azimuth+n for n in range(n_input_azimuth)]]
        else:
            raise ValueError('dimension not defined')
        
        a = tmp
        a = (a-np.amin(a))/(np.amax(a)-np.amin(a))*mamp
        
        sim_visual_neurons = np.zeros((n_input_azimuth*n_input_elevation, len(sim_t)))
        for i in range(len(sim_t)-1,-1,-1):
            tmp = np.roll(a,mcshift[i],axis=1)
            sim_visual_neurons[:,i] = tmp.flatten(order = 'F')
        
    # 2D natural tuning with a Gaussian only in the top or bottom row
    elif ((sim_cond == 'natural turning, gaussian, narrow, top row in 2D') |
          (sim_cond == 'natural turning, gaussian, narrow, bottom row in 2D')
         ):
        sim_visual_neurons = np.concatenate((sim_visual_neurons,sim_visual_neurons), axis = 0)
        for i in range(len(sim_t)-1,-1,-1):
            if 'top row' in sim_cond:
                sim_visual_neurons[::2,i] = np.roll(a, mcshift[i])
            elif 'bottom row' in sim_cond:
                sim_visual_neurons[1::2,i] = np.roll(a, mcshift[i])
                
    else:
        raise ValueError('Simulation condition not defined')
        
    sim_visual_neurons = sim_visual_neurons - np.amin(sim_visual_neurons)
    
    if (np.amax(sim_visual_neurons)>0):
        sim_visual_neurons = sim_visual_neurons/np.amax(sim_visual_neurons)*mamp # normalize the max
        
    if session.add_noise:
        m = 0
        v = np.amax(sim_visual_neurons)*0.5
        if v == 0:
            v = 0.1
        n = v*(np.random.rand(sim_visual_neurons.shape[0],sim_visual_neurons.shape[1])-0.5)+m
        sim_visual_neurons = sim_visual_neurons+n
        for sii in range(sim_visual_neurons.shape[0]):
            sim_visual_neurons[sii,:] = moving_avg(sim_visual_neurons[sii,:],5)

    
    return sim_visual_neurons