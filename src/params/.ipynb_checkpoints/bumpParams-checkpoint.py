import numpy as np

class bumpParams:
    ''' E-PG neurons (compass neurons), ring attractor
    The ring attractor parameters for simulation 
    tau and D affect the bump speed.
    Larger D or largaer tau mean slower flow
    beta_discrete has minimal to no impact on the speed of the bump.
    '''
    def __init__(self):
        self.n_wedge_neurons = 32
        self.tau_wedge = 0.05*np.ones(session.n_wedge_neurons)
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
        M = np.ceil(M_range[0]) # discrete variable
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