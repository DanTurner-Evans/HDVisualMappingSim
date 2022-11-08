import numpy as np
import sys
sys.path.insert(0, 'utils')
from half_wave_rectify import half_wave_rectify

def RingAttractorODESolver(t,y,session):

    nw = session.bumpParams.n_wedge_neurons
    ni = session.ringParams.n_input_nodes

    # These two variables will be updated
    y1 = y[0:nw] # wedge neuron activity (ring attractor activity)
    y2 = y[nw:]  # W_input
    W_input = np.reshape(y2, (nw, ni))

    # Take the index of the current time point
    dt = session.dt
    ind = t/dt
    indf = np.floor(ind).astype(int)
    indc = np.ceil(ind).astype(int)
    
    # Obtain interpolated inputs to wedge neurons
    # input_neuron_activity: input from ring neurons (negative if ring neurons are inhibitory)
    # vel_signal: velocity input
    # wedge_injection_signal: direct current injection
    
    if indf==indc:
        input_neuron_activity = session.sim_visual_neurons[:,indf]
        vel_signal = session.vel[indf]
        wedge_injection_signal = session.wedge_current_injection[:,indf]
    else:
        input_neuron_activity = (indc-ind)*session.sim_visual_neurons[:,indf] + \
        (ind-indf)*session.sim_visual_neurons[:,indc]
        vel_signal = (indc-ind)*session.vel[indf] + (ind-indf)*session.vel[indc]
        wedge_injection_signal = (indc-ind)*session.wedge_current_injection[:,indf] + \
        (ind-indf)*session.wedge_current_injection[:,indc]

    ## Calculate the current from each input type to wedge neurons
    
    ##
    # 1. Current from visual neurons
    input_current_from_visual_neurons = -np.dot(W_input,(input_neuron_activity * 2*np.pi/ni))
    
    ##
    # 2. Turning signal
    # Scale the vel signal (discretization)
    vel = vel_signal*nw/2/np.pi

    # To avoid asymmetricity, I used { [f(t+dt)-f(t)]/dt + [f(t)-f(t-dt)]/dt }/2
    turning_signal = vel*(np.concatenate(([y1[-1]],y1[:-1:])) - np.concatenate((y1[1::],[y1[0]])))/2
    
    ##
    # 3. Current injection
    wedge_current_injection = wedge_injection_signal * 2*np.pi/nw;
    
    
    ## Calculate the delta of the ring attractor
    tmp = np.dot(session.W_ring_attractor,y1) + 1 + input_current_from_visual_neurons + turning_signal + wedge_current_injection
    tmp[tmp>session.membrane_saturation] = session.membrane_saturation
    tmp[tmp<session.membrane_threshold] = session.membrane_threshold
    y1_tmp = tmp
    delta_y1 = [i/j for i,j in zip((y1_tmp - y1), session.bumpParams.tau_wedge)]
    
    ##################################################################
    ## Update the W_input with a plasticity rule
    
    ina_rep = np.tile(np.array(input_neuron_activity, ndmin=2),(nw,1))
    wedge_rep = np.tile(np.array(y1, ndmin=2).T,(1,ni))
    
    if session.learning_rule == 'No learning':
        dW_input_dt = np.zeros(np.shape(ina_rep))
        W_max = 0
    else:
        epsilon = session.plasticityParams.epsilon_W_input
        W_max = session.plasticityParams.W_max
        
        # interpolate
        if (indf==indc):
            fv = session.plasticityParams.adjusted_vel[indf]
        else:
            fv = (indc-ind)*session.plasticityParams.adjusted_vel[indf] + \
            (ind-indf)*session.plasticityParams.adjusted_vel[indc]
        
        # Compute dW
        
        if (session.learning_rule == 'SOM inhib, Post-synaptically gated, input profile'):
            f_th = 0.04 # about half of the maximum wedge neuron activity. So, this can be dynamically adjustable, but not implemented.
            [PF, NF] = half_wave_rectify(wedge_rep-f_th) # PF is the positive part, NF is the negative part. Both are positive.
            
            dW_input_dt = [i*j for i,j in zip(3*fv*epsilon*wedge_rep, (W_max - ina_rep - W_input))]
                
            # In case, the wedge neurons are noisy, it may need to be
            # thresholded.
            # dW_input = fv*epsilon*PF.*( W_max - ina_rep - W_input )
            
        if (session.learning_rule == 'Hebb inhib, Pre-synaptically gated, wedge profile'):
            ### *** IMPORTANT: The mamp in the "sim_cond.m" should be small.
            g_th = 0.1/3 # About a bit less than the median of input activity.
            [PG, NG] = half_wave_rectify(ina_rep-g_th) # PG is the positive part, NG is the negative part. Both are positive.
                
            dW_input_dt = [i*j for i,j in zip(6*fv*epsilon*(W_max-(wedge_rep/0.08)*W_max-W_input),PG)]
        
    # New W_input state
    W_input = W_input + dW_input_dt
    
    # Cap the value
    W_input[W_input<0] = 0
    if (W_max>0):
        W_input[W_input>W_max] = W_max
        
    # Calculate the delta
    delta_y2 = (W_input.flatten() - y2)
    
    ## Combine results
    dydt = [*delta_y1,*delta_y2]
    
    ## Occasionally display the simulation time
    if (((t % 20) < 0.001) & (np.random.rand() > 0.7)):
        print('  ' + str(t) + 's')
        
    return dydt