import numpy as np

def plasticityParams(session):
    
    if (session.rule == 'No learning'):
        plasticity.W_max = 0
    elif (session.rule == 'SOM inhib, Post-synaptically gated, input profile'):
        epsilon_W_input = 0.5
        if session.use_2D_input:
            epsilon_W_input = 0.75
            
        plasticity.epsilon_W_input = epsilon_W_input
        plasticity.W_max = 0.33
    elif (session.rule == 'Hebb inhib, Pre-synaptically gated, wedge profile'):
        plasticity.epsilon_W_input = 0.5
        plasticity.W_max = 0.33
    else:
        ValueError('no such plasticity rule')
        
    # The learning rate is assumed to be velocity dependent.
    adjusted_vel = [i**2 for i in session.vel]
    sss = np.mean(adjusted_vel)+1.5*np.std(adjusted_vel);
    adjusted_vel = adjusted_vel/sss
    plasticity.adjusted_vel = adjusted_vel
        
    return plasticity