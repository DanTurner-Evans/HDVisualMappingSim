from mat4py import loadmat
import numpy as np
import sys
sys.path.insert(0, 'utils')
from moving_avg import moving_avg

def vel_calc(pos_data, simulation_dt, add_noise, vel_1 = 0.1775):
    '''
    vel_1 is a motor parameter for simulation. This depends on the ring attractor parameters.
    So, it is a good idea not to modify the ring attractor parameters.
    These values were obtained by trial and error in simulations.
    Note that vel is not perfectly linear, but reasonably linear.'''
    
    # Load the natural turning information
    pos_data =loadmat(pos_data)['pos_data']
    xp = np.array(pos_data['xpos'])

    # Downsample the turning information to match the simulation time
    dt = simulation_dt
    ddt = np.median(np.diff(pos_data['time'][0:100]))
    stride = round(dt/ddt)
    xpos_all = xp[0::stride]
    sim_t_all = np.arange(0,(dt*np.size(xpos_all)-dt+1e-10),dt)

    # Get only the x positions withing the time range
    t_min = pos_data['time'][0]
    t_max = pos_data['time'][-1]
    tidx = [i for i, val in enumerate(sim_t_all) if ((val>=t_min) & (val<=t_max))]
    xpos = xpos_all[tidx]
    sim_t = sim_t_all[tidx]-sim_t_all[tidx[0]]
    xpos_radian = ((xpos/88*2*np.pi - np.pi - (2*np.pi/88/2))  % (2*np.pi)) - np.pi

    # Calculate the velocity
    vel_tmp = np.diff(xpos_radian)
    vel_tmp = np.concatenate(([vel_tmp[0]],vel_tmp))
    vel_tmp = ((vel_tmp+np.pi) % (2*np.pi)) - np.pi
    vel_tmp = vel_tmp/dt
    vel_tmp = moving_avg(vel_tmp, 101)
    sim_vel = vel_tmp/np.pi*vel_1
    
    if add_noise:
        m = 0
        v = np.amax(sim_vel)*0.2
        if v == 0:
            v = vel_1/4
        n = v*(np.random.rand(len(sim_vel))-0.5)+m
        sim_vel = sim_vel+n
        sim_vel = moving_avg(sim_vel,5)
        
    # Convert the x position to radians
    xpos_radian = ((xpos/88*2*np.pi-np.pi-(2*np.pi/88/2)) % (2*np.pi)) - np.pi
    
    return [sim_t, sim_vel, xpos_radian]