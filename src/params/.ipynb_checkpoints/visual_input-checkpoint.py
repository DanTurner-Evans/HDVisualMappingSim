import numpy as np
from mat4py import loadmat
import cv2

def visual_input(session):
    
    # Get the number of neurons around the azimuth
    n_input_azimuth = session.ringParams.n_input_azimuth
    n_input_elevation = session.ringParams.n_input_elevation
    
    # Get the visual condition
    vis_cond = session.vis_cond
    
    # Get the time points
    sim_t = session.t
    
    # Get the fly's x position
    xpos_radian = session.xpos_radian
    
    # Initialize the visual and wedge neuron activity
    sim_visual_neurons = np.zeros(n_input_azimuth, np.size(sim_t))
    [mamp, kappa, a] = vonMises_input(session)
    
    # Determine how to shift the visual input based on the animal's movements
    mcshift = np.ceil((xpos_radian % (2*np.pi))/(2*np.pi)*np.shape(sim_visual_neurons)[0])
    
    if ((sim_cond == 'no input') |
        (sim_cond == 'no visual input and no curent injection') |
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
                                    np.pi-np.pi*atm[0], kappa*atm[1]*3, 'radian')/mamp*atm[2]*0.7 + ...
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[3], kappa*atm[4]*3, 'radian')/mamp*atm[5]*0.7 + ...
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[6], kappa*atm[7]*3, 'radian')/mamp*atm[8]*0.7 + ...
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[9], kappa*atm[10]*3, 'radian')/mamp*atm[11]*0.7 + ...
            circularPdfVonMises(np.arange(0,(2*np.pi-0.0001),d),
                                    np.pi-np.pi*atm[12], kappa*atm[13]*3, 'radian')/mamp*atm[14]*0.7        
            a = (a-min(a))/(max(a)-min(a))*mamp
        
        for i in range(len(sim_t-1,-1,-1)):
            sim_visual_neurons[:,i] = np.roll(a,mcshift[i])
                          
        if ('two gaussians' in sim_cond):
            sn = np.shape(sim_visual_neurons,1)/2
            a = sim_visual_neurons + np.roll(sim_visual_neurons,sn)
            sim_visual_neurons = (a-min(a))/(max(a)-min(a))*mamp
            
    # Create 2D visual stimuli with one or two gaussians
    elif ((vis_cond == 'natural turning, gaussian, narrow, 2D') |
        (vis_cond == 'natural turning, two gaussians, narrow, 2D')):
        
        for i in range(len(sim_t)-1,-1,-1):
            sim_visual_neurons[:,i] = np.roll(a,mcshift[i])        
        
        if ('two gaussians' in vis_cond):
            sn = np.shape(sim_visual_neurons,1)/2
            a = sim_visual_neurons + np.roll(sim_visual_neurons,sn)
            sim_visual_neurons = (a-min(a))/(max(a)-min(a))*mamp        
        
        # Modify the array structure so that it is formatted appropriately for the differential equation solver
        tmp = np.zeros(n_input_azimuth*n_input_elevation, np.shape(sim_visual_neurons,2))
        for ri in range(n_input_azimuth):
            tmp[(ri*n_input_elevation + list(range(n_input_elevation))), :] = np.tile(sim_visual_neurons[ri,:],n_input_elevation);
        sim_visual_neurons = tmp/n_input_elevation
                          
    elif ((sim_cond == 'natural turning, 4 objects, same elevation, 2D') |
          (sim_cond == 'natural turning, 4 objects, diff elevation, 2D') |
          (sim_cond == 'natural turning, 4 objects, diff elevation - arrangement 2, 2D') |
          (sim_cond == 'natural turning, natural scene 1, 2D') |
          (sim_cond == 'natural turning, natural scene 2, 2D')):
        
        if ((n_input_elevation == 8) & (n_input_azimuth==32)):
            tmp = np.zeros(n_input_elevation, n_input_azimuth)
            if 'same elevation' in sim_cond:
                tmp[3:4, 0:3] = 1
                tmp[3:4, 8:11] = 1
                tmp[3:4, 16:19] = 1
                tmp[3:4, 24:27] = 1
            elif 'diff elevation, 2D' in sim_cond:
                tmp[0:1, 0:2] = 1
                tmp[2:3, 8:10] = 1
                tmp[4:5, 16:18] = 1
                tmp[6:7, 24:26] = 1
            elif 'arrangement 2' in sim_cond:
                tmp[6:7, 0:2] = 1
                tmp[4:5, 8:10] = 1
                tmp[2:3, 16:18] = 1
                tmp[0:1, 24:26] = 1
            elif 'natural scene 1' in sim_cond:
                imgs = loadmat('natural_scenes.mat')['imgs']
                tmp = imgs[0,0:14:2, round(list(range(31))*2.75)]
            elif 'natural scene 2' in sim_cond:
                imgs = loadmat('natural_scenes.mat')['imgs']
                tmp = imgs[1,0:14:2, round(list(range(31))*2.75)]

            tmpz = zeros(size(tmp));
            tmp2 = [[tmpz, tmpz, tmpz], [tmp, tmp, tmp], [tmpz, tmpz, tmpz]]

            sss = 1.5*[1,1]
            sss = 2*[1,1]
            tmp2 = cv2.GaussianBlur(tmp2, ksize=(0, 0), sigmaX=sss, borderType=cv2.BORDER_REPLICATE)
            tmp = tmp2[n_input_elevation+list(range(n_input_elevation)), n_input_azimuth+list(range(n_input_azimuth))]
        else:
            raise ValueError('dimension not defined')
        
        a = tmp
        a = (a-min(a))/(max(a)-min(a))*mamp
        
        sim_visual_neurons = np.zeros(n_input_azimuth*n_input_elevation, len(sim_t))
        for i in range(len(sim_t)-1,-1,-1):
            tmp = np.roll(a,mcshift[i],axis=2)
            sim_visual_neurons[:,i] = tmp
        
    # 2D natural tuning with a Gaussian only in the top or bottom row
    elif ((sim_cond == 'natural turning, gaussian, narrow, top row in 2D') |
          (sim_cond == 'natural turning, gaussian, narrow, bottom row in 2D')
         ):
        sim_visual_neurons = np.concatenate((sim_visual_neurons,sim_visual_neurons), axis = 1)
        for i in np.fliplr(np.arange(0,len(sim_t))):
            if 'top row' in sim_cond:
                sim_visual_neurons[::2,i] = np.roll(a, mcshift[i])
            elif 'bottom row' in sim_cond:
                sim_visual_neurons[1::2,i] = np.roll(a, mcshift[i])
                
    else:
        raise Value_Error('Simulation condition not defined')
        
    sim_visual_neurons = sim_visual_neurons - min(sim_visual_neurons)
    
    if max(sim_visual_neurons)>0:
        sim_visual_neurons = sim_visual_neurons/max(sim_visual_neurons)*mamp # normalize the max
        
    if session.add_noise:
        m = 0
        v = max(sim_visual_neurons)*0.5
        if v == 0:
            v = 0.1
        n = v*(np.random.rand(len(sim_visual_neurons))-0.5)+m
        sim_visual_neurons = sim_visual_neurons+n
        for sii in range(np.shape(sim_visual_neurons,1)):
            sim_visual_neurons[sii,:] = moving_avg(sim_visual_neurons[sii,:],5)

    
    return sim_visual_neurons