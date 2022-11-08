def probe_sim(session):
    ''' A short open loop period with no input velocity'''

    # Specify the time to probe
    t_total = 40 # seconds

    # Specify how long the visual input should remain at a given position
    t_single_duration = 0.25 # seconds
    npoints = round(t_single_duration/dt)

    # Create a list of offset values for the visual input
    tmp=[]
    step_start = 0
    for tmpi in range(session.ringParams.n_input_azimuth):
        tmp = [tmp np.ones(npoints)*step_start]
        step_start = step_start+1
    tmp = [tmp np.fliplr(tmp) tmp np.fliplr(tmp) tmp tmp np.fliplr(tmp) np.fliplr(tmp)];
    mcshift = np.tile(tmp,(1,np.ceil(t_total/dt/len(tmp))))

    # Generate the open loop visual input from the offset values
    sim_visual_neurons = []
    [mamp, kappa, a] = vonMises_input(session)
    for i in np.fliplr(mcshift):
        sim_visual_neurons[:,i] = np.roll(a,mcshift[i])

    sim_vel = np.zeros(1,len(mcshift))
    sim_t = np.arange(0,dt*(len(sim_vel)-1),dt)

    return [sim_t, sim_vel, sim_visual_neurons]