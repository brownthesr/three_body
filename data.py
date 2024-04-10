import time
import numpy as np
import scipy
solve_ivp = scipy.integrate.solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


##### ENERGY #####
def potential_energy(state):
    '''U=\sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1,1,state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i+1,state.shape[0]):
            r_ij = ((state[i:i+1,1:3] - state[j:j+1,1:3])**2).sum(1, keepdims=True)**.5
            m_i = state[i:i+1,0:1]
            m_j = state[j:j+1,0:1]
            tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=\sum_i .5*m*v^2'''
    energies = .5 * state[:,0:1] * (state[:,3:5]**2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)


##### DYNAMICS #####
def get_accelerations(state, epsilon=0):
    # shape of state is [bodies x properties]
    net_accs = [] # [nbodies x 2]
    for i in range(state.shape[0]): # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
        displacements = other_bodies[:, 1:3] - state[i, 1:3] # indexes 1:3 -> pxs, pys
        distances = (displacements**2).sum(1, keepdims=True)**0.5
        masses = other_bodies[:, 0:1] # index 0 -> mass
        pointwise_accs = masses * displacements / (distances**3 + epsilon) # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs
  
def update(t, state):
    state = state.reshape(-1,5) # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:,1:3] = state[:,3:5] # dx, dy = vx, vy
    deriv[:,3:5] = get_accelerations(state)
    return deriv.reshape(-1)


##### INTEGRATION SETTINGS #####
def get_orbit(state, update_fn=update, t_points=100, t_span=[0,2], nbodies=3, **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-9

    orbit_settings = locals()

    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval

    path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                     t_eval=t_eval, **kwargs)
    orbit = path['y'].reshape(nbodies, 5, t_points)
    return orbit, orbit_settings


##### INITIALIZE THE TWO BODIES #####
def rotate2d(p, theta):
  c, s = np.cos(theta), np.sin(theta)
  R = np.array([[c, -s],[s, c]])
  return (R @ p.reshape(2,1)).squeeze()

def random_config(nu=2e-1, min_radius=0.9, max_radius=1.2):
  '''This is not principled at all yet'''
  state = np.zeros((3,5))
  state[:,0] = 1
  p1 = 2*np.random.rand(2) - 1
  r = np.random.rand() * (max_radius-min_radius) + min_radius
  
  p1 *= r/np.sqrt( np.sum((p1**2)) )
  p2 = rotate2d(p1, theta=2*np.pi/3)
  p3 = rotate2d(p2, theta=2*np.pi/3)

  # # velocity that yields a circular orbit
  v1 = rotate2d(p1, theta=np.pi/2)
  v1 = v1 / r**1.5
  v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2)) # scale factor to get circular trajectories
  v2 = rotate2d(v1, theta=2*np.pi/3)
  v3 = rotate2d(v2, theta=2*np.pi/3)
  
  # make the circular orbits slightly chaotic
  v1 *= 1 + nu*(2*np.random.rand(2) - 1)
  v2 *= 1 + nu*(2*np.random.rand(2) - 1)
  v3 *= 1 + nu*(2*np.random.rand(2) - 1)

  state[0,1:3], state[0,3:5] = p1, v1
  state[1,1:3], state[1,3:5] = p2, v2
  state[2,1:3], state[2,3:5] = p3, v3
  return state
class CustomProgressBar:
    def __init__(self, total, bar_length=50):
        self.total = total
        self.bar_length = bar_length
        self.completed = 0
        self.start_time = time.time()

    def update(self, i):
        self.completed = i
        percent_completed = (self.completed / self.total) * 100
        num_hashes = int((self.completed / self.total) * self.bar_length)
        bar = '#' * num_hashes + '.' * (self.bar_length - num_hashes)

        # Calculate estimated time to completion
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if self.completed == 0:
            eta = 0
        else:
            eta = (elapsed_time / self.completed) * (self.total - self.completed)

        # Convert ETA from seconds to a more readable format
        eta_minutes, eta_seconds = divmod(int(eta), 60)
        eta_str = f"{eta_minutes}m {eta_seconds}s" if eta > 0 else "almost done"

        print(f"\r[{bar}] {percent_completed:.2f}% ETA: {eta_str}", end='', flush=True)
        
    def finish(self):
        print()  # Move to the next line

##### INTEGRATE AN ORBIT OR TWO #####
def sample_orbits(timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                  min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False, **kwargs):
    
    orbit_settings = locals()
    if verbose:
        print("Making a dataset of near-circular 3-body orbits:")
    
    x, dx, e = [], [], []
    N = timesteps*trials
    print("Generating dataset")
    progress_bar = CustomProgressBar(total=N)
    while len(x) < N:
        progress_bar.update(len(x))
        # print(len(x),N)
        state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
        # orbit is of the shape [B,S,T]
        # We can use this to simulate our problem

        # The next thing first reshapes our problem to be [T,B,S] then to [T, S*B]
        batch = orbit.transpose(2,0,1).reshape(-1,nbodies*5)

        # For each time step do:
        for state in batch:
            dstate = update(None, state)
            
            # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
            # the first thing is the mass, then the coordinates, then the velocities
            # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
            coords = state.reshape(nbodies,5).T[1:].flatten()
            # dcoords literally just calls the update function but remember that this update function is supposed to retur nthe derivative
            dcoords = dstate.reshape(nbodies,5).T[1:].flatten()
            x.append(coords)
            dx.append(dcoords)

            shaped_state = state.copy().reshape(nbodies,5,1)
            e.append(total_energy(shaped_state))

    data = {'coords': np.stack(x)[:N],
            'dcoords': np.stack(dx)[:N],
            'energy': np.stack(e)[:N] }
    return data, orbit_settings
def sample_orbits_grouped(timesteps=80, trials=5000, nbodies=3, orbit_noise=2e-1,
                  min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False, **kwargs):

    orbit_settings = locals()
    if verbose:
        print("Making a dataset of near-circular 3-body orbits:")
    
    x, dx, e = [], [], []
    N = trials
    print("Generating dataset")
    progress_bar = CustomProgressBar(total=N)
    while len(x) < N:
        progress_bar.update(len(x))
        # print(len(x),N)
        state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
        # orbit is of the shape [B,S,T]
        # We can use this to simulate our problem

        # The next thing first reshapes our problem to be [T,B,S] then to [T, S*B]
        batch = orbit.transpose(2,0,1).reshape(-1,nbodies*5)
        x_c = []
        dx_c = []
        e_c = []
        # For each time step do:
        for state in batch:
            dstate = update(None, state)
            
            # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
            # the first thing is the mass, then the coordinates, then the velocities
            # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
            coords = state.reshape(nbodies,5).T[1:].flatten()
            # dcoords literally just calls the update function but remember that this update function is supposed to retur nthe derivative
            dcoords = dstate.reshape(nbodies,5).T[1:].flatten()
            x_c.append(coords)
            dx_c.append(dcoords)

            shaped_state = state.copy().reshape(nbodies,5,1)
            e_c.append(total_energy(shaped_state))
        x.append(np.stack(x_c))
        dx.append(np.stack(dx_c))
        e.append(np.stack(e_c))
    print(len(x))
    data = {'coords': np.stack(x)[:N],
            'dcoords': np.stack(dx)[:N],
            'energy': np.stack(e)[:N] }
    return data, orbit_settings


##### MAKE A DATASET #####
def make_orbits_dataset(test_split=0.2, **kwargs):
    data, orbit_settings = sample_orbits(**kwargs)
    
    # make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    data['meta'] = orbit_settings
    return data


##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-orbits-dataset.pkl'.format(save_dir, experiment_name)

    # try:
    #     # data = from_pickle(path)
    #     print("Successfully loaded data from {}".format(path))
    # except:
    #     print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
    data = make_orbits_dataset(**kwargs)
        # to_pickle(data, path)

    return data

# v = get_dataset(None,None,trials=1,timesteps=60)
# coords = v["coords"]
# # The coords of the shape (x1,x2,x3,y1,y2,y3,vx1,vx2,vx3,vy1,vy2,vy3)
# test_coords = v["test_coords"]
# test_dcoords = v["test_dcoords"]
# dcoords = v["dcoords"]
# energy = v["energy"]
# test_energy = v["test_energy"]
# meta= v["meta"]
# print(coords.shape)
# print(dcoords.shape)
# print(energy.shape)