import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.integrate import odeint
from data import get_orbit,random_config,sample_orbits,sample_orbits_grouped
from scipy.integrate import odeint
from torchdiffeq import odeint as todeint
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os.path

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=16, nhidden=32):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ReLU()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc31 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc31(out)
        out = self.elu(out)
        out = self.fc4(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=16, obs_dim=12, nhidden=32, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.l1 = nn.Linear(obs_dim, nhidden)
        self.l2 = nn.Linear(nhidden, nhidden)
        self.l22 = nn.Linear(nhidden, nhidden)
        self.l3 = nn.Linear(nhidden, latent_dim)

    def forward(self, x):
        # combined = torch.cat((x, h), dim=1)
        h = torch.relu(self.l1(x))
        out = self.l2(h)
        out = torch.relu(out)
        out = self.l22(out)
        out = torch.relu(out)
        out = self.l3(out)
        return out



class Decoder(nn.Module):

    def __init__(self, latent_dim=16, obs_dim=12, nhidden=32):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc22 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc22(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from itertools import permutations
class HNN(torch.nn.Module):
    """Generic class to implement predicting the derivative of the three body system
    """
    def __init__(self, input_dim,nn_type="Hamiltonian"):
        super(HNN, self).__init__()
        output_dim = input_dim if nn_type == "Feedforward" else 1
        if nn_type == "Hamiltonian" or nn_type == "Feedforward":
            self.layers = nn.Sequential(nn.Linear(input_dim,200),
                                                  nn.Tanh(),
                                                  nn.Linear(200,200),
                                                  nn.Tanh(),
                                                  nn.Linear(200,output_dim))
        else:
            raise ValueError(f"{nn_type} is not a valid type please choose between Hamiltonian, Feedforward or NeuralODE")
        self.nn_type = nn_type

        # We make a permutation matrix M to later permute the derivatives so our loss acheives:
        # dp/dt=-dH/dq, dq/dt=dH/dp
        M = torch.eye(input_dim)
        self.M = torch.cat([M[input_dim//2:], -M[:input_dim//2]]).to("cuda")

    def forward(self, x):
        # We just pass it through the layers
        return self.layers(x)

    def time_derivative(self,t, x):
        """Returns the prediction of our function

        Args:
            x (torch.Tensor): The state of the system
            t (torch.Tensor, optional): The current time step. Defaults to None.
        """
        # IF we are just doing a feedforward we try to predict the derivatives directly
        if self.nn_type == "Feedforward":
            return self.layers(x)

        # Otherwise we calculate -dH/dq as our prediction for dp/dt and dH/dp as our prediction for dq/dt
        F2 = self.forward(x) 

        dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] 
        hamiltonian_derivative = dF2 @ self.M.T

        return hamiltonian_derivative

def animate_trajectories_with_tips(pred_x, x, batch, filepath="/mnt/data/trajectory_animation_with_tips.gif", fps=20):
    """
    Animate and save the trajectories of predicted and true positions with balls at the tips.

    Parameters:
    - pred_x: Predicted trajectories, expected shape (1, T, 6).
    - x: True trajectories, expected shape (1, T, 6).
    - batch: Index of the batch for which to plot the trajectories.
    - filepath: Path to save the animation GIF.
    - fps: Frames per second for the animation.
    """
    T = pred_x.shape[1]  # Number of time steps

    fig, ax = plt.subplots()

    # Initialize lines for predicted (dashed) and true (solid) trajectories
    lines = [plt.plot([], [], "--", color=f"C{i}")[0] for i in range(3)] + \
            [plt.plot([], [], "-", color=f"C{i}")[0] for i in range(3)]
    # Initialize markers for the tips of each trajectory
    markers = [plt.plot([], [], 'x', color=f"C{i}")[0] for i in range(3)]+[plt.plot([], [], 'o', color=f"C{i}")[0] for i in range(3)]
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    def init():
        for line in lines:
            line.set_data([], [])
        for marker in markers:
            marker.set_data([], [])
        return lines + markers

    def update(frame):
        for i in range(3):
            # Update predicted trajectories
            lines[i].set_data(pred_x[0, :frame, i], pred_x[0, :frame, i+3])
            # Update markers for predicted trajectories
            markers[i].set_data(pred_x[0, frame-1, i], pred_x[0, frame-1, i+3])
            
            # Update true trajectories
            lines[i+3].set_data(x[batch][ 0, :frame, i], x[batch][ 0, :frame, i+3])
            # Update markers for true trajectories
            markers[i+3].set_data(x[batch][ 0, frame-1, i], x[batch][0, frame-1, i+3])
        return lines + markers

    ani = FuncAnimation(fig, update, frames=np.arange(1, T+1), init_func=init, blit=True)

    # Save the animation as a GIF
    writer = PillowWriter(fps=fps)
    ani.save(filepath, writer=writer)

    plt.close(fig)  # Close the figure to avoid displaying it in a Jupyter notebook or similar environment


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl
fname = "grouped_data"
if not os.path.isfile(f"{fname}_x_coords.pt"):

    data,_ = sample_orbits_grouped(time_steps = 80,trials = 5000,t_span=[0,20])
    x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32)
    # test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dcoords'])
    torch.save(x,f"{fname}_x_coords.pt")
    torch.save(dxdt,f"{fname}_dxdt_vals.pt")
else:
    x = torch.load(f"{fname}_x_coords.pt")
    dxdt = torch.load(f"{fname}_dxdt_vals.pt")
import sys
latent_dim = 64#16
nhidden = 64#32
obs_dim = 18
rnn_nhidden = 64#32
noise_std = .001
device = "cuda"
batch_size = 256

func = LatentODEfunc(latent_dim,nhidden).to(device)
rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_size).to(device)
dec = Decoder(latent_dim, 12, nhidden).to(device)
params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
optimizer = torch.optim.Adam(params, lr=1e-2)
epochs = 2500
sample_ts = torch.linspace(0, 20, 80)
losses = []
x=x.to(device)
fname = "net_0_acc"
def l4_loss(prediction, target):
    error = torch.abs(prediction - target) ** 4
    loss = torch.mean(error)
    return loss
def l2_loss(prediction, target):
    error = torch.sqrt(torch.abs(prediction - target))
    loss = torch.mean(error)
    return loss
def equiv_func(p1,p2,p3,v1,v2,v3,rec,func,dec):
    sp2 = p2- p1
    sp3 = p3- p1
    sv1 = v1
    sv2 = v2
    sv3 = v3
    features = []
    features.append((sp2*sp2).sum(dim=-1,keepdim    =True))
    features.append((sp2*sp3).sum(dim=-1,keepdim    =True))
    features.append((sp3*sp3).sum(dim=-1,keepdim    =True))

    features.append((sv1*sp2).sum(dim=-1,keepdim    =True))
    features.append((sv2*sp2).sum(dim=-1,keepdim    =True))
    features.append((sv3*sp2).sum(dim=-1,keepdim    =True))
    
    features.append((sv1*sp3).sum(dim=-1,keepdim    =True))
    features.append((sv2*sp3).sum(dim=-1,keepdim    =True))
    features.append((sv3*sp3).sum(dim=-1,keepdim    =True))

    features.append((sv1*sv1).sum(dim=-1,keepdim    =True))
    features.append((sv2*sv1).sum(dim=-1,keepdim    =True))
    features.append((sv3*sv1).sum(dim=-1,keepdim    =True))
    features.append((sv1*sv2).sum(dim=-1,keepdim    =True))
    features.append((sv2*sv2).sum(dim=-1,keepdim    =True))
    features.append((sv3*sv2).sum(dim=-1,keepdim    =True))
    features.append((sv1*sv3).sum(dim=-1,keepdim    =True))
    features.append((sv2*sv3).sum(dim=-1,keepdim    =True))
    features.append((sv3*sv3).sum(dim=-1,keepdim    =True))
    features = torch.hstack(features)
    out = rec.forward(features)
    qz0_mean = out
    pred_z = todeint(func, qz0_mean, sample_ts).permute(1, 0, 2)
    pred_coeffs = dec(pred_z)
    equiv_coeffs_p = torch.softmax(pred_coeffs[:,:,:3],dim=-1)
    other_coeffs_p = pred_coeffs[:,:,3:6]
    new_p = p1.unsqueeze(1)*equiv_coeffs_p[:,:,0].unsqueeze(-1).repeat(1,1,2)+p2.unsqueeze(1)*equiv_coeffs_p[:,:,1].unsqueeze(-1).repeat(1,1,2)+ p3.unsqueeze(1)*equiv_coeffs_p[:,:,2].unsqueeze(-1).repeat(1,1,2)\
    + v1.unsqueeze(1)*other_coeffs_p[:,:,0].unsqueeze(-1).repeat(1,1,2)+v2.unsqueeze(1)*other_coeffs_p[:,:,1].unsqueeze(-1).repeat(1,1,2)+v3.unsqueeze(1)*other_coeffs_p[:,:,2].unsqueeze(-1).repeat(1,1,2)
    
    inv_coeffs = pred_coeffs[:,:,6:9]-torch.mean(pred_coeffs[:,:,6:9],dim=-1,keepdim=True)
    other_coeffs_v = pred_coeffs[:,:,9:]
    new_v = p1.unsqueeze(1)*inv_coeffs[:,:,0].unsqueeze(-1).repeat(1,1,2)+p2.unsqueeze(1)*inv_coeffs[:,:,1].unsqueeze(-1).repeat(1,1,2)+ p3.unsqueeze(1)*inv_coeffs[:,:,2].unsqueeze(-1).repeat(1,1,2)\
    + v1.unsqueeze(1)*other_coeffs_v[:,:,0].unsqueeze(-1).repeat(1,1,2)+v2.unsqueeze(1)*other_coeffs_v[:,:,1].unsqueeze(-1).repeat(1,1,2)+v3.unsqueeze(1)*other_coeffs_v[:,:,2].unsqueeze(-1).repeat(1,1,2)

    return new_p,new_v

for itr in tqdm(range(1, epochs + 1)):
    optimizer.zero_grad()
    # backward in time to infer q(z_0)
    batch = torch.randperm(len(x))[:batch_size].to(device)
    # batch = torch.arange(len(x))[:batch_size].to(device)
    # for t in reversed(range(x.size(1))):
    obs = x[batch, 0, :]
    # The first features are [x1,x2,x3,y1,y2,y3,vx1,vx2,vx3,vy1,vy2,vy3
    mask = [True,False,False]*2+[False]*6
    p1 = obs[:,mask]
    mask = [False,True,False]*2+[False]*6
    p2 = obs[:,mask]
    mask = [False,False,True]*2+[False]*6
    p3 = obs[:,mask]
    mask = [False]*6+[True,False,False]*2
    v1 = obs[:,mask]
    mask = [False]*6+[False,True,False]*2
    v2 = obs[:,mask]
    mask = [False]*6+[False,False,True]*2
    v3 = obs[:,mask]

    planets = [p1,p2,p3]
    velocities = [v1,v2,v3]
    pred_x = torch.zeros_like(x[batch])
    in1 = []
    in2 = []
    in3 = []
    in4 = []
    in5 = []
    in6 = []
    for i in range(3):
        in1.append(planets[i])
        in1.append(planets[i])
        in2.append(planets[(i+1)%3])
        in2.append(planets[(i+2)%3])
        in3.append(planets[(i+2)%3])
        in3.append(planets[(i+1)%3])
        in4.append(velocities[i])
        in4.append(velocities[i])
        in5.append(velocities[(i+1)%3])
        in5.append(velocities[(i+2)%3])
        in6.append(velocities[(i+2)%3])
        in6.append(velocities[(i+1)%3])
    in1 = torch.cat(in1,dim=0)
    in2 = torch.cat(in2,dim=0)
    in3 = torch.cat(in3,dim=0)
    in4 = torch.cat(in4,dim=0)
    in5 = torch.cat(in5,dim=0)
    in6 = torch.cat(in6,dim=0)
    c = (v1+v2+v3)/3
    c = c.unsqueeze(1).repeat(1,80,1)
    output_p,output_v = equiv_func(in1,in2,in3,in4,in5,in6,rec,func,dec)
    for i in range(3):
        new_p = output_p[2*i*batch_size:(2*i+1)*batch_size]
        new_v = output_v[2*i*batch_size:(2*i+1)*batch_size]
        avg = 1/3*(output_v[2*0*batch_size:(2*0+1)*batch_size]+output_v[2*1*batch_size:(2*1+1)*batch_size]+output_v[2*2*batch_size:(2*2+1)*batch_size])
        new_pn = output_p[(2*i+1)*batch_size:(2*i+2)*batch_size]
        new_vn = output_v[(2*i+1)*batch_size:(2*i+2)*batch_size]
        avg_n = 1/3*(output_v[(2*0+1)*batch_size:(2*0+2)*batch_size]+output_v[(2*1+1)*batch_size:(2*1+2)*batch_size]+output_v[(2*2+1)*batch_size:(2*2+2)*batch_size])
        # new_p,new_v= equiv_func(planets[i],planets[(i+1)%3],planets[(i+2)%3],velocities[i],velocities[(i+1)%3],velocities[(i+2)%3],rec,func,dec)
        # new_pn,new_vn = equiv_func(planets[i],planets[(i+2)%3],planets[(i+1)%3],velocities[i],velocities[(i+2)%3],velocities[(i+1)%3],rec,func,dec)
        avg = avg+avg_n
        new_p  = new_p + new_pn
        new_v  = (new_v + new_vn) - avg + c
        pred_x[:,:,i] = new_p[:,:,0]
        pred_x[:,:,3+i] = new_p[:,:,1]
        pred_x[:,:,6+i] = new_v[:,:,0]
        pred_x[:,:,9+i] = new_v[:,:,1]

    # Plan, for each planet I need to do this
    # shftcthe locations of the other planets around our current planet
    # compute the dot products between everything, feed them into the model to compute one order
    # produce 12 coefficients 6 add to velocity 6 add to position
    # The coefficients of the ps in the positions add to one, while for the velocity they add to zero
    if itr %500==0:
        animate_trajectories_with_tips(pred_x.detach().cpu(),x.detach().cpu(),batch.detach().cpu(),fname+".gif")

    loss = F.mse_loss(pred_x,x[batch])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -np.mean(losses)))
plt.clf()
plt.plot(losses)
plt.savefig(fname + ".png")
torch.save(func,"laten_ode.pt")
torch.save(dec, "dec.pt")
torch.save(rec, "enc.pt")