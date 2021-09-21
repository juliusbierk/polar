import numpy as np
import time
import itertools
import os
from polarcore import Polar
from initsystems import init_random_system
import pickle
import torch
import matplotlib.pyplot as plt


# Initialize a random system
n = 500
x, p, q = init_random_system(n)

x *= 3
beta = 0.0 + np.zeros(len(x))  # cell division rate
lam = np.array([1.0, 0.0, 0.0, 0.0])
eta = 1e-5  # noise

# Make one cell polar and divide it faster
index = np.argmin(np.sum(x**2, axis=1))
lam = np.repeat(lam[None, :], len(x), axis=0)
lam[index, :] = (0, 1, 0, 0)
beta[index] = 0.0025

# Simulation parameters
timesteps = 15000
yield_every = 5  # save simulation state every x time steps


# Potential
def potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)
    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + lam[:, :, 2] * S2 + lam[:, :, 3] * S3
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)

    # Add:
    red_blue_interaction = (lam_i[:, :, 0] < 0.1) * (lam_j[:, :, 0] > 0.9)  # i is polar
    red_blue_interaction = torch.sum(dx * pi, dim=2) * red_blue_interaction
    Vij2 = red_blue_interaction * (torch.exp(-d) - torch.exp(-d / 5))

    # Keep red in middle:
    red = lam_i[:, :, 0] < 0.1
    Vij3 = 0.005 * torch.sum(x**2, dim=1)[:, None] * red

    return Vij + 0.5 * Vij2 + Vij3


# Make the simulation runner object:
sim = Polar(device="cuda", init_k=50)
runner = sim.simulation(x, p, q, lam, beta, eta=eta, yield_every=yield_every, potential=potential)

# Running the simulation
data = []  # For storing data
i = 0
t1 = time.time()
print('Starting')

for xx, pp, qq, lam in itertools.islice(runner, timesteps):
    i += 1
    print(f'Running {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(xx)} cells)')
    data.append((xx, pp, qq, lam))

    if len(xx) > 1500:
        print('Stopping')
        break

try:
    os.mkdir('data')
except:
    pass
with open('data/test2.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
