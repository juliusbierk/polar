import numpy as np
import time
import itertools
import os
from polarcore import Polar
from initsystems import init_random_system
import pickle
import torch

save_name = 'test1'

# Initialize a random system
n = 2500
x, p, q = init_random_system(n)
x *= 3

# Parameters
beta = np.zeros(len(x))  # cell division rate
lam = np.array([0.0, 1.0, 0.0, 0.0])
eta = 0.0  # noise

# Allow per-cell polarity parameters
lam = np.repeat(lam[None, :], len(x), axis=0)

# Simulation parameters
timesteps = 100
yield_every = 50  # save simulation state every x time steps


# Potential
def potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)

    lam = 0.5 * (lam_i + lam_j)

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + lam[:, :, 2] * S2 + lam[:, :, 3] * S3
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


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

    # if len(xx) > 1000:
    #     print('Stopping')
    #     break

try:
    os.mkdir('data')
except:
    pass
with open('data/test1.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
