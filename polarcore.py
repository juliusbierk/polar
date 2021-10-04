import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree


class Polar:
    """
    Class to define and run simulations of the polarity model of cell movement

    Examples:
    ----------
    ```
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

        if len(xx) > 1000:
            print('Stopping')
            break
    ```
    """
    def __init__(self, device='cpu', dtype=torch.float, init_k=100,
                 callback=None):
        self.device = device
        self.dtype = dtype

        self.k = init_k
        self.true_neighbour_max = init_k//2
        self.d = None
        self.idx = None
        self.callback = callback

    @staticmethod
    def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers = -1):
        """
        Uses cKDTree to compute potential nearest-neighbors of each cell

        Parameters
        ----------
        x : array_like
            Position of each cell in 3D space
        k : list of integer or integer
            The list of k-th nearest neighbors to return. If k is an integer it is treated as a list of [1, ... k] (range(1, k+1)). Note that the counting starts from 1.
        distance_upper_bound : nonnegative float, optional
            Return only neighbors within this distance. This is used to prune tree searches, so if you are doing a series of nearest-neighbor queries, it may help to supply the distance to the nearest neighbor of the most recent point. Default: np.inf
        workers: int, optional
            Number of workers to use for parallel processing. If -1 is given, all CPU threads are used. Default: -1.

        Returns
        ----------
        d : array
            distance from each cell to each of its potential neighbors
        idx : array
            index of each cell's potential neighbors
        """
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, n_jobs=workers)
        return d[:, 1:], idx[:, 1:]

    def find_true_neighbours(self, d, dx):
        """
        Finds the true neighbors of each cell

        Parameters
        ----------
        d : array
            distance from each cell to each of its potential neighbors
        dx : array
            displacement vector from each cell to each of its potential neighbors
        """
        with torch.no_grad():
            z_masks = []
            i0 = 0
            batch_size = 250
            i1 = batch_size
            while True:
                if i0 >= dx.shape[0]:
                    break
                # ?
                n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
                # ??
                n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

                z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0
                z_masks.append(z_mask)

                if i1 > dx.shape[0]:
                    break
                i0 = i1
                i1 += batch_size
        z_mask = torch.cat(z_masks, dim=0)
        return z_mask

    def potential(self, x, p, q, idx, d, lam, potential):
        """
        Computes the potential energy of the system
        
        Parameters
        ----------
        x : torch.Tensor
            Position of each cell in 3D space
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        idx : array_like
            indices of potential nearest-neighbors of each cell
        d : array_like
            distances from each cell to each of the potential nearest-neighbors specified by idx
        lam : array
            array of weights for the terms that make up the potential
        potential : callable
            function that computes the value of the potential between two cells, i and j
            call signature (x, d, dx, lam_i, lam_j, pi, pj, qi, qj)
        
        Returns
        ----------
        V : torch.Tensor
            value of the potential
        m : int
            largest number of true neighbors of any cell
        """
        # Find neighbours
        full_n_list = x[idx]

        dx = x[:, None, :] - full_n_list
        z_mask = self.find_true_neighbours(d, dx)

        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)

        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(idx, 1, sort_idx)

        m = torch.max(torch.sum(z_mask, dim=1)) + 1

        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        # Calculate S
        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        lam_i = lam[:, None, :].expand(p.shape[0], idx.shape[1], lam.shape[1])
        lam_j = lam[idx]

        Vij = potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj)
        V = torch.sum(z_mask.float() * Vij)

        return V, int(m)

    def init_simulation(self, dt, lam, p, q, x, beta):
        """
        Checks input dimensions, cleans and converts into torch.Tensor types

        Parameters
        ----------
        dt : float
            size of time step for simulation
        lam : array
            weights for the terms of the potential function, possibly different for each cell.
        p : array_like
            AB polarity vector of each cell
        q : array_like
            PCP vector of each cell
        x : array_like
            Position of each cell in 3D space
        beta : array_like
            for each cell, probability of division per unit time

        Returns
        ----------
        lam : torch.Tensor
            weights for the terms of the potential function for each cell.
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        sqrt_dt : float
            square root of dt. To be used for normalizing the size of the noise added per time step
        x : torch.Tensor
            Position of each cell in 3D space
        beta : torch.Tensor
            for each cell, probability of division per unit time
        """
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(beta) == len(x)

        sqrt_dt = np.sqrt(dt)
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)

        beta = torch.tensor(beta, dtype=self.dtype, device=self.device)

        lam = torch.tensor(lam, dtype=self.dtype, device=self.device)
        # if lam is not given per-cell, return an expanded view
        if len(lam.shape) == 1:
            lam = lam.expand(x.shape[0], lam.shape[0]).clone()
        return lam, p, q, sqrt_dt, x, beta

    def update_k(self, true_neighbour_max, tstep):
        """
        Dynamically adjusts the number of neighbors to look for.

        If very few of the potential neighbors turned out to be true, you can look for fewer potential neighbors next time.
        If very many of the potential neighbors turned out to be true, you should look for more potential neighbors next time.

        Parameters
        ----------
        true_neighbor_max : int
            largest number of true neighbors of any cell found most recently
        tstep : int
            how many time steps of simulation have elapsed

        Returns
        ----------
        k : int
            new max number of potential neighbors to seek
        n_update : int
            controls when to next check for potential neighbors
        """
        k = self.k
        fraction = true_neighbour_max / k
        if fraction < 0.25:
            k = int(0.75 * k)
        elif fraction > 0.75:
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
        self.k = k
        return k, n_update

    def time_step(self, dt, eta, lam, beta, p, q, sqrt_dt, tstep, x, potential):
        """
        Move the simulation forward by one time step

        Parameters
        ----------
        dt : float
            Size of the time step
        eta : float
            Strength of the added noise
        lam : torch.Tensor
            weights for the terms of the potential function for each cell.
        beta : torch.Tensor
            for each cell, probability of division per unit time
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        sqrt_dt : float
            square root of dt. To be used for normalizing the size of the noise added per time step
        tstep : int
            how many simulation time steps have elapsed
        x : torch.Tensor
            Position of each cell in 3D space
        potential : callable
            function that computes the value of the potential between two cells, i and j
            call signature (x, d, dx, lam_i, lam_j, pi, pj, qi, qj)

        Returns
        ----------
        x : torch.Tensor
            Position of each cell in 3D space
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        lam : torch.Tensor
            weights for the terms of the potential function for each cell.
        beta : torch.Tensor
            for each cell, probability of division per unit time
        """
        # Start with cell division
        division, x, p, q, lam, beta = self.cell_division(x, p, q, lam, beta, dt)

        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        k, n_update = self.update_k(self.true_neighbour_max, tstep)
        k = min(k, len(x) - 1)

        if division or tstep % n_update == 0 or self.idx is None:
            d, idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
        idx = self.idx
        d = self.d

        # Normalise p, q
        with torch.no_grad():
            p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]
            q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]

        # Calculate potential
        V, self.true_neighbour_max = self.potential(x, p, q, idx, d, lam, potential=potential)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x += -x.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
            p += -p.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
            q += -q.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt

            if self.callback is not None:
                self.callback(tstep * dt, x, p, q, lam)

        # Zero gradients
        x.grad.zero_()
        p.grad.zero_()
        q.grad.zero_()

        return x, p, q, lam, beta

    def simulation(self, x, p, q, lam, beta, eta, potential, yield_every=1, dt=0.1):
        """
        Generator to implement the simulation

        Parameters
        ----------
        x : torch.Tensor
            Position of each cell in 3D space
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        lam : torch.Tensor
            weights for the terms of the potential function for each cell.
        beta : torch.Tensor
            for each cell, probability of division per unit time
        eta : float
            Strength of the added noise
        potential : callable
            function that computes the value of the potential between two cells, i and j
            call signature (x, d, dx, lam_i, lam_j, pi, pj, qi, qj)
        yield_every : int
            How many simulation time steps to take between yielding the system state
        dt : float, optional
            Size of the time step. Default: 0.1

        Yields
        ----------
        x : numpy.ndarray
            Position of each cell in 3D space
        p : numpy.ndarray
            AB polarity vector of each cell
        q : numpy.ndarray
            PCP vector of each cell
        lam : numpy.ndarray
            weights for the terms of the potential function for each cell.
        """
        lam, p, q, sqrt_dt, x, beta = self.init_simulation(dt, lam, p, q, x, beta)

        tstep = 0
        while True:
            tstep += 1
            x, p, q, lam, beta = self.time_step(dt, eta, lam, beta, p, q, sqrt_dt, tstep, x, potential=potential)

            if tstep % yield_every == 0:
                xx = x.detach().to("cpu").numpy().copy()
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()
                ll = lam.detach().to("cpu").numpy().copy()
                yield xx, pp, qq, ll

    @staticmethod
    def cell_division(x, p, q, lam, beta, dt, beta_decay = 1.0):
        """
        Decides which cells divide, and if they do, places daughter cells.
        If a cell divides, one daughter cell is placed at the same position as the parent cell, and the other is placed one cell diameter away in a uniformly random direction

        Parameters
        ----------
        x : torch.Tensor
            Position of each cell in 3D space
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        lam : torch.Tensor
            weights for the terms of the potential function for each cell.
        beta : torch.Tensor
            for each cell, probability of division per unit time
        dt : float
            Size of the time step.

        Returns
        ---------
        division : bool
            True if cell division has taken place, otherwise False
        x : torch.Tensor
            Position of each cell in 3D space
        p : torch.Tensor
            AB polarity vector of each cell
        q : torch.Tensor
            PCP vector of each cell
        lam : torch.Tensor
            weights for the terms of the potential function for each cell.
        beta : torch.Tensor
            for each cell, probability of division per unit time
        """
        if torch.sum(beta) < 1e-5:
            return False, x, p, q, lam, beta

        # set probability according to beta and dt
        d_prob = beta * dt
        # flip coins
        draw = torch.empty_like(beta).uniform_()
        # find successes
        events = draw < d_prob
        division = False

        if torch.sum(events) > 0:
            with torch.no_grad():
                division = True
                # find cells that will divide
                idx = torch.nonzero(events)[:, 0]

                x0 = x[idx, :]
                p0 = p[idx, :]
                q0 = q[idx, :]
                l0 = lam[idx, :]
                b0 = beta[idx] * beta_decay

                # make a random vector and normalize to get a random direction
                move = torch.empty_like(x0).normal_()
                move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

                # place new cells
                x0 = x0 + move

                # append new cell data to the system state
                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                lam = torch.cat((lam, l0))
                beta = torch.cat((beta, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, lam, beta
