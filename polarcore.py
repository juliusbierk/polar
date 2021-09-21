import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree


class Polar:
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
    def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf):
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, n_jobs=-1)
        return d[:, 1:], idx[:, 1:]

    def find_true_neighbours(self, d, dx):
        with torch.no_grad():
            z_masks = []
            i0 = 0
            batch_size = 250
            i1 = batch_size
            while True:
                if i0 >= dx.shape[0]:
                    break

                n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
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
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(beta) == len(x)

        sqrt_dt = np.sqrt(dt)
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)

        beta = torch.tensor(beta, dtype=self.dtype, device=self.device)

        lam = torch.tensor(lam, dtype=self.dtype, device=self.device)
        if len(lam.shape) == 1:
            lam = lam.expand(x.shape[0], lam.shape[0]).clone()
        return lam, p, q, sqrt_dt, x, beta

    def update_k(self, true_neighbour_max, tstep):
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
    def cell_division(x, p, q, lam, beta, dt):
        if torch.sum(beta) < 1e-5:
            return False, x, p, q, lam, beta

        d_prob = beta * dt
        draw = torch.empty_like(beta).uniform_()
        events = draw < d_prob
        division = False

        if torch.sum(events) > 0:
            with torch.no_grad():
                division = True
                idx = torch.nonzero(events)[:, 0]

                x0 = x[idx, :]
                p0 = p[idx, :]
                q0 = q[idx, :]
                l0 = lam[idx, :]
                b0 = beta[idx]

                move = torch.empty_like(x0).normal_()
                move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

                x0 = x0 + move

                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                lam = torch.cat((lam, l0))
                beta = torch.cat((beta, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, lam, beta
