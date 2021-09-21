import numpy as np
from mayavi import mlab
# mlab.options.backend = 'envisage'
import pickle

fname = 'data/test2.pkl'

with open(fname, 'rb') as f:
    data = pickle.load(f)  # contains x, p, q, lam

t = -1

plot = mlab.points3d(data[t][0][:, 0], data[t][0][:, 1], data[t][0][:, 2], data[t][3][:, 1].copy(), scale_factor=2.5,
    scale_mode='none', resolution=8)


@mlab.animate(delay=10)
def anim():
    for t in range(0, len(data), 5):
        # mlab.view(azimuth=-90 + t / 10., distance=390, elevation=10 + t / len(x))

        if t >= len(data):
            t = len(data) - 1

        colors = data[t][0][:, 1].copy()
        # colors[0] = 1

        print(t, '/', len(data), f'  (n = {len(data[t][0])})')
        mask = data[t][0][:, 1] > 0
        plot.mlab_source.reset(x=data[t][0][mask, 0],
                               y=data[t][0][mask, 1],
                               z=data[t][0][mask, 2],
                               scalars=colors[mask])

        yield


anim()
mlab.show()
