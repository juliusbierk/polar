import numpy as np
import pandas as pd
import plotly.express as px

import pickle

fname = input('Enter data filename: ')#'data/test1.pkl'

print('Loading data from file '+fname)
with open(fname, 'rb') as f:
    data = pickle.load(f)  # contains x, p, q, lam
# data[t][0] == x, x[i, k] = position of particle i in dimension k
# data[t][1] == p, p[i, k] = AB polarity of particle i in dimension k
# data[t][2] == q, q[i, k] = PCP of particle i in dimension k

# create dataframe
row_chunks = list()
for t, dat in enumerate(data):
    n = dat[0].shape[0]
    row_chunks.append(np.hstack([np.ones((n,1)) * t, np.arange(n)[:,np.newaxis], dat[0], dat[1], dat[2]]))

df = pd.DataFrame(np.vstack(row_chunks), columns = ['t', 'i', 'x1', 'x2', 'x3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3'])

fig = px.scatter_3d(df, x='x1', y = 'x2', z = 'x3', animation_frame = 't', color = 'x1')

fig.write_html('animations/'+fname.split('/')[1].split('.')[0]+'.html', include_plotlyjs = 'directory', full_html = False, animation_opts = {'frame':{'duration':100}})
