import matplotlib.pyplot as plt
import numpy as np
import os

'''
I was getting some weird threading errors from having the plotting code in the same script
as the umap data generation - no idea why.
The easiest solution as just to save the unplotted data instead and have a separate
python script to the plotting with none of the weird threading libraries to mess it up.
'''

for name in [x for x in os.listdir('imgs') if x.endswith('.npy')]:
    with open(f'imgs/{name}', 'rb') as f:
        reduced_embeddings = np.load(f)
        c = np.load(f)

    plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], c=c)
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(f'imgs/{name[:-4]}.png')
    plt.clf()
