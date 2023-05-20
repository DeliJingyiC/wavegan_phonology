import pickle
from pathlib import Path
import numpy as np

cwd = Path.cwd()
file = cwd / "down_sounds"
file.mkdir(exist_ok=True)
with open("/users/PAS2062/delijingyic/project/wavegan/wavegan/output/22-07-06-182455/latent_v", 'rb') as f:
    for i in f:
        with open(i, 'rb') as p:
            data = pickle.load(p)

for key, val in data.items():
    print(key)
    print(val)
    
