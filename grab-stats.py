import numpy as np
import sys
import re

if len(sys.argv) != 2:
    print('usage: {sys.argv[0]} [results file]')
    sys.exit(1)
file = sys.argv[1]

with open(file, 'r') as f:
    content = f.read()

pat = re.compile('Mean accuracy : ([0-9.]+)')
vals = []
for mean in re.findall(pat, content):
    vals.append(float(mean))
vals = np.array(vals).reshape(-1, 8)

print('mean', vals.mean(axis = 1))
print('std', vals.std(axis = 1))
