from matplotlib import pyplot as plt
import numpy as np
from dtw import dtw


x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 0, 0, 1, 1, 2, 4, 2, 1, 2]).reshape(-1, 1)
# y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)
dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0]-0.5))
plt.ylim((-0.5, acc.shape[1]-0.5))

plt.show()