#%% cell 0
import matplotlib.pyplot as plt
import numpy as np
import math


dataO = np.random.uniform(0, 1, [200,2])
dataT = np.random.uniform(100, 105, [200,2])

plt.subplot(121)
plt.plot(dataO)
plt.title('First Data')
plt.subplot(122)
plt.plot(dataT)
plt.title('Seconde Data')
plt.show()
