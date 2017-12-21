#%% cell 0
import numpy as np
import matplotlib.pyplot as plt

X = np.random.normal(2,1,50)
Y = np.random.normal(5,20,50)

plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random normal cross variable distribution")
plt.xscale('log')
plt.yscale('log')
plt.show()