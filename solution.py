import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data1 = np.loadtxt('solution_domain1.txt')
# data2 = np.loadtxt('solution_domain2.txt')
# exact1 = np.loadtxt('exact_domain1.txt')
# exact2 = np.loadtxt('exact_domain2.txt')

# Create figure with two subplots
fig = plt.figure(figsize=(15, 6))

# Plot numerical solution
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_trisurf(data1[:,0], data1[:,1], data1[:,2], cmap='viridis', alpha=0.7)
# surf2 = ax1.plot_trisurf(data2[:,0], data2[:,1], data2[:,2], cmap='viridis', alpha=0.7)
ax1.set_title('Solution numérique (X0)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Solution')

# ;
plt.tight_layout()
plt.show()