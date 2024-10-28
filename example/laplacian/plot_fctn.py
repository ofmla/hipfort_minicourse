import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the test function based on the CUDA kernel logic
def test_function(x, y, z, Lx, Ly, Lz):
    c = 0.5
    return c * (x * (x - Lx) + y * (y - Ly) + z * (z - Lz))

# Define the grid size and steps (equivalent to hx, hy, hz)
nx, ny, nz = 50, 50, 50
Lx, Ly, Lz = 1.0, 1.0, 1.0

# Create mesh grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)

# Create 3D meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Pick a fixed z value for a 2D slice
Z_value = 0.5 * Lz
Z = test_function(X, Y, Z_value, Lx, Ly, Lz)

# Plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

# Labels and title
ax.set_title('3D Plot of the Test Function (Slice at z=0.5*Lz)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x, y, z)')

plt.show()

