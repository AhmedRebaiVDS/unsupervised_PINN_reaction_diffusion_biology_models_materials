import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 2.8e-4
b = 5e-3
tau = .1
k = -.005

# Grid size and step sizes
size = 100
dx = 2. / size

# Total time and number of iterations
T = 15.0
dt = .001
n = int(T / dt)

# Initialize U and V as random arrays
U = np.random.rand(size, size)
V = np.random.rand(size, size)

# Function to compute the Laplacian of an array
def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

# Function to plot the patterns
def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.Reds,
              interpolation='bilinear',
              extent=[-1, 1, -1, 1])
    ax.set_axis_off()
    
# Set up the plot
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
step_plot = n // 15

# Iterate and update the variables
for i in range(n):
    # Compute the Laplacians of U and V
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # Take the values of U and V inside the grid
    Uc = U[1:-1, 1:-1]
    Vc = V[1:-1, 1:-1]
    # Update U and V
    U[1:-1, 1:-1], V[1:-1, 1:-1] = (
        Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k),
        Vc + dt * (b * deltaV + Uc - Vc) / tau
    )
    # Neumann conditions: set derivatives at the edges to zero
    for Z in (U, V):
        Z[0, :] = Z[1, :]
        Z[-1, :] = Z[-2, :]
        Z[:, 0] = Z[:, 1]
        Z[:, -1] = Z[:, -2]

    # Plot the state of the system at 15 different times
    if i % step_plot == 0 and i < 15 * step_plot:
        ax = axes.flat[i // step_plot]
        show_patterns(U, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
