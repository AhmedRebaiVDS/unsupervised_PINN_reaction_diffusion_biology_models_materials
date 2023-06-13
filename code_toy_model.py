import numpy as np
import matplotlib.pyplot as plt


def solve_pde(a, b, c, Delta_x, Delta_t):
    # Number of grid points in spatial and temporal domains
    N = int(2 / Delta_x)
    M = int(1 / Delta_t)

    # Define grid points
    x = np.linspace(0, 2, N+1)
    t = np.linspace(0, 1, M+1)

    # Initialize solution array
    u = np.zeros((N+1, M+1))

    # Set initial condition
    u[:, 0] = 6 * np.exp(-3 * x)

    # Set boundary conditions
    u[0, :] = 6 * np.exp(-2 * t)
    u[N, :] = 6 * np.exp(-6 - 2 * t)

    # Iterate over temporal domain
    for j in range(M):
        # Iterate over spatial domain
        for i in range(1, N):
            # Use finite difference equation to update solution
            u[i, j+1] = (1 / (1 + 2 * Delta_t / Delta_x)) * (u[i+1, j] - 2 * u[i, j] + u[i-1, j] - 2 * Delta_t / Delta_x * (u[i, j+1] - u[i, j-1]))

    # Return solution
    return u

# Define function to calculate analytical solution
def analytical_solution(x, t):
    return 6 * np.exp(-2 * t - 3 * x)

# Set discretization step sizes
Delta_x = 0.1
Delta_t = 0.01

# Calculate numerical solution
solution = solve_pde(2, 1, -1, Delta_x, Delta_t)

# Define grid points for plotting
x_plot = np.linspace(0, 2, int(2 / Delta_x) + 1)
t_plot = np.linspace(0, 1, int(1 / Delta_t) + 1)
X, T = np.meshgrid(x_plot, t_plot)

# Calculate analytical solution for plotting
analytical = analytical_solution(X, T)

# Plot numerical and analytical solutions
plt.figure()
plt.title("Numerical and analytical solutions")
plt.pcolor(X, T, solution, cmap="coolwarm")
plt.colorbar()
plt.contour(X, T, analytical, colors="k")
plt.xlabel("x")
plt.ylabel("t")
plt.show()
