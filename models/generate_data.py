import os
import sys
import sympy as sp
import numpy as np
# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from physical import HelmholtzEquation
import torch

from training import interpolate_phys_solution
import numpy as np


def generate_data(rhs, N):
    fname = 'data/diffusion/pinn'
    device = torch.device('cpu')
    L = 6.0
    num_gaussians = 1
    alpha = torch.tensor([4, 1, 1, 1.0], device=device).float()
    k = torch.tensor([3, -1, -1, 1], device=device).float()
    solver = HelmholtzEquation(device, L, N, num_gaussians, alpha=alpha, k=k, func=rhs, verbose=True)
    solution = solver().reshape(N,N)

    # Choose sensor positions randomly in [1.5, 2.5]x[1.5, 2.5]
    num_sensors = 500
    sensors =  6*torch.rand((num_sensors, 2)) - 3

    # Interpolate the solution to the sensor positions
    sensor_values = interpolate_phys_solution(sensors, solution)

    # Save the data
    data = torch.cat((sensors, sensor_values), dim=1)
    np.savetxt(fname + 'data.txt', data.cpu().detach().numpy())


    # Uncomment for visualization if desired
    x = torch.linspace(-L/2, L/2, N)
    y = torch.linspace(-L/2, L/2, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    vmin = solution.min()
    vmax = solution.max()

    import matplotlib.pyplot as plt
    plt.scatter(X.cpu().numpy(), Y.cpu().numpy(), c=solution.cpu().detach().numpy(), cmap='viridis', vmin = vmin, vmax = vmax)
    plt.scatter(sensors[:,0].cpu().numpy(), sensors[:,1].cpu().numpy(), c=sensor_values.cpu().detach().numpy(), cmap='viridis', vmin = vmin, vmax = vmax, edgecolors='k')
    plt.colorbar()
    plt.title('Solution to the Poisson Equation')
    plt.savefig(fname + 'solution.png')