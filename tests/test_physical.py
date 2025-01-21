# To verify that the finite difference method is correct

import torch
import torch.nn as nn
import os
import sys
import sympy as sp
# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from physical import PoissonEquation

def test_poisson_equation():
    device = torch.device("cpu")
    L = 2  # Domain size [-3, 3]
    N = 100  # Grid resolution
    num_gaussians = 1  # Number of Gaussians in the diffusion map

    
    # Define the exact solution and its RHS
    def exact_solution(x, y):
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    
    def diffusion_map(x, y):
        return 1 + torch.exp(-((x)**2 + (y)**2))

    def rhs_function_calculation_gaussian():
        # use simpy to derive the corect rhs
        x, y = sp.symbols('x y')
        u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
        a = 1 + sp.exp(-((x)**2 + (y)**2))
        u_x = sp.diff(u, x)
        u_y = sp.diff(u, y)
        au_xx = sp.diff(a * u_x, x)
        au_yy = sp.diff(a * u_y, y)
        rhs = au_xx + au_yy
        print(rhs)


    def rhs_function_gauss(x, y):
        return -(-2*torch.pi*x*torch.exp(-x**2 - y**2)*torch.sin(torch.pi*y)*torch.cos(torch.pi*x) - 2*torch.pi*y*torch.exp(-x**2 - y**2)*torch.sin(torch.pi*x)*torch.cos(torch.pi*y) - 2*torch.pi**2*(torch.exp(-x**2 - y**2) + 1)*torch.sin(torch.pi*x)*torch.sin(torch.pi*y))


    alpha = torch.tensor([1,0,0,1]).float()

    # Initialize the PoissonEquation class
    poisson_solver = PoissonEquation(
        device=device,
        L=L,
        N=N,
        num_gaussians=num_gaussians,
        func=rhs_function_gauss,
        verbose=True,
        alpha=alpha
    ).to(device)

    # Solve the Poisson equation
    u_numerical = poisson_solver()

    # Compute the exact solution on the grid
    x, y = torch.meshgrid(
        torch.linspace(-L / 2, L / 2, N, device=device),
        torch.linspace(-L / 2, L / 2, N, device=device),
        indexing="ij",
    )
    u_exact = exact_solution(x, y).flatten()

    # Compute the error
    error = torch.norm(u_numerical - u_exact) / torch.norm(u_exact)

    # Print the results
    print(f"Relative error: {error:.6e}")

    # Optional: Visualize the numerical and exact solutions
    import matplotlib.pyplot as plt

    u_numerical = u_numerical.reshape((N, N)).cpu().detach().numpy()
    u_exact = u_exact.reshape((N, N)).cpu().detach().numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(u_numerical, extent=(-L / 2, L / 2, -L / 2, L / 2), origin="lower")
    plt.colorbar()
    plt.title("Numerical Solution")
    plt.subplot(1, 2, 2)
    plt.imshow(u_exact, extent=(-L / 2, L / 2, -L / 2, L / 2), origin="lower")
    plt.colorbar()
    plt.title("Exact Solution")
    plt.savefig("solution_comparison.png")

# Run the test
test_poisson_equation()
