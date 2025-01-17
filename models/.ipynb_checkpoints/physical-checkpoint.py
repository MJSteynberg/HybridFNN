import torch
import torch.nn as nn


class PoissonEquation(nn.Module):
    def __init__(self, device, L, N, num_gaussians, omega=0.5, tol=1e-5, max_iter=10000, alpha=None):
        super(PoissonEquation, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        self.dx = L / (N - 1)
        self.omega = omega # In (0, 1)
        self.tol = tol
        self.max_iter = max_iter
        
        self.num_params = 4 * num_gaussians  # pos_x, pos_y, sigma, amplitude
        self.num_gaussians = num_gaussians
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape != (self.num_params,):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.ones((self.num_params,), dtype=torch.float32, device=self.device))

    def create_diffusion_map(self):
        # Placeholder for diffusion map creation logic
        return torch.ones((self.N, self.N), device=self.device)

    def step(self, f, u):
        """
        Perform a single SOR step.
        """
        u_new = u.clone()
        u_new[1:-1, 1:-1] = (1 - self.omega) * u[1:-1, 1:-1] + self.omega * 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - self.dx**2*f[1:-1, 1:-1]
        )
        return u_new

    def forward(self, f):
        """
        Solve the Poisson equation using SOR.
        """
        self.diffusion_map = self.create_diffusion_map()
        u = torch.zeros((self.N, self.N), device=self.device, requires_grad=True)
        for iteration in range(self.max_iter):
            u_new = self.step(f, u)
            if iteration % 500 == 0:
                norm = torch.norm(u_new - u).item()
                print(f"Iteration {iteration}, Residual: {norm}")
                if norm < self.tol:
                    print(f'Converged in {iteration} iterations')
                    break
            u = u_new.clone()

        # Apply Dirichlet boundary conditions
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
        return u


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L = 1.0
    N = 50
    num_gaussians = 5
    f = torch.rand((N, N), device=device)
    solver = PoissonEquation(device, L, N, num_gaussians)
    solution = solver(f)

    # Uncomment for visualization if desired
    # import matplotlib.pyplot as plt
    # plt.imshow(solution.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Solution to the Poisson Equation')
    # plt.show()
