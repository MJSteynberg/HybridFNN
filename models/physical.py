import torch
import torch.nn as nn

class PoissonEquation(nn.Module):
    def __init__(self, device, L, N, num_gaussians, func, alpha = None):
        super(PoissonEquation, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        self.dx = L / (N - 1)
        
        self.num_params = 4 * num_gaussians  # pos_x, pos_y, sigma, amplitude
        self.num_gaussians = num_gaussians

        self.A = self._assemble_lhs()
        self.f = self._assemble_rhs(func).flatten()
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape != (self.num_params,):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.random((self.num_params,), dtype=torch.float32, device=self.device))

    def _create_diffusion_map(self):
        diffusion_map = torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N, device=self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N, device=self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        for i in range(self.num_gaussians):
            diffusion_map += self.alpha[i] * torch.exp(-((x - self.alpha[self.num_gaussians + i]) ** 2 + (y - self.alpha[2*self.num_gaussians + i]) ** 2))
        return diffusion_map
    
    def _assemble_rhs(self, func):
        f = torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N, device=self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N, device=self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')

        f = func(x, y)
        return f

    def _assemble_lhs(self):
        # Assemble the left-hand side matrix
        # This is a simple 5-point stencil
        A = torch.zeros((self.N * self.N, self.N * self.N), dtype=torch.float32, device=self.device)
        for i in range(self.N):
            for j in range(self.N):
                idx = i * self.N + j
                A[idx, idx] = 4.0
                if i > 0:
                    A[idx, idx - self.N] = -1.0
                if i < self.N - 1:
                    A[idx, idx + self.N] = -1.0
                if j > 0:
                    A[idx, idx - 1] = -1.0
                if j < self.N - 1:
                    A[idx, idx + 1] = -1.0
        return A
        
   
        
    def penalization(self):
        # penalize negative values in the first num_gaussians parameters
        return torch.sum(torch.relu(-self.alpha[:self.num_gaussians]))

    def forward(self, func):
        """
        Solve the Poisson equation using SOR.
        """
        self.diffusion_map = self._create_diffusion_map().flatten()
        u = torch.zeros((self.N*self.N), device=self.device, requires_grad=True)
        

        
        # diffusion map ^T A u = f
        u = torch.linalg.solve(self.diffusion_map * self.A, self.f)

        return u

def rhs(x, y):
    return 0.1*torch.sin(x)*torch.cos(y)

