import torch
import torch.nn as nn

def gaussian(param, num_gaussians=1, N=100, L=6):
    # interpolate alpha1 to the grid
    gaussian_map = torch.ones((N, N), dtype=torch.float32, device=param.device)
    
    x = torch.linspace(-L//2, L//2, N, device=param.device)
    y = torch.linspace(-L//2, L//2, N, device=param.device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    for i in range(num_gaussians):
        gaussian_map += param[i] * torch.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2*num_gaussians + i]) ** 2))
    return gaussian_map

class PoissonEquation(nn.Module):
    def __init__(self, device, L, N, num_gaussians, func, alpha=None, verbose=False):
        super(PoissonEquation, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        self.dx = L / (N - 1)
        self.num_gaussians = num_gaussians
        self.func = func
        self.verbose = verbose
        
        self.num_params = 4 * num_gaussians
        self.alpha = nn.Parameter(
            alpha if alpha is not None else torch.rand((self.num_params,), dtype=torch.float32, device=device)
        )
        if alpha is not None and self.alpha.shape != (self.num_params,):
            raise ValueError("Shape of alpha does not match the parameters.")
        
        # Precompute meshgrid
        self.x, self.y = torch.meshgrid(
            torch.linspace(-L / 2, L / 2, N, device=device),
            torch.linspace(-L / 2, L / 2, N, device=device),
            indexing="ij",
        )
        
        # Placeholder for cached matrices
        self.D2 = None
        self.Dx, self.Dy = None, None

    def _create_diffusion_map(self):
        """
        Create the diffusion map as a tensor operation.
        """
        diffusion_map = gaussian(self.alpha, num_gaussians=self.num_gaussians, N=self.N, L=self.L)

        # plot diffusion map
        if self.verbose:
            import matplotlib.pyplot as plt
            plt.imshow(diffusion_map.cpu().detach().numpy(), extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("Diffusion map")
            plt.savefig("diffusion.png")
            plt.clf()
        return diffusion_map

    def _assemble_rhs(self):
        """
        Assemble the right-hand side vector f.
        """
        f = self.func(self.x, self.y)
        return f.flatten() * self.dx ** 2

    def _assemble_D2(self):
        """
        Assemble the 5-point stencil matrix using sparse tensors.
        """
        N2 = self.N * self.N
        indices = []
        values = []
        
        for i in range(self.N):
            for j in range(self.N):
                idx = i * self.N + j
                indices.append((idx, idx))
                values.append(-4.0)
                if i > 0:  # Top
                    indices.append((idx, idx - self.N))
                    values.append(1.0)
                if i < self.N - 1:  # Bottom
                    indices.append((idx, idx + self.N))
                    values.append(1.0)
                if j > 0:  # Left
                    indices.append((idx, idx - 1))
                    values.append(1.0)
                if j < self.N - 1:  # Right
                    indices.append((idx, idx + 1))
                    values.append(1.0)
        
        indices = torch.tensor(indices, dtype=torch.long, device=self.device).T
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        D2 = torch.sparse_coo_tensor(indices, values, (N2, N2))
        return D2

    def _assemble_D(self):
        """
        Assemble Dx and Dy matrices.
        """

        N2 = self.N * self.N
        indices_x, values_x = [], []
        indices_y, values_y = [], []

        for i in range(self.N):
            for j in range(self.N):
                idx = i * self.N + j
                if i > 0:  # Top
                    indices_x.append((idx, idx - self.N))
                    values_x.append(1.0)
                if i < self.N - 1:  # Bottom
                    indices_x.append((idx, idx + self.N))
                    values_x.append(-1.0)
                if j > 0:  # Left
                    indices_y.append((idx, idx - 1))
                    values_y.append(1.0)
                if j < self.N - 1:  # Right
                    indices_y.append((idx, idx + 1))
                    values_y.append(-1.0)



        indices_x = torch.tensor(indices_x, dtype=torch.long, device=self.device).T
        values_x = torch.tensor(values_x, dtype=torch.float32, device=self.device)
        indices_y = torch.tensor(indices_y, dtype=torch.long, device=self.device).T
        values_y = torch.tensor(values_y, dtype=torch.float32, device=self.device)

        self.Dx = torch.sparse_coo_tensor(indices_x, values_x, (N2, N2)) / 2
        self.Dy = torch.sparse_coo_tensor(indices_y, values_y, (N2, N2)) / 2
        return self.Dx, self.Dy

    def _assemble_lhs(self):
        """
        Assemble the left-hand side matrix with element-wise multiplication instead of diagonal matrices.
        """
        if self.D2 is None:
            self.D2 = self._assemble_D2()
        if self.Dx is None or self.Dy is None:
            self.Dx, self.Dy = self._assemble_D()

        # Flatten the diffusion map
        diffusion_map = self._create_diffusion_map()
        D2 = self.D2.to_dense()
        Dx = self.Dx.to_dense()
        Dy = self.Dy.to_dense()

        # plot Dx @ diffusion_map + the actual derivative of alpha
        if self.verbose:
            import matplotlib.pyplot as plt
            plt.imshow((-(Dx @ diffusion_map.flatten())/self.dx).cpu().detach().numpy().reshape(self.N, self.N)[1:-1,1:-1], extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("Dx @ diffusion_map")
            plt.savefig("Dx_diffusion.png")
            plt.clf()
            plt.imshow(-(Dy @ diffusion_map.flatten()/self.dx).cpu().detach().numpy().reshape(self.N, self.N)[1:-1,1:-1], extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("Dy @ diffusion_map")
            plt.savefig("Dy_diffusion.png")

            plt.clf()
            plt.imshow(torch.gradient(diffusion_map, spacing=(self.dx,), dim = 0)[0].cpu().detach().numpy(), extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("d/dx diffusion_map")
            plt.savefig("d_dx_diffusion.png")
            plt.clf()

            plt.imshow(torch.gradient(diffusion_map, spacing=(self.dx,), dim = 1)[0].cpu().detach().numpy(), extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("d/dy diffusion_map")
            plt.savefig("d_dy_diffusion.png")
            plt.clf()



        # Assemble the left-hand side matrix
        d_diff_x = (Dx @ diffusion_map.flatten()).reshape(self.N, self.N)
        d_diff_y = (Dy @ diffusion_map.flatten()).reshape(self.N, self.N)
        # set the boundary to zero
        d_diff_x[0, :] = 0
        d_diff_x[-1, :] = 0
        d_diff_x[:, 0] = 0
        d_diff_x[:, -1] = 0
        d_diff_y[0, :] = 0
        d_diff_y[-1, :] = 0
        d_diff_y[:, 0] = 0
        d_diff_y[:, -1] = 0



        lhs = -(
            diffusion_map.flatten()[:,None] * D2
             + d_diff_x.flatten()[:,None] * Dx
             + d_diff_y.flatten()[:,None] * Dy
        )
        return lhs.to_dense()


    def penalization(self):
        """
        Penalize negative values in the Gaussian amplitudes.
        """
        return torch.sum(torch.relu(-self.alpha[: self.num_gaussians]))

    def forward(self):
        """
        Solve the Poisson equation.
        """
        if self.verbose:
            print("Setting up LHS and RHS.")
        lhs = self._assemble_lhs()
        rhs = self._assemble_rhs()

        if self.verbose:
            print("Solving the linear system.")
  
        u = torch.linalg.solve(lhs, rhs)

        return u
    
class HelmholtzEquation(nn.Module):
    def __init__(self, device, L, N, num_gaussians, func, alpha=None, k = None, verbose=False):
        super(HelmholtzEquation, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        self.dx = L / (N - 1)
        self.num_gaussians = num_gaussians
        self.func = func
        self.verbose = verbose
        
        self.num_params = 4 * num_gaussians

        self.alpha = nn.Parameter(
            alpha if alpha is not None else torch.rand((self.num_params,), dtype=torch.float32, device=device)
        )
        if alpha is not None and self.alpha.shape != (self.num_params,):
            raise ValueError("Shape of alpha does not match the parameters.")
        
        self.k = nn.Parameter(
            k if k is not None else torch.rand((self.num_params,), dtype=torch.float32, device=device)
        )
        if k is not None and self.k.shape != (self.num_params,):
            raise ValueError("Shape of k does not match the parameters.")
        
        # Precompute meshgrid
        self.x, self.y = torch.meshgrid(
            torch.linspace(-L / 2, L / 2, N, device=device),
            torch.linspace(-L / 2, L / 2, N, device=device),
            indexing="ij",
        )
        
        # Placeholder for cached matrices
        self.D2 = None
        self.Dx, self.Dy = None, None
        

    def _create_diffusion_map(self):
        """
        Create the diffusion map as a tensor operation.
        """
        diffusion_map = gaussian(self.alpha, num_gaussians=self.num_gaussians, N=self.N, L=self.L)

        # plot diffusion map
        if self.verbose:
            import matplotlib.pyplot as plt
            plt.imshow(diffusion_map.cpu().detach().numpy(), extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("Diffusion map")
            plt.savefig("diffusion.png")
            plt.clf()
        return diffusion_map
    
    def _create_k_map(self):
        k_map = gaussian(self.k, num_gaussians=self.num_gaussians, N=self.N, L=self.L)
        return k_map ** 2 * self.dx ** 2

    def _assemble_rhs(self):
        """
        Assemble the right-hand side vector f.
        """
        f = self.func(self.x, self.y)
        return f.flatten() * self.dx ** 2

    def _assemble_D2(self):
        """
        Assemble the 5-point stencil matrix using sparse tensors.
        """
        N2 = self.N * self.N
        indices = []
        values = []
        
        for i in range(self.N):
            for j in range(self.N):
                idx = i * self.N + j
                indices.append((idx, idx))
                values.append(-4.0)
                if i > 0:  # Top
                    indices.append((idx, idx - self.N))
                    values.append(1.0)
                if i < self.N - 1:  # Bottom
                    indices.append((idx, idx + self.N))
                    values.append(1.0)
                if j > 0:  # Left
                    indices.append((idx, idx - 1))
                    values.append(1.0)
                if j < self.N - 1:  # Right
                    indices.append((idx, idx + 1))
                    values.append(1.0)
        
        indices = torch.tensor(indices, dtype=torch.long, device=self.device).T
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        D2 = torch.sparse_coo_tensor(indices, values, (N2, N2))
        return D2

    def _assemble_D(self):
        """
        Assemble Dx and Dy matrices.
        """

        N2 = self.N * self.N
        indices_x, values_x = [], []
        indices_y, values_y = [], []

        for i in range(self.N):
            for j in range(self.N):
                idx = i * self.N + j
                if i > 0:  # Top
                    indices_x.append((idx, idx - self.N))
                    values_x.append(1.0)
                if i < self.N - 1:  # Bottom
                    indices_x.append((idx, idx + self.N))
                    values_x.append(-1.0)
                if j > 0:  # Left
                    indices_y.append((idx, idx - 1))
                    values_y.append(1.0)
                if j < self.N - 1:  # Right
                    indices_y.append((idx, idx + 1))
                    values_y.append(-1.0)



        indices_x = torch.tensor(indices_x, dtype=torch.long, device=self.device).T
        values_x = torch.tensor(values_x, dtype=torch.float32, device=self.device)
        indices_y = torch.tensor(indices_y, dtype=torch.long, device=self.device).T
        values_y = torch.tensor(values_y, dtype=torch.float32, device=self.device)

        self.Dx = torch.sparse_coo_tensor(indices_x, values_x, (N2, N2)) / 2
        self.Dy = torch.sparse_coo_tensor(indices_y, values_y, (N2, N2)) / 2
        return self.Dx, self.Dy
        


    def _assemble_lhs(self):
        """
        Assemble the left-hand side matrix with element-wise multiplication instead of diagonal matrices.
        """
        if self.D2 is None:
            self.D2 = self._assemble_D2()
        if self.Dx is None or self.Dy is None:
            self.Dx, self.Dy = self._assemble_D()

        # Flatten the diffusion map
        diffusion_map = self._create_diffusion_map()
        k_map = self._create_k_map()
        D2 = self.D2.to_dense()
        Dx = self.Dx.to_dense()
        Dy = self.Dy.to_dense()
        K = k_map.flatten()[:,None]
       
        

        # plot Dx @ diffusion_map + the actual derivative of alpha
        if self.verbose:
            import matplotlib.pyplot as plt
            plt.imshow((-(Dx @ diffusion_map.flatten())/self.dx).cpu().detach().numpy().reshape(self.N, self.N)[1:-1,1:-1], extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("Dx @ diffusion_map")
            plt.savefig("Dx_diffusion.png")
            plt.clf()
            plt.imshow(-(Dy @ diffusion_map.flatten()/self.dx).cpu().detach().numpy().reshape(self.N, self.N)[1:-1,1:-1], extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("Dy @ diffusion_map")
            plt.savefig("Dy_diffusion.png")

            plt.clf()
            plt.imshow(torch.gradient(diffusion_map, spacing=(self.dx,), dim = 0)[0].cpu().detach().numpy(), extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("d/dx diffusion_map")
            plt.savefig("d_dx_diffusion.png")
            plt.clf()

            plt.imshow(torch.gradient(diffusion_map, spacing=(self.dx,), dim = 1)[0].cpu().detach().numpy(), extent=(-self.L / 2, self.L / 2, -self.L / 2, self.L / 2), origin='lower')
            plt.colorbar()
            plt.title("d/dy diffusion_map")
            plt.savefig("d_dy_diffusion.png")
            plt.clf()



        # Assemble the left-hand side matrix
        d_diff_x = (Dx @ diffusion_map.flatten()).reshape(self.N, self.N)
        d_diff_y = (Dy @ diffusion_map.flatten()).reshape(self.N, self.N)
        # set the boundary to zero
        d_diff_x[0, :] = 0
        d_diff_x[-1, :] = 0
        d_diff_x[:, 0] = 0
        d_diff_x[:, -1] = 0
        d_diff_y[0, :] = 0
        d_diff_y[-1, :] = 0
        d_diff_y[:, 0] = 0
        d_diff_y[:, -1] = 0



        lhs = -(
            diffusion_map.flatten()[:,None] * D2
             + K
             + d_diff_x.flatten()[:,None] * Dx
             + d_diff_y.flatten()[:,None] * Dy
        )
        return lhs.to_dense()


    def penalization(self):
        """
        Penalize negative values in the Gaussian amplitudes.
        """
        return torch.sum(torch.relu(-self.alpha[: self.num_gaussians])) + torch.sum(torch.relu(-self.k[: self.num_gaussians]))

    def forward(self):
        """
        Solve the Poisson equation.
        """
        if self.verbose:
            print("Setting up LHS and RHS.")
        lhs = self._assemble_lhs()
        rhs = self._assemble_rhs()

        if self.verbose:
            print("Solving the linear system.")
  
        u = torch.linalg.solve(lhs, rhs)

        return u



