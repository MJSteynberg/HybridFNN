import torch 
import torch.nn as nn 
import os
import sys
import sympy as sp
import numpy as np
# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from synthetic import NeuralNetwork2D

class PINN(NeuralNetwork2D):
    def __init__(self, data_dim, hidden_dim, num_layers, alpha, kappa, activation=nn.Tanh, device='cpu'):
        super().__init__(data_dim, hidden_dim, num_layers, activation, device)
        self.alpha = nn.Parameter(alpha)
        self.kappa = nn.Parameter(kappa)

    def gaussian_kernel(self, x, y, param, num_gaussians):
        gaussian_map = 1
        for i in range(num_gaussians):
            gaussian_map += param[i] * torch.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2*num_gaussians + i]) ** 2))
        return gaussian_map

        
    def forward(self, x):
        return self.network(x)
    
    def synth_loss(self, data, rhs, loss_func):
        # Compute the solution
        solution = self(data[:, :2])
        # Compute the loss
        loss = loss_func(solution, data[:, 2]) 
        return loss
    
    def phys_loss(self, data, _rhs, loss_func):
        # Generate collocation points in the domain
        collocation_points = 6 * torch.rand((50, 2), requires_grad=True).to(self.device) - 3
        # Compute the synthetic solution on the collocation points
        solution = self(collocation_points)
        
        # compute the laplacian of the solution
        u_x = torch.autograd.grad(solution, collocation_points, grad_outputs=torch.ones_like(solution), create_graph=True)[0][:,0]
        au_x = self.gaussian_kernel(collocation_points[:,0], collocation_points[:,1], self.alpha, 1) * u_x
        au_xx = torch.autograd.grad(au_x, collocation_points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0]

        u_y = torch.autograd.grad(solution, collocation_points, grad_outputs=torch.ones_like(solution), create_graph=True)[0][:,1]
        au_y = self.gaussian_kernel(collocation_points[:,0], collocation_points[:,1], self.alpha, 1) * u_y
        au_yy = torch.autograd.grad(au_y, collocation_points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,1]

        lhs = au_xx + au_yy + self.gaussian_kernel(collocation_points[:,0], collocation_points[:,1], self.kappa, 1) * solution
        rhs = _rhs(collocation_points[:,0], collocation_points[:,1])

        # Enforce Dirichlet boundary conditions
        # Generate collocation points on the boundary
        x_b = torch.rand((10, 1), requires_grad=True).to(self.device) * 6 - 3
        y_b = torch.rand((10, 1), requires_grad=True).to(self.device) * 6 - 3
        collocation_points_b = torch.cat((torch.cat((x_b, -3 * torch.ones_like(x_b)), dim=1), torch.cat((x_b, 3 * torch.ones_like(x_b)), dim=1), torch.cat((-3 * torch.ones_like(y_b), y_b), dim=1), torch.cat((3 * torch.ones_like(y_b), y_b), dim=1)))
        # plot points to show where they are:
        plt.clf()
        plt.scatter(collocation_points_b[:,0].detach().numpy(), collocation_points_b[:,1].detach().numpy())
        plt.scatter(collocation_points[:,0].detach().numpy(), collocation_points[:,1].detach().numpy())
        plt.savefig('points.png')

        loss = loss_func(lhs, rhs) + loss_func(self(collocation_points_b), torch.zeros_like(collocation_points_b[:,0].reshape(-1,1)))
        return loss
    

class PINN_Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_func = nn.MSELoss()
    
    def train(self, data, rhs, num_epochs):
        for epoch in range(num_epochs):
            loss = self._train_epoch(data, rhs)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}: alpha = {self.model.alpha}, kappa = {self.model.kappa}")
        
    def _train_epoch(self, data, rhs):
        
        # Zero the gradients
        loss_data = self.model.synth_loss(data, rhs, self.loss_func)
        loss_col = self.model.phys_loss(data, rhs, self.loss_func)

        
        loss = loss_col + loss_data
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()


def rhs(x, y):
    # a gaussian source
    return (4 * torch.exp(-((x - 2) ** 2 + (y - 2) ** 2)) + 4 * torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)))

if __name__ == '__main__':
    from generate_data import generate_data 
    # Note this only works when the parameters are initialized very close to their actual position. As it is now, it faisl miserably

    data_dim = 1
    hidden_dim = 500
    num_layers = 4
    alpha_orig = [6, -1,1,  1.0]
    k_orig = [4, 1, -1, 1.0]
    alpha = torch.tensor(alpha_orig, requires_grad=True)
    kappa = torch.tensor(k_orig, requires_grad=True)
    data = torch.from_numpy(np.loadtxt('data/diffusion/pinn/data.txt')).float()

    

    model = PINN(data_dim, hidden_dim, num_layers, alpha, kappa)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    device = 'cpu'
    model.to(device)
    trainer = PINN_Trainer(model, optimizer, scheduler, device)
    trainer.train(data, rhs, num_epochs=1000)

    # plot the solution
    import matplotlib.pyplot as plt
    x = torch.linspace(-3, 3, 100)
    y = torch.linspace(-3, 3, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack((X.flatten(), Y.flatten()), dim=1)
    Z = model(xy).reshape(100, 100)
    plt.imshow(Z.detach().numpy(), extent=(-3, 3, -3, 3))
    plt.savefig('solution_pinn.png')

