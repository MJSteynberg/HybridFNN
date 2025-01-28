import torch 
import torch.nn as nn 
import os
import sys
import sympy as sp
import numpy as np
# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from synthetic import NeuralNetwork2D
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving PDEs.
    Wraps a NeuralNetwork2D model and incorporates physical constraints.
    """
    def __init__(self, network, alpha, kappa, device='cpu'):
        """
        Args:
            network (NeuralNetwork2D): The underlying neural network.
            alpha (float or torch.Tensor): Coefficient for the Gaussian kernel in the PDE.
            kappa (float or torch.Tensor): Coefficient for the solution term in the PDE.
            device (str): Device to run the model ('cpu' or 'cuda').
        """
        super().__init__()
        self.network = network.to(device)
        self.alpha = torch.tensor(alpha, device=device, requires_grad=True) if not isinstance(alpha, torch.Tensor) else alpha
        self.kappa = torch.tensor(kappa, device=device, requires_grad=True) if not isinstance(kappa, torch.Tensor) else kappa
        self.device = device

    def gaussian_kernel(self, x, y, param, num_gaussians):
        """
        Computes a Gaussian kernel map.
        Args:
            x (torch.Tensor): x-coordinates.
            y (torch.Tensor): y-coordinates.
            param (torch.Tensor): Gaussian parameters [amplitudes, x-centers, y-centers].
            num_gaussians (int): Number of Gaussians.
        Returns:
            torch.Tensor: Gaussian kernel map.
        """
        gaussian_map = 1
        for i in range(num_gaussians):
            gaussian_map += param[i] * torch.exp(
                -((x - param[num_gaussians + i]) ** 2 + (y - param[2 * num_gaussians + i]) ** 2)
            )
        return gaussian_map

    def forward(self, x):
        """
        Forward pass through the neural network.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x)

    def synth_loss(self, data, loss_func):
        """
        Compute the synthetic data loss.
        Args:
            data (torch.Tensor): Synthetic data with inputs (x, y) and target values.
            loss_func (nn.Module): Loss function (e.g., MSE).
        Returns:
            torch.Tensor: Loss value.
        """
        inputs = data[:, :2]
        targets = data[:, 2]
        predictions = self(inputs).reshape(-1)
        return 0

    def phys_loss(self, rhs_func, num_collocation=1000, num_boundary=500, loss_func=nn.MSELoss()):
        """
        Compute the physics loss using collocation and boundary points.
        Args:
            rhs_func (callable): Function for the right-hand side of the PDE.
            num_collocation (int): Number of collocation points.
            num_boundary (int): Number of boundary points.
            loss_func (nn.Module): Loss function (e.g., MSE).
        Returns:
            torch.Tensor: Physics loss value.
        """
        # Collocation points in the domain
        collocation_points = 6 * torch.rand((num_collocation, 2), device=self.device, requires_grad=True) - 3

        # Compute solution and its gradients
        solution = self(collocation_points).reshape(-1)

        u_x = torch.autograd.grad(
            solution, collocation_points, grad_outputs=torch.ones_like(solution),
            create_graph=True
        )[0][:, 0]
        au_x = self.gaussian_kernel(collocation_points[:, 0], collocation_points[:, 1], self.alpha, 1) * u_x
        au_xx = torch.autograd.grad(
            au_x, collocation_points, grad_outputs=torch.ones_like(au_x),
            create_graph=True
        )[0][:, 0]

        u_y = torch.autograd.grad(
            solution, collocation_points, grad_outputs=torch.ones_like(solution),
            create_graph=True
        )[0][:, 1]
        au_y = self.gaussian_kernel(collocation_points[:, 0], collocation_points[:, 1], self.alpha, 1) * u_y
        au_yy = torch.autograd.grad(
            au_y, collocation_points, grad_outputs=torch.ones_like(au_y),
            create_graph=True
        )[0][:, 1]

        # Compute the PDE residual
        lhs = -au_xx - au_yy - self.gaussian_kernel(collocation_points[:, 0], collocation_points[:, 1], self.kappa, 1) * solution
        rhs = rhs_func(collocation_points[:, 0], collocation_points[:, 1]).detach()

        # Boundary points
        x = torch.linspace(-3, 3, num_boundary, device=self.device).view(-1, 1)
        y = torch.linspace(-3, 3, num_boundary, device=self.device).view(-1, 1)

        left = torch.cat((-3 * torch.ones_like(y), y), dim=1)
        right = torch.cat((3 * torch.ones_like(y), y), dim=1)
        top = torch.cat((x, 3 * torch.ones_like(x)), dim=1)
        bottom = torch.cat((x, -3 * torch.ones_like(x)), dim=1)
        boundary_points = torch.cat((left, right, top, bottom), dim=0)

        # Loss combining physics residual and boundary conditions
        boundary_loss = loss_func(self(boundary_points), torch.zeros_like(boundary_points[:, :1]))
        physics_loss = loss_func(lhs, rhs)
        return physics_loss + boundary_loss

    

class PINN_Trainer:
    """
    Trainer for Physics-Informed Neural Networks (PINNs).
    """
    def __init__(self, model, optimizer, scheduler, device, freq=1):
        """
        Args:
            model (PINN): PINN model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            device (str): Device to run the training ('cpu' or 'cuda').
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_func = nn.MSELoss()
        self.freq = freq

    def train(self, data, rhs_func, num_epochs):
        """
        Train the PINN model.
        Args:
            data (torch.Tensor): Synthetic training data.
            rhs_func (callable): Function for the right-hand side of the PDE.
            num_epochs (int): Number of training epochs.
        """
        for epoch in range(num_epochs):
            loss = self._train_epoch(data, rhs_func)
            if epoch % self.freq == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, alpha = {self.model.alpha}, kappa = {self.model.kappa}")

    def _train_epoch(self, data, rhs_func):
        """
        Train the model for one epoch.
        Args:
            data (torch.Tensor): Synthetic training data.
            rhs_func (callable): Function for the right-hand side of the PDE.
        Returns:
            float: Loss value for the epoch.
        """
        

        def closure():
            # Compute synthetic and physics losses
            loss_data = self.model.synth_loss(data, self.loss_func)
            loss_phys = self.model.phys_loss(rhs_func, loss_func=self.loss_func)

            # Combine losses
            total_loss = loss_phys + 100 * loss_data
            self.optimizer.zero_grad()
            total_loss.backward()
            return total_loss

        if isinstance(self.optimizer, torch.optim.LBFGS):
            total_loss = self.optimizer.step(closure)
        else:
            # Optimize
            # Compute synthetic and physics losses
            loss_data = self.model.synth_loss(data, self.loss_func)
            loss_phys = self.model.phys_loss(rhs_func, loss_func=self.loss_func)

            # Combine losses
            total_loss = loss_phys + 100 * loss_data
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return total_loss.item()
