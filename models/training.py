import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

def interpolate_phys_solution(coords, F, L=6):
    """
    Interpolate values at given coordinates using a grid and plot the results.

    Parameters:
        coords (torch.Tensor): Coordinates for interpolation (num_points x 2).
        F (torch.Tensor): Grid values of the function (Nx x Ny).
        L (float): Length of the interval for the grid (default: 6).
    """

    # Normalize coordinates to [-1, 1] range for grid_sample
    coords_normalized = (coords[:,:2] + L/2) * 2 / L - 1
    coords_normalized = coords_normalized.unsqueeze(0)  # Add batch dimension

    # Perform interpolation using torch.nn.functional.grid_sample
    F_grid = F.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    coords_grid = coords_normalized.flip(-1).unsqueeze(0)  # Flip to (y, x) order for grid_sample

    interpolated_values = torch.nn.functional.grid_sample(
        F_grid,  # Input grid
        coords_grid,  # Normalized coordinates
        mode='bilinear', align_corners=True
    ).squeeze()

    return interpolated_values.reshape(-1,1)






    




class Trainer:
    def __init__(self,
                 model_synth,
                 model_phys,
                 optimizer_synth,
                 optimizer_phys,
                 scheduler_synth,
                 scheduler_phys,
                 device,
                 N,
                 print_freq=50, 
                 interaction = True):
        self.model_synth = model_synth.to(device)
        self.model_phys = model_phys.to(device)
        self.optimizer_synth = optimizer_synth
        self.optimizer_phys = optimizer_phys
        self.scheduler_synth = scheduler_synth
        self.scheduler_phys = scheduler_phys
        self.device = device
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.N = N
        self.interaction = interaction  

    def train(self, data, rhs, num_epochs):
        if not self.interaction:
            for epoch in range(num_epochs):
                loss = self._train_epoch_nointeraction(data, rhs)
                if epoch % self.print_freq == 0:
                    print(f"Epoch {epoch}: Loss = {loss:.4f}")
        else:
            for epoch in range(num_epochs):
                loss_FD, loss_synth, loss_hybrid = self._train_epoch(data, rhs)
                if epoch % self.print_freq == 0:
                    print(f"Epoch {epoch}: Loss FD = {loss_FD:.4f}, Loss Synth = {loss_synth:.4f}, Loss Hybrid = {loss_hybrid:.4f}")
        
    

            
    
    def _train_epoch_nointeraction(self, data, rhs):
        # Compute phys solution using the current value of alpha
        phys_solution = self.model_phys(rhs).reshape(self.N, self.N).to(self.device)
        data = data.to(self.device)
        interpolated_phys = interpolate_phys_solution(data, phys_solution)
        

        # Calculate the loss
        loss_FD = self.loss_func(interpolated_phys, data[:, 2].reshape(-1,1))
        loss = 100 * loss_FD + 1e6 * self.model_phys.penalization()
        # Optimizer steps
        self.optimizer_phys.zero_grad()
        loss.backward()
        self.optimizer_phys.step()
        self.scheduler_phys.step()


        return loss_FD.item()
    
    def _train_epoch(self, data, rhs):

        # Compute phys solution using the current value of alpha
        phys_solution = self.model_phys(rhs).reshape(self.N, self.N).to(self.device)
        data = data.to(self.device)
        # Choose collocation points in [-3,3]x[-3,3]
        collocation_points = 6 * torch.rand((10, 2)).to(self.device) - 3


        # Compute the synthetic solution on data points and collocation points
        synth_data = self.model_synth(data[:,:2])
        synth_col = self.model_synth(collocation_points)

        # Interpolate the physical solution at the collocation points and data points
        phys_data = interpolate_phys_solution(data[:,:2], phys_solution)
        phys_col = interpolate_phys_solution(collocation_points, phys_solution)

        loss_FD = self.loss_func(phys_data, data[:, 2].reshape(-1,1))
        loss_synth = self.loss_func(synth_data, data[:, 2].reshape(-1,1))
        loss_hybrid = self.loss_func(phys_col, synth_col)

        loss = 100 * loss_FD + 100 * loss_synth + loss_hybrid + 1e6 * self.model_phys.penalization()
        
        # Optimizer steps
        self.optimizer_phys.zero_grad()
        self.optimizer_synth.zero_grad()
        loss.backward()
        self.optimizer_phys.step()
        self.optimizer_synth.step()
        self.scheduler_phys.step()
        self.scheduler_synth.step()
        return loss_FD.item(), loss_synth.item(), loss_hybrid.item()
        
        
        
    
    # def _train_epoch(self, u_train, u_target, u0):
    #     # Compute phys solution using the current value of alpha
    #     phys_solution = self.model_phys(u0)

    #     # Compute trajectory predicted by NODE
    #     traj = self.model_node.f(u_train)

    #     # # Find node prediction on random points to compare with fd
    #     forward_random_points = 0.7*(6 * torch.rand((10, 2)).to(self.device) - 3)
    #     u0_init = interpolate_phys_solution(forward_random_points.unsqueeze(0), u0.unsqueeze(0))
    #     init = torch.cat((forward_random_points, u0_init.T), dim=1)
    #     grid_traj_forward = self.model_node.f(init)

    #     # Interpolate phys at trajectories
    #     interpolated_phys_traj = interpolate_phys_solution(grid_traj_forward, phys_solution)
        
    #     interpolated_phys_target = interpolate_phys_solution(u_target, phys_solution)
        

    #     # Calculate the loss
    #     if interpolated_phys_target.all() < 1e2:
    #         loss_FD = self.loss_func(interpolated_phys_target, u_target[:, :, 2])
    #         loss_hybrid = self.loss_func(interpolated_phys_traj, grid_traj_forward[:,:,2])
    #     else:
    #         loss_FD = self.model_phys.stabilize()
    #         loss_hybrid = 0
    #     loss = (100 * self.loss_func(traj, u_target)
    #              + self.model_phys.penalization()
    #             + 100 * loss_FD
    #             + loss_hybrid)
       

    #     # Optimizer steps
    #     self.optimizer_node.zero_grad()
    #     self.optimizer_phys.zero_grad()
    #     loss.backward()
    #     self.optimizer_node.step()
    #     self.optimizer_phys.step()
    #     self.scheduler_anode.step()
    #     self.scheduler_phys.step()

    #     return loss.item(), loss_FD.item()
