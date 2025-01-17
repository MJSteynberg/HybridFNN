
from models.generate_data import generate_data
from models.physical import PoissonEquation
from models.synthetic import NeuralNetwork2D
from models.training import Trainer
import torch
import numpy as np

def rhs(x, y):
    # a gaussian
    return torch.exp(-((x-1)**2 + (y-1)**2)/0.5) - 0.5*torch.exp(-((x+1)**2 + (y+1)**2)/0.5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Generate data
if input("Generate new data? (y/n): ") == 'y':
    generate_data(rhs)

# Load data
data = torch.from_numpy(np.loadtxt('data/diffusion/data.txt')).float()

# Define the synthetic model
data_dim = 1
hidden_dim = 128
num_layers = 6
learning_rate = 1e-3
num_epochs = 1000
model_synth = NeuralNetwork2D(data_dim, hidden_dim, num_layers, device=device)
optimizer_synth = torch.optim.Adam(model_synth.parameters(), lr=learning_rate)
scheduler_synth = torch.optim.lr_scheduler.OneCycleLR(optimizer_synth, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)

# Define the physical model
L = 6.0
N = 47
num_gaussians = 2
alpha = torch.tensor([3, 2.1, 1.1, -2.1, -1, -2, 1.0, 1.0], device=device)
model_phys = PoissonEquation(device, L, N, num_gaussians, rhs, alpha=alpha)
optimizer_phys = torch.optim.Adam(model_phys.parameters(), lr=1e-1)
scheduler_phys = torch.optim.lr_scheduler.LambdaLR(optimizer_phys, lr_lambda=lambda epoch: 1.0e-1)

# Create trainer
trainer = Trainer(model_synth, model_phys, optimizer_synth, optimizer_phys, scheduler_synth, scheduler_phys, device, N, interaction=True)

# Train the model
trainer.train(data, rhs, num_epochs)
print("Training finished.")

# print the final value of alpha
print("Final value of alpha:")
print(model_phys.alpha)
