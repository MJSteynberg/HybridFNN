
from models.generate_data import generate_data
from models.physical import HelmholtzEquation
from models.synthetic import NeuralNetwork2D
from models.training import Trainer
import torch
import numpy as np
from datetime import datetime
import pandas as pd

def rhs(x, y):
    # a gaussian source
    return (4 * torch.exp(-((x - 2) ** 2 + (y - 2 ) ** 2)) + 4 * torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Generate data
if input("Generate new data? (y/n): ") == 'y':
    generate_data(rhs, 100)

# Load data
data = torch.from_numpy(np.loadtxt('data/diffusion/data.txt')).float()

# Define the synthetic model
data_dim = 1
hidden_dim = 500
num_layers = 4
learning_rate = 1e-3
num_epochs = 3000

# Define the physical model
L = 6.0
N = 21
num_gaussians = 1
alpha_orig = [6, -1,1,  1.0]
k_orig = [4, 1, -1, 1]
alpha = torch.tensor(alpha_orig, device=device).float()
k = torch.tensor(k_orig, device=device).float()


# Create trainer
model_synth = NeuralNetwork2D(data_dim, hidden_dim, num_layers, device=device)
optimizer_synth = torch.optim.Adam(model_synth.parameters(), lr=learning_rate)
scheduler_synth = torch.optim.lr_scheduler.OneCycleLR(optimizer_synth, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
model_phys = HelmholtzEquation(device, L, N, num_gaussians, rhs, alpha=alpha,k=k)
optimizer_phys = torch.optim.Adam(model_phys.parameters(), lr=1e-1)
scheduler_phys = torch.optim.lr_scheduler.LambdaLR(optimizer_phys, lr_lambda=lambda epoch: 1.0e-1)
trainer = Trainer(model_synth, model_phys, optimizer_synth, optimizer_phys, scheduler_synth, scheduler_phys, device, N, interaction=True)

# Train the model
trainer.train(data, rhs, num_epochs)
print("Training finished.")

# print the final value of alpha
alpha_hybrid = model_phys.alpha
k_hybrid = model_phys.k

print(f"Hybrid alpha: {alpha_hybrid}")
print(f"Hybrid k: {k_hybrid}")


alpha = torch.tensor(alpha_orig, device=device).float()
k = torch.tensor(k_orig, device=device).float()
# Create trainer
model_synth = NeuralNetwork2D(data_dim, hidden_dim, num_layers, device=device)
optimizer_synth = torch.optim.Adam(model_synth.parameters(), lr=learning_rate)
scheduler_synth = torch.optim.lr_scheduler.OneCycleLR(optimizer_synth, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
model_phys = HelmholtzEquation(device, L, N, num_gaussians, rhs, alpha=alpha, k=k)
optimizer_phys = torch.optim.Adam(model_phys.parameters(), lr=1e-1)
scheduler_phys = torch.optim.lr_scheduler.LambdaLR(optimizer_phys, lr_lambda=lambda epoch: 1.0e-1)
trainer = Trainer(model_synth, model_phys, optimizer_synth, optimizer_phys, scheduler_synth, scheduler_phys, device, N, interaction=False)

# Train the model
trainer.train(data, rhs, num_epochs)
print("Training finished.")

# print the final value of alpha
alpha_phys = model_phys.alpha
k_phys = model_phys.k

print(f"Physics alpha: {alpha_phys}")
print(f"Physics k: {k_phys}")

# Save all parameters to parameters folder using date and time in name
alpha_real = np.array([4, 1, 1, 1.0])
k_real = np.array([3, -1, -1, 1])

alpha = pd.DataFrame(np.stack([alpha_real.flatten(), alpha_hybrid.flatten().detach().cpu(), alpha_phys.flatten().detach().cpu()]).T, columns=["alpha_real", "alpha_hybrid", "alpha_phys"])
k = pd.DataFrame(np.stack([k_real.flatten(), k_hybrid.flatten().detach().cpu(), k_phys.flatten().detach().cpu()]).T, columns=["k_real", "k_hybrid", "k_phys"])

# Get the current date and time to include in the filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Save the DataFrame to CSV with the timestamp in the filename
alpha.to_csv(f'parameters/diffusion/alpha/param_{timestamp}.csv', index=False)
k.to_csv(f'parameters/diffusion/k/param_{timestamp}.csv', index=False)