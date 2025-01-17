
from models.generate_data import generate_data
from models.physical import PoissonEquation
from models.synthetic import NeuralNetwork2D
from models.training import Trainer
import torch
import numpy as np
from datetime import datetime
import pandas as pd

def rhs(x, y):
    # a sine wave
    return 10*torch.sin(2 * np.pi * x / 6) * torch.sin(2 * np.pi * y / 6)

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
num_epochs = 2000

# Define the physical model
L = 6.0
N = 50
num_gaussians = 2
orig = [1, 1, -2, 1.4, -1.1, -1.3, 1.0, 1.0]
alpha = torch.tensor(orig, device=device)


# Create trainer
model_synth = NeuralNetwork2D(data_dim, hidden_dim, num_layers, device=device)
optimizer_synth = torch.optim.Adam(model_synth.parameters(), lr=learning_rate)
scheduler_synth = torch.optim.lr_scheduler.OneCycleLR(optimizer_synth, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
model_phys = PoissonEquation(device, L, N, num_gaussians, rhs, alpha=alpha)
optimizer_phys = torch.optim.Adam(model_phys.parameters(), lr=1e-1)
scheduler_phys = torch.optim.lr_scheduler.LambdaLR(optimizer_phys, lr_lambda=lambda epoch: 1.0e-1)
trainer = Trainer(model_synth, model_phys, optimizer_synth, optimizer_phys, scheduler_synth, scheduler_phys, device, N, interaction=True)

# Train the model
trainer.train(data, rhs, num_epochs)
print("Training finished.")

# print the final value of alpha
params_hybrid = model_phys.alpha


alpha = torch.tensor(orig, device=device)
# Create trainer
model_synth = NeuralNetwork2D(data_dim, hidden_dim, num_layers, device=device)
optimizer_synth = torch.optim.Adam(model_synth.parameters(), lr=learning_rate)
scheduler_synth = torch.optim.lr_scheduler.OneCycleLR(optimizer_synth, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
model_phys = PoissonEquation(device, L, N, num_gaussians, rhs, alpha=alpha)
optimizer_phys = torch.optim.Adam(model_phys.parameters(), lr=1e-1)
scheduler_phys = torch.optim.lr_scheduler.LambdaLR(optimizer_phys, lr_lambda=lambda epoch: 1.0e-1)
trainer = Trainer(model_synth, model_phys, optimizer_synth, optimizer_phys, scheduler_synth, scheduler_phys, device, N, interaction=False)

# Train the model
trainer.train(data, rhs, num_epochs)
print("Training finished.")

# print the final value of alpha
params_phys = model_phys.alpha

# Save all parameters to parameters folder using date and time in name
params_real = np.array([3, 2, -2, 1, -2, 1, 1.0, 1.0])
params = pd.DataFrame(np.stack([params_real.flatten(), params_hybrid.flatten().detach().cpu(), params_phys.flatten().detach().cpu()]).T, columns=["params_real", "params_hybrid", "params_phys"])

# Get the current date and time to include in the filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Save the DataFrame to CSV with the timestamp in the filename
params.to_csv(f'parameters/diffusion/param_{timestamp}.csv', index=False)