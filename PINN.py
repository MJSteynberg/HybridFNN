from models.pinn import PINN, PINN_Trainer
from models.synthetic import NeuralNetwork2D
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.generate_data import generate_data

def rhs(x, y):
    # A Gaussian source term
    return 4 * torch.exp(-((x - 2) ** 2 + (y - 2) ** 2)) + 4 * torch.exp(-((x + 1) ** 2 + (y + 1) ** 2))

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic data
    # generate_data(rhs, 100)
    data = torch.from_numpy(np.loadtxt('data/diffusion/pinn/data.txt')).float().to(device)

    # Network and training parameters
    data_dim = 1
    hidden_dim = 500
    num_layers = 4
    alpha_orig = [4, 1, 1, 1.0]
    k_orig = [3, -1, -1, 1.0]
    alpha = torch.tensor(alpha_orig, requires_grad=True, device=device)
    kappa = torch.tensor(k_orig, requires_grad=True, device=device)

    # Initialize the model
    network = NeuralNetwork2D(data_dim, hidden_dim, num_layers, device=device, activation=torch.nn.Tanh)
    model = PINN(network, alpha, kappa, device=device)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=1000, cycle_momentum=False)

    # Trainer
    trainer = PINN_Trainer(model, optimizer, scheduler, device, freq = 100)

    # Train the model
    num_epochs = 2000
    trainer.train(data, rhs, num_epochs)

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e0, max_iter=100)
    trainer = PINN_Trainer(model, optimizer, scheduler, device, freq = 1)
    trainer.train(data, rhs, 100)

    # Evaluate and visualize the solution
    x = torch.linspace(-3, 3, 100)
    y = torch.linspace(-3, 3, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack((X.flatten(), Y.flatten()), dim=1).to(device)

    Z = model(xy).reshape(100, 100).detach().cpu().numpy()

    # Plot predicted solution
    plt.imshow(Z, extent=(-3, 3, -3, 3), origin='lower', cmap='viridis')
    plt.colorbar(label='Solution')
    plt.title('Predicted Solution')
    plt.savefig('solution_pinn_predicted.png')
    plt.show()

    # Compare with ground truth
    rhs_vals = rhs(X.flatten(), Y.flatten()).reshape(100, 100).detach().cpu().numpy()
    plt.imshow(rhs_vals, extent=(-3, 3, -3, 3), origin='lower', cmap='viridis')
    plt.colorbar(label='Ground Truth')
    plt.title('Ground Truth (RHS)')
    plt.savefig('solution_pinn_ground_truth.png')
    plt.show()
