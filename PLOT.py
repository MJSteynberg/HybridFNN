# plot gaussians with given parameters

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os 
from datetime import datetime


def gaussian(param, num_gaussians=1, N=100, L=6):
    # interpolate alpha1 to the grid
    gaussian_map = 0.1*np.ones((N, N), dtype=np.float32)
    
    x = np.linspace(-L//2, L//2, N)
    y = np.linspace(-L//2, L//2, N)
    x, y = np.meshgrid(x, y, indexing='ij')
    for i in range(num_gaussians):
        gaussian_map += param[i] * np.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2*num_gaussians + i]) ** 2))
    return gaussian_map

def load_files(folder):

    # List all files in the directory
    files = os.listdir(folder)

    # Filter files that match the naming pattern 'param_YYYY-MM-DD_HH-MM-SS.csv'
    param_files = [f for f in files if f.startswith('param_') and f.endswith('.csv')]

    # Sort files based on the timestamp in the filename
    param_files.sort(key=lambda x: datetime.strptime(x[6:25], '%Y-%m-%d_%H-%M-%S'), reverse=True)

    print(f"Found {len(param_files)} parameter files.")

    # Load the most recent param and index files
    recent_param_file = param_files[0]

    param_df = pd.read_csv(os.path.join(folder, recent_param_file))

    return param_df
def plot_gaussians():
    torch.set_rng_state(torch.manual_seed(42).get_state())
    device = 'cpu'

    results_folder = f'parameters/diffusion/'
    data_folder = 'data/diffusion/'

    alpha = load_files(results_folder + 'alpha/')
    k = load_files(results_folder + 'k/')
    data = np.loadtxt(data_folder + 'data.txt')
    coords = data[:,:2]
    
    


    alpha_real = gaussian(alpha[f"alpha_real"].values, num_gaussians = 1)
    alpha_hybrid = gaussian(alpha[f"alpha_hybrid"].values, num_gaussians = 1)
    alpha_phys = gaussian(alpha[f"alpha_phys"].values, num_gaussians = 1)

    k_real = gaussian(k[f"k_real"].values, num_gaussians = 1)
    k_hybrid = gaussian(k[f"k_hybrid"].values, num_gaussians = 1)
    k_phys = gaussian(k[f"k_phys"].values, num_gaussians = 1)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    colormap = 'viridis'

    L = 6
    global_max_a = np.max(alpha_real)
    global_max_k = np.max(k_real)


    # Setup all heatmaps
    im1 = axs[0][0].imshow(alpha_phys,
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_a )
    im2 = axs[0][1].imshow(alpha_hybrid,
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_a )
    im3 = axs[0][2].imshow(alpha_real,
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_a )
    
    im4 = axs[1][0].imshow(k_phys,
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_k )
    im5 = axs[1][1].imshow(k_hybrid,
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_k )
    im6 = axs[1][2].imshow(k_real,
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_k )

    

    # Calculate L1 errors
    alpha1_mean_error_physics = abs(alpha_phys - alpha_real).mean()
    alpha1_mean_error_hybrid = abs(alpha_hybrid - alpha_real).mean()

    k_mean_error_physics = abs(k_phys - k_real).mean()
    k_mean_error_hybrid = abs(k_hybrid - k_real).mean()



    # Set titles including errors
    plt.rcParams['axes.titlesize'] = 14
    axs[0][0].set_title(r"$\kappa(x)   $ Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % alpha1_mean_error_physics)
    axs[0][1].set_title(r"$\kappa(x)   $ Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % alpha1_mean_error_hybrid)
    axs[0][2].set_title(r"$\kappa(x)  $ Real")

    axs[1][0].set_title(r"$\eta(x)   $ Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % k_mean_error_physics)
    axs[1][1].set_title(r"$\eta(x)   $ Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % k_mean_error_hybrid)
    axs[1][2].set_title(r"$\eta(x)  $ Real")


    # Set axis limits
    for ax in axs:
        for a in ax:
            a.set_xlim(-3, 3)
            a.set_ylim(-3, 3)

    axs[0][0].scatter(coords[:, 0], coords[:, 1], c='k', s=10)
    axs[0][1].scatter(coords[:, 0], coords[:, 1], c='k', s=10)
    axs[1][0].scatter(coords[:, 0], coords[:, 1], c='k', s=10)
    axs[1][1].scatter(coords[:, 0], coords[:, 1], c='k', s=10)

    fig.colorbar(im3, ax=axs, orientation='vertical', shrink=0.8, label="Legend")
    plt.savefig(f'{results_folder}helmholtz.png', dpi=500)

def plot_error():

    torch.set_rng_state(torch.manual_seed(42).get_state())
    device = 'cpu'

    results_folder = f'parameters/diffusion/'
    data_folder = 'data/diffusion/'

    alpha = load_files(results_folder + 'alpha/')
    k = load_files(results_folder + 'k/')
    data = np.loadtxt(data_folder + 'data.txt')
    coords = data[:,:2]


    


    alpha_real = gaussian(alpha[f"alpha_real"].values, num_gaussians = 1)
    alpha_hybrid = gaussian(alpha[f"alpha_hybrid"].values, num_gaussians = 1)
    alpha_phys = gaussian(alpha[f"alpha_phys"].values, num_gaussians = 1)

    k_real = gaussian(k[f"k_real"].values, num_gaussians = 1)
    k_hybrid = gaussian(k[f"k_hybrid"].values, num_gaussians = 1)
    k_phys = gaussian(k[f"k_phys"].values, num_gaussians = 1)

    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colormap = 'viridis'
    L = 6
    global_max_a = max(np.max(alpha_phys-alpha_real), np.max(alpha_hybrid-alpha_real))
    global_max_k = max(np.max(k_phys-k_real), np.max(k_hybrid-k_real))
    # Setup heatmaps
    im1 = axs[0][0].imshow(np.abs(alpha_phys - alpha_real),
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_a)
    im2 = axs[0][1].imshow(np.abs(alpha_hybrid - alpha_real),
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_a)
    
    im3 = axs[1][0].imshow(np.abs(k_phys - k_real),
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_k)
    im4 = axs[1][1].imshow(np.abs(k_hybrid - k_real),
                        extent=(-L/2, L/2, -L/2, L/2),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max_k)
    


    # Calculate L1 errors
    alpha_mean_error_physics = abs(alpha_phys - alpha_real).mean()
    alpha_mean_error_hybrid = abs(alpha_hybrid - alpha_real).mean()

    k_mean_error_physics = abs(k_phys - k_real).mean()
    k_mean_error_hybrid = abs(k_hybrid - k_real).mean()


    # Set titles including errors
    plt.rcParams['axes.titlesize'] = 14
    axs[0][0].set_title(r"$\kappa(x)   $ Error Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_physics)
    axs[0][1].set_title(r"$\kappa(x)   $ Error Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_hybrid)
    axs[1][0].set_title(r"$\eta(x)   $ Error Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % k_mean_error_physics)
    axs[1][1].set_title(r"$\eta(x)   $ Error Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % k_mean_error_hybrid)
    


    # Set axis limits
    for ax in axs:
        for a in ax:
            a.set_xlim(-3, 3)
            a.set_ylim(-3, 3)

    # Plot training data
    plt.scatter(coords[:, 0], coords[:, 1], c='k', s=10)

    fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.8, label="Legend")
    plt.savefig(f'{results_folder}helmholtz_error.png', dpi=500)





def compare_predictions():
    """
    Compares the predictions of alpha1 (physics, hybrid, real) and centers (physics, hybrid, real) on the same plot.
    """
    # Reshape alpha1 and center values into 3x2 and 3x4 matrices
    
    
    results_folder_alpha = f'parameters/diffusion/alpha/'
    results_folder_k = f'parameters/diffusion/k/'

    alpha = load_files(results_folder_alpha)
    k = load_files(results_folder_k)

    alpha1_real = np.array([alpha[f'alpha_real'][0], k[f'k_real'][0]])
    alpha1_hybrid = np.array([alpha[f'alpha_hybrid'][0], k[f'k_hybrid'][0]])
    alpha1_phys = np.array([alpha[f'alpha_phys'][0], k[f'k_phys'][0]])
    
    alpha1_values = np.array([alpha1_real, alpha1_hybrid, alpha1_phys])

    centers_real = np.array(np.concatenate((alpha[f'alpha_real'][1:3], k[f'k_real'][1:3])))
    centers_hybrid = np.array(np.concatenate((alpha[f'alpha_hybrid'][1:3], k[f'k_hybrid'][1:3])))
    centers_phys = np.array(np.concatenate((alpha[f'alpha_phys'][1:3], k[f'k_phys'][1:3])))
    centers_values = np.array([centers_real.ravel(), centers_hybrid.ravel(), centers_phys.ravel()])
    
    

    # X-axis labels for 6 parameters
    parameter_names = [
        r'$\kappa1_1$', r'$\kappa1_2$', r'$c_{1,x}$', r'$c_{1,y}$', r'$c_{2,x}$', r'$c_{2,y}$'
    ]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the x-axis positions for each parameter
    x_positions = np.arange(6)

    # Enable the grid
    # Set zorder of grid to be below the points
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=14)  # Increase font size for both x and y axis ticks

    # Plot the alpha1 and centers values
    for i in range(3):
        # Set the marker and size for each scatter plot
        marker = 'o'  # Default is a circle
        if i == 0:  # For 'Real' we use a star
            marker = '*'
            size = 400  # Increased size for Real (stars)
        elif i == 2:
            marker = 'X'
            size = 150  # Default size for Hybrid and Physics
        else: 
            size = 150  # Default size for Hybrid and Physics

        # Plot alpha1 values (Red for Real, Green for Hybrid, Blue for Physics)
        ax.scatter(x_positions[:2], alpha1_values[i, :], c='r' if i == 0 else 'b' if i == 1 else 'g',
                   label=r'$\kappa1_{1,2}$ (' + ["Real", "Hybrid", "Physics"][i] + ')', s=size, edgecolors='k', marker=marker)

        # Plot center values (Green for Real, Blue for Hybrid, Red for Physics)
        ax.scatter(x_positions[2:], centers_values[i, :], c='r' if i == 0 else 'b' if i == 1 else 'g',
                   label=f'Centers ({["Real", "Hybrid", "Physics"][i]})', s=size, edgecolors='k', marker=marker)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(parameter_names)

    # Set y-axis label
    ax.set_ylabel('Predicted Values', fontsize=14)

    # Add title
    ax.set_title(r'Comparison of Predicted $\kappa1$ and Centers (Real vs Hybrid vs Physics)', fontsize=14)

    # Add a legend
    ax.legend()

    # Save the plot in the usual folder
    plt.savefig(f'parameters/diffusion/compare_predictions.png', dpi=500)


plot_gaussians()
plot_error()
compare_predictions()


