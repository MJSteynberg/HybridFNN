import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a neural network class that inherits from nn.Module
class NeuralNetwork2D(nn.Module):
    """
    A 2D Neural Network class that constructs a feedforward neural network with a specified number of layers and hidden dimensions.
    Attributes:
        network (nn.Sequential): A sequential container of the network layers.
    """
    def __init__(self, data_dim, hidden_dim, num_layers, activation=nn.ReLU, device='cpu'):
        """
        Initializes the NeuralNetwork2D class with the given parameters.
        Args:
            data_dim (int): The dimension of the input data.
            hidden_dim (int): The number of neurons in the hidden layers.
            num_layers (int): The number of layers in the network.
            activation (nn.Module): The activation function to use between layers. Default is nn.ReLU.
            device (str): The device to run the model on. Default is 'cpu'.
        """
        super(NeuralNetwork2D, self).__init__()
        
        self.device = device
        
        # Define input and output dimensions
        input_dim = 2
        output_dim = data_dim

        # Initialize a list to hold the layers of the network
        layers = []
        
        # Add the first layer with input_dim and hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())  # Add the activation function

        # Add hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # Add the activation function

        # Add the final layer with hidden_dim and output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Create a sequential container with the defined layers
        self.network = nn.Sequential(*layers).to(self.device)

        # Initialize the weights of the network
        self._initialize_weights()

    # Method to initialize weights of the network
    def _initialize_weights(self):
        """
        Initializes the weights of the network using Xavier normal distribution for the weights and zeros for the biases.
        """
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Initialize weights with Xavier normal distribution
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero

    # Define the forward pass of the network
    def forward(self, x):
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = x.to(self.device)
        return self.network(x)

