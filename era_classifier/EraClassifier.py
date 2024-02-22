import torch.nn as nn
from torch.nn import Linear,Sigmoid, BatchNorm1d, LeakyReLU, PReLU

class EraClassifier(nn.Module):
    def __init__(self, hidden_dim, activation_name):
        """
        
        Function that initialises the era classifier

        Args:
            hidden_dim (List): A list of integer corresponding to the number of nodes at each hidden layer
            activation_name (str): The name for the activation function to use at each hidden layer
        """
        super().__init__()
        # List to store the layer definition of the model
        layers = []
        hidden_dim.insert(0, 8)
        
        # Iterating through each hidden dim and creating the corresponding hidden layer
        for i in range(len(hidden_dim) - 1):
            current_features = hidden_dim[i]
            next_features = hidden_dim[i + 1]

            # Determines what type of non-linear activation function to use for hidden layer
            if activation_name == "PReLU":
                activation = PReLU()
            elif activation_name == "LeakyReLU":
                activation = LeakyReLU()
            
            # Adding the hidden layer to the list of layers
            hidden_layer = [Linear(current_features,next_features), BatchNorm1d(next_features), activation]
            layers += hidden_layer
            
        # Adding the final binary classification layer to the list
        layers += [Linear(hidden_dim[-1], 1), Sigmoid()]
        self.model = nn.Sequential(*layers)
    
        
    def forward(self, X):
        """
        
        Performs forward propagation on the model using the input data

        Args:
            X (Tensor): A 2D tensor of the dataset

        Returns:
            Tensor: The predicted label for each example in the dataset 
        """
        y = self.model(X)
        return y
