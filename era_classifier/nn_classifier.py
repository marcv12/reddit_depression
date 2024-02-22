from EraClassifier import EraClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from CovidDataset import CovidDataset
import matplotlib.pyplot as plt
import tqdm
import time
import argparse
import os
import csv

def save_model( model, model_save_dir, model_save_name, best_validation_model_idx,best_validation_model_accuracy):  
    """
    
    Function saves a trained model into a specified directory along with some metadata

    Args:
        model (nn.Module): The trained neural network model to be saved
        model_save_dir (str): The directory to save the model
        model_save_name (str): The file name to save the model
        best_validation_model_idx (int): The epoch where the model achieved the highest validation accuracy
        best_validation_model_accuracy (float): The highest validation accuracy achieved by the model
    """
    state = {}
    state['network'] = model
    state['best_val_model_idx'] = best_validation_model_idx 
    state['best_val_model_accuracy'] = best_validation_model_accuracy
    torch.save(state, f=os.path.join(model_save_dir, model_save_name)) 
    
def save_statistics(experiment_log_dir, filename, stats_dict):
    """
    
    Save the model statistics into a specified directory

    Args:
        experiment_log_dir (str): The directory to save the model statistics
        filename (str): The file name to save the model statistics
        stats_dict (Dict): A dictionary containing the model statistics
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    with open(summary_filename, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list(stats_dict.keys()))

        
        total_rows = len(list(stats_dict.values())[0])
        for idx in range(total_rows):
            row_to_add = [value[idx] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

def list_of_ints(arg):
    """
    
    Utility function for parsing the number of hidden nodes at each hidden layer

    Args:
        arg (Namespace): An object containing the supplied hyperparameters

    Returns:
        List: A list of integers representing the # hidden nodes at each hidden layer
    """
    return list(map(int, arg.split(',')))

def get_args():
    """
    
    Function to parse the hyperparmaeters arguments from the command line
    
    Returns:
        Namespace: An object containing the supplied hyperparameters
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
    parser.add_argument('--lr', nargs="?", type=float,  default = 1e-3, help = "Learning rate for experiment")
    parser.add_argument('--epochs', nargs="?", type=int, default=50, help = "Number of epoch for experiment")
    parser.add_argument('--hidden_dim', nargs="?", type=list_of_ints, default=[4, 2], help = "Hidden layers for experiment")
    parser.add_argument('--activation', nargs="?", type=str, default="LeakyReLU", help = "Activation function for experiment")
    parser.add_argument('--model_id', nargs="?", type=int, default=1, help = "Model ID")
    args = parser.parse_args()

    return args

def plot_training(train_data, valid_data, metric):
    """
    
    Plotting the training and validation data of the trained model

    Args:
        train_data (List): A list of training data recorded at each epoch
        valid_data (List): A list of validation data recorded at each epoch
        metric (str): _description_
    """
    plt.plot(range(len(train_data)), train_data, label=f"Training {metric}")
    plt.plot(range(len(valid_data)), valid_data, label=f"Validation {metric}") 
    plt.xlabel("Epochs")
    plt.ylabel(metric) 
    plt.title(f"Training and Validation {metric} VS Epoch")
    plt.legend()
    plt.show()  

torch.manual_seed(20)

# Defining the hyperparameters for the classifier
args = get_args()
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs
hidden_dim = args.hidden_dim
activation_name = args.activation
model_id = args.model_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading the experimental dataset as a Dataset object
dataset = CovidDataset("experimental")
# Splitting experimental dataset into train, validation and test set
train_set, valid_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])
# Training mean and dtd used for normalising the data
train_mean = torch.mean(train_set.dataset.X[train_set.indices], dim=0)
train_std = torch.std(train_set.dataset.X[train_set.indices], dim=0)

train_loader = DataLoader(train_set, batch_size= batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size= batch_size, shuffle=True)

# Defining the model to be trained nad evaluated, the loss function and optimiser to use
model = EraClassifier(hidden_dim, activation_name)
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# Defining the variables to track while training and evaluating them model
total_losses = {"train_loss" : [], "train_accuracy" : [], "valid_loss" : [], "valid_accuracy" : []}
best_model_accuracy = 0
best_model_idx = None

# Running through each epoch , training the model and then validating it
for epoch_idx in range(num_epochs): 
    model.train()
    current_epoch_loss  = {"train_loss" : [], "train_accuracy" : [], "valid_loss" : [], "valid_accuracy" : []}
    epoch_start_time = time.time()
    
    with tqdm.tqdm(total = len(train_loader)) as pbar_train:
        # Training the model on each batch in the training data
        for (X,  y) in train_loader:  
            # Normalising the training data with training mean and std
            X = (X.to(device) - train_mean) / train_std
            y = y.to(device)
            
            # Performing forward propagation and computing training loss
            y_pred = model(X)
            loss_val = loss(y_pred, y)
            
            # Performing backprogation of gradients
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # Computing training accuracy 
            y_pred = y_pred.round()
            accuracy_val = y_pred.eq(y).cpu().data.numpy().mean()
            
            current_epoch_loss['train_loss'].append(loss_val.cpu().data.numpy())
            current_epoch_loss['train_accuracy'].append(accuracy_val)
            pbar_train.update(1)
            pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss_val, accuracy_val))

    
    model.eval()
    
    with tqdm.tqdm(total = len(valid_loader)) as pbar_valid:
        # Validating the model on each batch in the validation data
        for (X, y) in valid_loader:      
            # Normalising the validation data with training mean and std
            X = (X.to(device) - train_mean) / train_std
            y = y.to(device)
            
            # Performing forward propagation and computing validation loss
            y_pred = model(X)
            loss_val = loss(y_pred, y)
            
            # Computing validation accuracy
            y_pred = y_pred.round()
            accuracy_val = y_pred.eq(y).cpu().data.numpy().mean()
            
            current_epoch_loss['valid_loss'].append(loss_val.cpu().data.numpy())
            current_epoch_loss['valid_accuracy'].append(accuracy_val)
            pbar_valid.update(1)
            pbar_valid.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss_val, accuracy_val))
    
    # Computing the average training and validation loss and accuracy across all batch in current epoch
    for key, value in current_epoch_loss.items():
        total_losses[key].append(np.mean(value))
    
    # Checking if the average validation accuracy is better than the highest score so far
    val_mean_accuracy = np.mean(current_epoch_loss['valid_accuracy'])
    if val_mean_accuracy > best_model_accuracy:
        best_model_accuracy = val_mean_accuracy 
        best_model_idx = epoch_idx
        
    out_string = " ".join(
            ["{}={:.4f}".format(key, np.mean(value)) for key, value in current_epoch_loss.items()])
    epoch_elapsed_time = time.time() - epoch_start_time  
    epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
    print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

# Saving the model and their training and validation metrics
os.makedirs(f"results/pytorch/model_{model_id}")
save_model(model, f"results/pytorch/model_{model_id}",  "model_weight", best_model_idx,best_model_accuracy)
save_statistics(f"results/pytorch/model_{model_id}","model_summary.csv", total_losses)

# Plotting the training and validation accuracy for the model
print(f"Best Model Accuracy : {best_model_accuracy}")
plot_training(total_losses["train_accuracy"], total_losses["valid_accuracy"], "Accuracy")
plot_training(total_losses["train_loss"], total_losses["valid_loss"], "Loss")

# Testing the model on the test set
current_epoch_loss = {'test_loss' : [], 'test_accuracy' : []}

model.eval()

with tqdm.tqdm(total=len(test_loader)) as pbar_test:
    # Testing the model on each batch in the test data
    for (X, y) in test_loader:
        # Normalising the test data with training mean and std
        X = (X.to(device) - train_mean) / train_std
        y = y.to(device)
        
        # Performing forward propagation and computing test loss
        y_pred = model(X)
        loss_val = loss(y_pred, y)
        
         # Computing test accuracy
        y_pred = y_pred.round()
        accuracy_val = y_pred.eq(y).cpu().data.numpy().mean()
        
        current_epoch_loss["test_loss"].append(loss_val.cpu().data.numpy())
        current_epoch_loss["test_accuracy"].append(accuracy_val)
        pbar_test.update(1)
        pbar_test.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss_val, accuracy_val))
        

test_losses = {key : [np.mean(value)] for key, value in current_epoch_loss.items()}
# Saving the model's testing metrics
save_statistics(f"results/pytorch/model_{model_id}", f"model_test.csv", test_losses)
    