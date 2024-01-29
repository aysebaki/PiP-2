import matplotlib.pyplot as plt
from dataset import get_dataset
from a4_ex1 import SimpleNetwork
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        show_progress: bool = False
) -> tuple[list, list]:
    # Define a function called 'training_loop' that takes network, train_data, eval_data, num_epochs, and show_progress as inputs
    # The function returns a tuple containing two lists

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Determine the device to be used for training (CUDA if available, otherwise CPU)

    network = network.to(device)
    # Move the network model to the chosen device

    optimizer = optim.Adam(network.parameters(), lr=0.002)
    # Create an Adam optimizer to update the parameters of the network during training

    loss_function = torch.nn.MSELoss(reduction="mean")
    # Create a mean squared error (MSE) loss function

    train_losses = []
    eval_losses = []
    # Create empty lists to store the training and evaluation losses

    batch_size = 32
    # Define the batch size for the data loader

    training_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0)
    eval_loader = DataLoader(eval_data, shuffle=False, batch_size=batch_size, num_workers=0)
    # Create data loaders for the training and evaluation data, with shuffling and batch processing

    best_eval_loss = float("inf")
    # Initialize the best evaluation loss to positive infinity

    epochs_without_improvement = 0
    # Initialize a counter for the number of epochs without improvement

    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        # Iterate over the specified number of epochs, using tqdm to show progress if enabled

        network.train()
        # Set the network to training mode

        train_loss = 0
        # Initialize the training loss to zero

        for data_sub, tar_sub in training_loader:
            # Iterate over the training data loader, getting batches of input data and target values

            data_sub, tar_sub = data_sub.float().to(device), tar_sub.float().to(device)
            # Move the data and target values to the chosen device

            optimizer.zero_grad()
            # Zero the gradients of the network parameters

            output = network(data_sub).squeeze()
            # Forward pass: compute the output of the network

            main_loss = loss_function(output, tar_sub)
            # Calculate the loss between the output and target values

            main_loss.backward()
            # Backpropagate the gradients through the network

            optimizer.step()
            # Update the network parameters using the optimizer

            train_loss += main_loss.item()
            # Accumulate the training loss

        train_loss /= len(training_loader)
        # Calculate the average training loss for the epoch

        train_losses.append(train_loss)
        # Append the training loss to the list of training losses

        network.eval()
        # Set the network to evaluation mode

        eval_loss = 0
        # Initialize the evaluation loss to zero

        with torch.no_grad():
            # Disable gradient computation for evaluation

            for data_sub, tar_sub in eval_loader:
                # Iterate over the evaluation data loader, getting batches of input data and target values

                data_sub, tar_sub = data_sub.float().to(device), tar_sub.float().to(device)
                # Move the data and target values to the chosen device

                output = network(data_sub).squeeze()
                # Forward pass: compute the output of the network

                main_loss = loss_function(output, tar_sub)
                # Calculate the loss between the output and target values

                eval_loss += main_loss.item()
                # Accumulate the evaluation loss

            eval_loss /= len(eval_loader)
            # Calculate the average evaluation loss for the epoch

            eval_losses.append(eval_loss)
            # Append the evaluation loss to the list of evaluation losses

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_without_improvement = 0
                # Update the best evaluation loss and reset the counter if a new best is found
            else:
                epochs_without_improvement += 1
                # Increment the counter if no improvement is found

            if epochs_without_improvement >= 3:
                break
                # Stop training if there is no improvement for the last 3 epochs

    return train_losses, eval_losses
    # Return the lists of training and evaluation losses


def plot_losses(train_losses: list, eval_losses: list):
    # Define a function called 'plot_losses' that takes train_losses and eval_losses as inputs

    epochs = range(1, len(train_losses) + 1)
    # Create a range of epoch numbers

    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, eval_losses, label='Evaluation Loss')
    # Plot the training and evaluation losses against the epoch numbers

    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.show()
    # Set labels, legend, and display the plot


if __name__ == "__main__":
    # If the script is being run as the main module

    torch.random.manual_seed(0)
    # Set the random seed for reproducibility

    train_data, eval_data = get_dataset()
    # Get the training and evaluation datasets

    network = SimpleNetwork(32, 128, 1)
    # Create an instance of a simple neural network

    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100)
    # Perform the training loop and obtain the training and evaluation losses

    plot_losses(train_losses, eval_losses)
    # Plot the training and evaluation losses

#This code defines a training loop for a neural network, performs the training loop, and plots the training and evaluation losses.
#The network is trained using the Adam optimizer and the mean squared error loss function.
#The training loop stops early if therse is no improvement in the evaluation loss for the last 3 epochs.