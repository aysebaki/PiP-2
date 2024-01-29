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
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = network.to(device)  # Move the network to the chosen device

    # Create an Adam optimizer to update the network parameters
    optimizer = optim.Adam(network.parameters(), lr=0.002)
    # Define the loss function (mean squared error)
    loss_function = torch.nn.MSELoss(reduction="mean")

    # Initialize empty lists to store training and evaluation losses
    train_losses = []
    eval_losses = []

    batch_size = 32

    # Create data loaders for the training and evaluation datasets
    training_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0)
    eval_loader = DataLoader(eval_data, shuffle=False, batch_size=batch_size, num_workers=0)

    # Start the training loop for the specified number of epochs
    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        network.train()  # Set the network in training mode
        train_loss = 0

        # Iterate over batches of data in the training loader
        for data_sub, tar_sub in training_loader:
            data_sub, tar_sub = data_sub.float().to(device), tar_sub.float().to(device)
            optimizer.zero_grad()  # Clear the gradients of the optimizer

            output = network(data_sub).squeeze()  # Forward pass through the network
            main_loss = loss_function(output, tar_sub)  # Compute the loss

            main_loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update the network parameters

            train_loss += main_loss.item()  # Accumulate the training loss

        train_loss /= len(training_loader)  # Calculate the average training loss per batch
        train_losses.append(train_loss)  # Store the training loss for the current epoch

        network.eval()  # Set the network in evaluation mode
        eval_loss = 0

        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Iterate over batches of data in the evaluation loader
            for data_sub, tar_sub in eval_loader:
                data_sub, tar_sub = data_sub.float().to(device), tar_sub.float().to(device)

                output = network(data_sub).squeeze()  # Forward pass through the network
                main_loss = loss_function(output, tar_sub)  # Compute the loss

                eval_loss += main_loss.item()  # Accumulate the evaluation loss

            eval_loss /= len(eval_loader)  # Calculate the average evaluation loss per batch
            eval_losses.append(eval_loss)  # Store the evaluation loss for the current epoch

    # Return the lists of training and evaluation losses
    return train_losses, eval_losses


if __name__ == "__main__":
    torch.random.manual_seed(0)  # Set the random seed for reproducibility
    train_data, eval_data = get_dataset()  # Obtain the training and evaluation datasets
    network = SimpleNetwork(32, 128, 1)  # Create an instance of the SimpleNetwork class
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=10)

    # Print the training and evaluation losses for each epoch
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")


#Overall, this code defines a training loop for a neural network using the PyTorch framework.
#It performs training over a specified number of epochs, updating the network parameters using the Adam optimizer
#and calculating the loss using the mean squared error (MSE) loss function.
#The training and evaluation losses are computed and stored for each epoch, and at the end, the losses are printed for analysis.