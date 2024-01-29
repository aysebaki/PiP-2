import torch

###Defines a class called SimpleCNN that inherits from torch.nn.Module.
###Represents a simple convolutional neural network model.
class SimpleCNN(torch.nn.Module):
    ###Defines the constructor method of the SimpleCNN class.
    ###Initializes the instance variables (of all things listed with self.) with the provided values.
    ###The kernel_size parameter defaults to 3, and the activation_function parameter defaults to torch.nn.ReLU() if not explicitly specified.
    def __init__(self, input_channels, hidden_channels, num_hidden_layers, use_batch_normalization,
                 num_classes, kernel_size=3, activation_function=torch.nn.ReLU()):
        ###Calls torch.nn.Module(the constructor of the parent class) to properly initialize the SimpleCNN object.
        super(SimpleCNN, self).__init__()
        ###Assign the provided values to the instance variables of the SimpleCNN object for later use.
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation_function = activation_function

        ###Create the convolutional layers of the neural network using torch.nn.Conv2d.
        ###The number of layers is determined by num_hidden_layers.
        ###The input channels, output channels, and kernel size are specified.
        ###If use_batch_normalization is set to True, batch normalization layers are added after each convolutional layer.
        ###The activation function is applied after each layer.
        ###The layers are stored in a torch.nn.ModuleList called conv_layers.
        self.conv_layers = torch.nn.ModuleList()
        in_channels = input_channels
        out_channels = hidden_channels
        for _ in range(num_hidden_layers):
            conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            self.conv_layers.append(conv_layer)
            if use_batch_normalization:
                batch_norm = torch.nn.BatchNorm2d(out_channels)
                self.conv_layers.append(batch_norm)
            self.conv_layers.append(activation_function)
            in_channels = out_channels

        ###Creates the output layer of the neural network using torch.nn.Linear.
        ###The input size is calculated by multiplying hidden_channels, the height (64), and the width (64) of the feature maps.
        ###The output size is specified as num_classes.
        self.output_layer = torch.nn.Linear(hidden_channels * 64 * 64, num_classes)
    ###Defines the forward method of the SimpleCNN class.
    ###It specifies the forward pass of the neural network.
    def forward(self, input_images):
        ###Define the sequence of operations in the forward pass.
        ###The input input_images is passed through each layer in self.conv_layers.
        ###The resulting feature maps are then flattened using x.view(x.size(0), -1).
        ###Finally, the flattened tensor is passed through the output layer to obtain the output of the network.
        x = input_images
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        output = self.output_layer(x)
        return output
###This line checks if the script is being run directly (as opposed to being imported as a module).
if __name__ == "__main__":
    ###Sets the random seed for Torch, ensuring reproducibility of random operations.
    torch.random.manual_seed(0)
    ###Creates an instance of the SimpleCNN class with 3 input channels, 32 hidden channels, 3 hidden layers, batch normalization enabled, 10 output classes, and ELU activation function.
    ###The instance is assigned to the variable network.
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=torch.nn.ELU())
    ###Creates a random input tensor with dimensions (1, 3, 64, 64) using the torch.randn function.
    ###It simulates a single input image with 3 channels and a size of 64x64 pixels.
    input = torch.randn(1, 3, 64, 64)
    ###Passes the input through the network object, invoking the forward method.
    ###The resulting output is assigned to the variable output.
    output = network(input)
    ###Prints the output tensor to the console.
    ###It represents the predicted values or logits for each class in the classification task.
    print(output)
