import torch

###Defines the class SimpleNetwork that inherits from torch.nn.Module
###Serves as a blueprint for creating instances of a neural network model.
class SimpleNetwork(torch.nn.Module):
    ###This line defines the constructor method of the SimpleNetwork class.
    ###It initializes the instance variables (everything in __init__) with the provided values.
    ###The activation_function parameter defaults to torch.nn.ReLU() if not explicitly specified.
    def __init__(self, input_neurons, hidden_neurons, output_neurons, activation_function=torch.nn.ReLU()):
        ###Calls torch.nn.Module(the constructor of the parent class) to properly initialize the SimpleNetwork object.
        super(SimpleNetwork, self).__init__()
        ###Assign the provided values to the instance variables of the SimpleNetwork object for later use.
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.activation_function = activation_function

        ###Create the layers of the neural network using the torch.nn.Linear class.
        ###Each layer is instantiated with the specified input and output sizes.
        ###The weights and biases of these layers will be learned during training.
        self.input_layer = torch.nn.Linear(input_neurons, hidden_neurons)
        self.hidden_layer1 = torch.nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer2 = torch.nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer = torch.nn.Linear(hidden_neurons, output_neurons)
    ###Defines the forward method of the SimpleNetwork class.
    ###Specifies the forward pass of the neural network.
    def forward(self, x):
        ###These lines define the sequence of operations in the forward pass.
        ###The input x is passed through each layer, followed by an activation function.
        ###Finally, the output of the last layer is returned as the result of the forward pass.
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer1(x))
        x = self.activation_function(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x
###Checks if the script is being run directly (as opposed to being imported as a module).
if __name__ == "__main__":
    ###Sets the random seed for Torch, ensuring reproducibility of random operations.
    torch.random.manual_seed(0)
    ###Creates an instance of the SimpleNetwork class with 10 input neurons, 20 hidden neurons, and 5 output neurons.
    ###The instance is assigned to the variable simple_network.
    simple_network = SimpleNetwork(10, 20, 5)
    ###Creates a random input tensor with dimensions (1, 10) using the torch.randn function.
    ###It simulates a single input sample for the neural network.
    input = torch.randn(1, 10)
    ###Passes the input through the simple_network object, invoking the forward method.
    ###The resulting output is assigned to the variable output.
    output = simple_network(input)
    #prints the output tensor to the console.
    #The output tensor has dimensions (1,5) because the neural network has 5 output neurons.
    print(output)
