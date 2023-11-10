import torch
import torch.nn as nn
import torch.nn.functional as Fun



class VariableCNN1D(nn.Module):
    def __init__(self, input_size, num_classes, window_size, layer_dims, conv_kernel_size=5, max_pool_kernel_size=2):
        super(VariableCNN1D, self).__init__()
        

        self.bn = nn.BatchNorm1d(input_size)

        self.layers = nn.ModuleList()
        
        # Create convolutional layers based on layer_dims
        in_channels = input_size
        for out_channels in layer_dims:
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size),
                nn.ReLU(),
            )
            self.layers.append(conv_layer)
            in_channels = out_channels
        self.layers.append(nn.MaxPool1d(kernel_size=max_pool_kernel_size))
        
        self.fc = nn.Linear(self.calculate_flatten_size( input_size, window_size), num_classes)

    def calculate_flatten_size(self, input_size, window_size):
        # Create a temporary tensor to compute the size
        temp_input = torch.randn(1, input_size, window_size)  
        for layer in self.layers:
            temp_input = layer(temp_input)
        return temp_input.view(1, -1).size(1)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.bn(out)
        for layer in self.layers:
            out = layer(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = Fun.sigmoid(out)
        return out



