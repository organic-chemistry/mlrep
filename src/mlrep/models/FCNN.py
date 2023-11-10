import torch
import torch.nn as nn
import torch.nn.functional as Fun



class FCNN(nn.Module):
    def __init__(self, input_size, num_classes, window_size):
        super(FCNN, self).__init__()

        self.fc = nn.Linear(self.calculate_flatten_size( input_size, window_size), num_classes)

    def calculate_flatten_size(self, input_size, window_size):
        # Create a temporary tensor to compute the size
        temp_input = torch.randn(1, input_size, window_size)  
    
        return temp_input.view(1, -1).size(1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = Fun.sigmoid(out)
        return out



