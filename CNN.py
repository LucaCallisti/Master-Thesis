import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Dataset
import time

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers, size_img, pool_size=2):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.pool_size = pool_size
        
        self.size_img = size_img
        in_channels = input_channels
        for out_channels, kernel_size, stride, padding in conv_layers:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            in_channels = out_channels
            size_img = np.floor((size_img - kernel_size + 2 * padding) / stride) + 1
            size_img = int(np.floor((size_img - self.pool_size) / self.pool_size) + 1)
        
        in_channels = in_channels * size_img * size_img
        self.fc = nn.Linear(int(in_channels), num_classes)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.avg_pool2d(x, self.pool_size)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def count_model_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    return total_params

class function_f_and_Sigma():
    def __init__(self, model, x_train, y_train):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(device)
        self.x_train = x_train.to(device)
        self.y_train = y_train.to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.loss = None
        self.expected_value_loss = None
        self.expected_loss_grad = None
        self.sigma = None

        self.already_computed_loss = False
        self.already_computed_expected_loss_grad = False
        self.already_computed_sigma = False

        self.number_of_parameters = count_model_parameters(self.model)
    
    def update_model_params(self, new_params):
        self.model.load_state_dict(new_params)

        self.already_computed_loss = False
        self.already_computed_expected_loss_grad = False
        self.already_computed_sigma = False

    def apply_f(self):
        if self.already_computed_loss:
            return self.f
        
        result = self.model(self.x_train)

        # Calculate Loss
        self.loss = self.criterion(result, self.y_train)
        
        # Mean over dataset
        self.expected_value_loss = self.loss.mean(axis=0) 
        self.already_computed_loss = True

        assert self.loss.shape[0] == self.x_train.shape[0], 'Loss shape is not the same as the number of samples'
        assert self.expected_value_loss.dim() == 0, 'Expected loss shape is not the same as the number of samples'

        return self.expected_value_loss
    
    def compute_gradients_f(self):
        if self.already_computed_expected_loss_grad:
            return self.f_grad
        
        self.model.zero_grad()

        if self.already_computed_loss:
            output = self.expected_value_loss
        else:
            output = self.apply_f()

        gradients = torch.autograd.grad(output.mean(), self.model.parameters(), create_graph=True)
        gradients = torch.cat([grad.reshape(-1) for grad in gradients])

        self.expected_loss_grad = gradients
        self.already_computed_expected_loss_grad = True

        assert self.expected_loss_grad.shape[0] == self.number_of_parameters, 'Expected loss gradient shape is not the same as the number of parameters'

        return self.expected_loss_grad
    
    def apply_sigma(self):
        if self.already_computed_sigma:
            return self.sigma

        Sigma = 0
        for l in self.loss:
            self.model.zero_grad()
            l_grad = torch.autograd.grad(l, self.model.parameters(), create_graph=True)
            l_grad = torch.cat([grad.reshape(-1) for grad in l_grad])

            Sigma += torch.ger(l_grad-self.expected_loss_grad, l_grad-self.expected_loss_grad)

        self.sigma = Sigma / self.loss.shape[0]
        self.already_computed_sigma = True

        assert self.sigma.shape == (self.number_of_parameters, self.number_of_parameters), 'Sigma shape is not the same as the number of parameters'

        return self.sigma
    
    def compute_gradients_sigma_diag(self):
        if not self.already_computed_sigma:
            self.apply_sigma()
        
        sigma_diag = torch.diag(self.sigma)
        grad_sigma_diag = []
        for d in sigma_diag:
            self.model.zero_grad()
            gradients = torch.autograd.grad(d, self.model.parameters(), create_graph=True)
            gradients = torch.cat([grad.reshape(-1) for grad in gradients])
            grad_sigma_diag.append(gradients)
        self.grad_sigma_diag = torch.stack(grad_sigma_diag)

        assert self.grad_sigma_diag.shape == (self.number_of_parameters, self.number_of_parameters), 'Gradient of Sigma diagonal shape is not the same as the number of parameters'

        return self.grad_sigma_diag


