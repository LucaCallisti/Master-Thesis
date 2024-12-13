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

class function_f():
    def __init__(self, model, x_train):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(device)
        self.dataset = x_train.to(device)

    def apply(self):
        result = self.model(self.dataset).mean(axis=0) 

        # Calculate Loss
        loss = result.mean()
        return loss
    
    def update_model_params(self, new_params):
        self.model.load_state_dict(new_params)
    
    def compute_gradients(self):
        self.model.zero_grad()
        output = self.apply()
        gradients = torch.autograd.grad(output.mean(), self.model.parameters(), create_graph=True)
        gradients = torch.cat([grad.view(-1) for grad in gradients])
        return gradients
    
def test():
    conv_layers = [
        (32, 3, 1, 1),  # filter number, kernel size, stride, padding
        (64, 3, 1, 1),
        (128, 3, 1, 1)
    ]
    
    print('Random example')
    model = CNN(input_channels=3, num_classes=10, conv_layers=conv_layers, size_img=64)
    print(model)

    input_tensor = torch.randn(1, 3, 64, 64)
    output = model(input_tensor)
    print(output.shape)

    print('CIFAR10 example')
    dataset = Dataset.CIFAR10Dataset()
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

    print('CIFAR10 grayscale example')
    dataset = Dataset.CIFAR10Dataset()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

    print('CIFAR10 grayscale example')
    dataset = Dataset.CIFAR10Dataset()
    dataset.to_grayscale()
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

    print('rescaled CIFAR10 grayscale example')
    dataset = Dataset.CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

if __name__ == '__main__':
    dataset = Dataset.CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    conv_layers = [
        (32, 3, 1, 1),  # filter number, kernel size, stride, padding
        (32, 3, 1, 1),
        (32, 3, 1, 1)
    ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    total_params = count_model_parameters(model)
    print(f"Total number of parameters: {total_params}")

    f = function_f(model, dataset.x_train)
    start = time.time()
    aux = f.apply()
    print(f'Elapsed time: {time.time() - start:.2f}s, calculated value: {aux:.2f}')

    start = time.time()
    grad = f.compute_gradients()
    print(f'Elapsed time: {time.time() - start:.2f}s, gradient size: {grad.shape[0]}')
    

    model1 = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    f.update_model_params(model1.state_dict())
    start = time.time()
    aux = f.apply()
    print(f'Elapsed time: {time.time() - start:.2f}s, calculated value: {aux}')

    start = time.time()
    grad1 = f.compute_gradients()
    print(f'Elapsed time: {time.time() - start:.2f}s, old grad != new grad: {(grad != grad1).any()}')



