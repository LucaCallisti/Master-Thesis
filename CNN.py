import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Dataset

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


def main():
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
    main()


