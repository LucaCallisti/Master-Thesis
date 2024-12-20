from CNN import CNN, function_f_and_Sigma, count_model_parameters, Train_n_times
from Dataset import CIFAR10Dataset

import torch
import numpy as np
import time

def Test_dataset():
    dataset = CIFAR10Dataset()
    print(dataset.get_image_size())
    dataset.show_image(0)
    dataset.downscale(75)
    print(dataset.get_image_size())
    dataset.show_image(0)
    dataset.to_grayscale()
    print(dataset.get_image_size())
    dataset.show_image(0)
    dataset.downscale(75)
    print(dataset.get_image_size())
    dataset.show_image(0)


def Test_CNN_and_dataset():
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
    dataset = CIFAR10Dataset()
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

    print('CIFAR10 grayscale example')
    dataset = CIFAR10Dataset()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

    print('CIFAR10 grayscale example')
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)

    print('rescaled CIFAR10 grayscale example')
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    output = model(dataset.x_train[0])
    print(output.shape)
    output = model(dataset.x_train[0:10])
    print(output.shape)


def Test_for_f_and_Sigma():
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    conv_layers = [
        (2, 3, 1, 1),  # filter number, kernel size, stride, padding
        (2, 3, 1, 1),
        (2, 3, 1, 1)
    ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    total_params = count_model_parameters(model)
    print(f"Total number of parameters: {total_params}")

    f = function_f_and_Sigma(model, dataset.x_train[:20], dataset.y_train[:20])
    start = time.time()
    aux = f.apply_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, calculated value: {aux:.2f}')

    start = time.time()
    grad = f.compute_gradients_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, gradient size: {grad.shape[0]}')
    

    model1 = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    f.update_model_params(model1.state_dict())
    start = time.time()
    aux = f.apply_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, calculated value: {aux:.2f}')

    start = time.time()
    grad1 = f.compute_gradients_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, old grad != new grad: {(grad != grad1).any()}')
    
    start = time.time()
    sigma = f.apply_sigma()
    print(f'Elapsed time: {time.time() - start:.2f}s, sigma shape: {sigma.shape[0]}')
 
    start = time.time()
    grad_sigma_diag = f.compute_gradients_sigma_diag()
    print(f'Elapsed time: {time.time() - start:.2f}s, grad sigma diag shape: {grad_sigma_diag.shape}')


def Test_Train_n_times():
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    conv_layers = [
        (2, 3, 1, 1),  
        (2, 3, 1, 1),
        (2, 3, 1, 1)
    ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)

    Train = Train_n_times(model, dataset, steps=100, lr=0.01, optimizer_name='ADAM')
    FinalDict = Train.train_n_times(1)
    breakpoint()