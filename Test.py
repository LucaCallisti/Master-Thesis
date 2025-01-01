from CNN import CNN, Train_n_times
from SDE import RMSprop_SDE
from Dataset import CIFAR10Dataset

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torchsde

import calculation
from calculation import function_f_and_Sigma

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

    f = function_f_and_Sigma(model, dataset.x_train[:20], dataset.y_train[:20])
    start = time.time()
    aux = f.apply_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, calculated value: {aux:.2f}')

    start = time.time()
    grad = f.compute_gradients_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, gradient size: {grad.shape}')
    

    model1 = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    f.update_model_params(model1.state_dict())
    start = time.time()
    aux = f.apply_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, calculated value: {aux:.2f}')

    start = time.time()
    grad1 = f.compute_gradients_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, old grad != new grad: {(grad != grad1).any()}')

    start = time.time()
    hess = f.compute_hessian_f()
    print(f'Elapsed time: {time.time() - start:.2f}s, hessian shape: {hess.shape}')
    
    start = time.time()
    sigma_sqrt, _ = f.apply_sigma()
    print(f'Elapsed time: {time.time() - start:.2f}s, sigma shape: {sigma_sqrt.shape[0]}')
 
    start = time.time()
    grad_sigma_diag = f.compute_gradients_sigma_diag()
    print(f'Elapsed time: {time.time() - start:.2f}s, grad sigma diag shape: {grad_sigma_diag.shape}')

    start = time.time()
    var_z = f.compute_var_z_squared()
    print(f'Elapsed time: {time.time() - start:.2f}s, var z shape: {var_z.shape}')




def Test_Train_n_times():
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    conv_layers = [
        (4, 3, 1, 1),  
        (8, 3, 1, 1),
        (16, 3, 1, 1),
        (32, 3, 1, 1)
    ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    print('model parameters:', model.get_number_parameters())
    print('dimension dataset:', dataset.x_train.shape[0])
    steps, batch_size = 10000, 32
    print('number of epoch:', steps * batch_size / dataset.x_train.shape[0])
    Train = Train_n_times(model, dataset, steps=steps, lr=0.01, optimizer_name='ADAM')
    FinalDict = Train.train_n_times(n = 1, batch_size=batch_size)

    run_1 = FinalDict[1]
    Loss = run_1['Loss']
    Accuracy = run_1['Accuracy']

    aux = dataset.x_train.shape[0] // batch_size
    avg_loss = [np.mean(Loss[i:i+50]) for i in range(0, len(Loss), aux)]
    avg_accuracy = [np.mean(Accuracy[i:i+50]) for i in range(0, len(Accuracy), aux)]

    plt.plot(avg_loss)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.figure()
    plt.plot(avg_accuracy)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()
    breakpoint()

Test_Train_n_times()


def Test_SDE():
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    conv_layers = [
        (1, 3, 1, 1),  # filter number, kernel size, stride, padding
        (1, 3, 1, 1),
        (1, 3, 1, 1),
        (1, 3, 1, 1)
    ]
    conv_layers = [
        (4, 3, 1, 1),  
        (8, 3, 1, 1),
        (16, 3, 1, 1),
        (32, 3, 1, 1)
    ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)


    params, names  = calculation.extract_weights(model) 
    input_params = torch.cat([v.reshape(-1) for v in params]) 
    
    # To convert input_params back to params
    split_sizes = [v.numel() for v in params]
    params_reconstructed = torch.split(input_params, split_sizes)
    params_reconstructed = tuple([p.reshape(v.shape) for p, v in zip(params_reconstructed, params)])

    x0 =  torch.concat(((input_params).clone(), 1e-8*torch.ones_like(input_params), torch.zeros_like(input_params)), dim = 0)
    
    f = function_f_and_Sigma(model, params, names, dataset.x_train[:100], dataset.y_train[:100])

    eta = 0.001
    beta = 0.99
    eps = 1e-8
    sde = RMSprop_SDE(eta, beta, eps, f)

    # Intervallo di tempo
    t0 = 0.0
    t1 = 0.1
    N = 101
    t = torch.linspace(t0, t1, N)

    result = torchsde.sdeint(sde, x0.unsqueeze(0), t, method = 'euler', dt =1e-3)
    all_grad = sde.get_gradient()
    torch.save(result, 'result_1.pt')
    torch.save(all_grad, 'all_grad_1.pt')

Test_SDE()


