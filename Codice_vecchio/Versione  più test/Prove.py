import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers, size_img, pool_size=2, batch_norm=False):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

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
            x = conv(x)
            x = F.avg_pool2d(x, self.pool_size)
            x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_number_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)       
    

def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


def flatten_and_concat(tensor_list):
    # Appiattisci ogni tensore nella lista e concatenali lungo la dimensione 1
    flattened_tensors = [t.view(t.size(0), -1) for t in tensor_list]
    concatenated_tensor = torch.cat(flattened_tensors, dim=1)
    return concatenated_tensor


class functional_CNN():
    def __init__(self, model, loss_fn=nn.CrossEntropyLoss(reduction='none')):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.number_parameters = model.get_number_parameters()
        self.initial_params, self.initial_names = extract_weights(model)
        print('lunghezza dizionario param', len(self.initial_params))
        for i in range(len(self.initial_params)):
            print(i, self.initial_params[i].shape)
        self.loss_fn = loss_fn


    def set_input(self, x, y):
        self.x_batch = x.to(self.device)
        self.y_batch = y.to(self.device)
    
    def function(self, *params : Tuple[Tensor, ...]) -> Tensor:
        params = list(params)
        for i in range(len(params)):
            params[i] = params[i].reshape(self.initial_params[i].shape)
        params = tuple(params)
        load_weights(self.model, self.initial_names, params)
        output = self.model(self.x_batch)
        loss = self.loss_fn(output, self.y_batch) 
        return loss
    
    def jaconian(self, *params : Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        result = torch.autograd.functional.jacobian(self.function, params)
        result = flatten_and_concat(result)
        print('shape di result', result.shape, result.is_cuda)
        return result
    
    def hessian(self, *params : Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        result = torch.autograd.functional.hessian(self.function, params)
        breakpoint()
        return result
    
    def my_funct(self, *params : Tuple[Tensor, ...]) -> Tensor:
        grad = self.jaconian(*params)
        grad_detach = grad.detach()
        result = grad_detach * grad
        result = torch.sum(result, dim=0)
        breakpoint()
        return result[:-1]
    
    def jacobian_my_funct(self, *params : Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        result = torch.autograd.functional.jacobian(self.my_funct, params)
        breakpoint()
        result = flatten_and_concat(result)         # dovrebbe mettere il contributo di grad_detach lungo le righe
        return result

    
    def get_initial_params(self):
        return self.initial_params
    
    def get_initial_names(self):
        return self.initial_names
    

def test():
    model = CNN(1, 10, [(16, 3, 1, 1), (32, 3, 1, 1)], 28)
    x_batch = torch.randn(5, 1, 28, 28)
    y_batch = torch.tensor([0, 1, 2, 3, 4])
    f = functional_CNN(model)
    f.set_input(x_batch, y_batch)
    params = f.get_initial_params()
    Res = f.function(*params)
    Grad = f.jaconian(*params)
    
    x = x_batch[0, :, :, :].unsqueeze(0)
    y = y_batch[0].unsqueeze(0)
    f.set_input(x, y)
    r = f.function(*params)
    g = f.jaconian(*params)

    print('shape gradienti', g.shape, Grad.shape)
    print('Same result?', torch.allclose(r, Res[0]), torch.allclose(g, Grad[0, :]))

    model1 = CNN(1, 10, [(16, 3, 1, 1), (32, 3, 1, 1)], 28)
    model1 = model1.to('cuda')
    params1, _ = extract_weights(model1)
    f.set_input(x_batch, y_batch)
    Res1 = f.function(*params1)
    Grad1 = f.jaconian(*params1)
    print('Device', Res1.is_cuda, Grad1.is_cuda, Res.is_cuda, Grad.is_cuda)
    print('Different result?', torch.allclose(Res, Res1), torch.allclose(Grad, Grad1))



    x_batch = torch.randn(256, 1, 28, 28)
    y_batch = torch.randint(0, 10, (256,))
    f.set_input(x_batch, y_batch)
    start = time.time()
    r = f.my_funct(*params)
    Result = f.jacobian_my_funct(*params)
    print('shape Result', Result.shape)
    print('Time elapsed', time.time() - start)
    

    Hessian = f.hessian(*params)
    print('shape Hessian', Hessian.shape)


if __name__ == "__main__":
    test()