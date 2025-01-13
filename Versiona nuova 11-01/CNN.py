import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
import copy





class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers, size_img, pool_size=2):
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
            x = torch.nn.functional.gelu(x)
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
    if len(tensor_list[0].shape) == 1:
        return torch.cat(tensor_list, dim=0)
    else:
        flattened_tensors = [t.view(t.size(0), -1) for t in tensor_list]
        concatenated_tensor = torch.cat(flattened_tensors, dim=1)
        return concatenated_tensor
 


class functional_CNN():
    def __init__(self, model, loss_fn=nn.CrossEntropyLoss(reduction='none')):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.number_parameters = model.get_number_parameters()
        self.initial_params, self.initial_names = extract_weights(model)
        self.loss_fn_red_none = loss_fn
        self.loss_fn_red_mean = nn.CrossEntropyLoss(reduction='mean')
        self.which_loss = 'none'


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
        if self.which_loss == 'none':
            loss = self.loss_fn_red_none(output, self.y_batch)
        elif self.which_loss == 'mean':
            loss = self.loss_fn_red_mean(output, self.y_batch)
        return loss
    
    def jaconian(self, *params : Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        result = torch.autograd.functional.jacobian(self.function, params)
        result = flatten_and_concat(result)
        return result
    
    def expected_loss_hessian(self, *params : Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        self.which_loss = 'mean'
        result = torch.autograd.functional.hessian(self.function, params)
        temp = [flatten_and_concat(r) for r in result]
        hessian = torch.cat(temp, dim = 0)
        return hessian
    

    def expected_loss_gradient(self, params : Tuple[Tensor, ...]) -> Tensor:
        self.which_loss = 'mean'
        result = torch.autograd.functional.jacobian(self.function, params)
        return flatten_and_concat(result)
    
    def All_data_gradient(self, params : Tuple[Tensor, ...]) -> Tensor:
        self.which_loss = 'none'
        result = torch.autograd.functional.jacobian(self.function, params)
        result = flatten_and_concat(result)
        return result
    
    def my_funct(self, *params : Tuple[Tensor, ...]) -> Tensor:
        self.which_loss = 'none'
        grad = self.jaconian(*params)       # grad_ij = (\partial_j f_i)_{i,j}
        grad_detach = grad.detach()         # grad_detach_ij = (\partial_j ~f_i)_{i,j}
        result = grad_detach * grad         # result_ij = (\partial_j ~f_i \partial_j f_i)_{i,j}
        result = (1/(grad.shape[0]-1)) * torch.sum(result, dim=0)  # result_j = ( \sum_i (\partial_j ~f_i \partial_j f_i) )_j
        return result
    
    def jacobian_my_funct(self, *params : Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        self.which_loss = 'none'
        result = torch.autograd.functional.jacobian(self.my_funct, params)      # restituisce una tupla lunga quanto il numero di layer del modello, in cui nella posizione i c'è un tensore lungo dim_outup_my_function x dim_layer_i
        result = flatten_and_concat(result)         # dovrebbe mettere il contributo di grad_detach lungo le righe
        return result

    
    def get_initial_params(self):
        return self.initial_params
    
    def get_initial_names(self):
        return self.initial_names
    

class Train_n_times():
    def __init__(self, model, dataset, steps=100, lr=0.01, optimizer_name='RMSPROP', beta = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.steps = steps
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.beta = beta
        self.initial_model_params = copy.deepcopy(model.state_dict())
        self.criterion = nn.CrossEntropyLoss()
    


    def train_n_times(self, n = 1, batch_size=1):
        '''
        Train the model n times with the same dataset and return the history of the optimization process
        The dictionary returned has the following structure:
        dict =  {number_of_run : {  'Params' : lr, betas, etc , 
                                    'Loss' : [loss_1, loss_2, ..., loss_n],
                                    'step' : { 'model' : model_parameters,
                                                'square_avg' : square_avg_parameters (if optimizer is RMSprop),
                                                'adam_param' : adam_param_parameters (if optimizer is Adam)
                                            }
                            }
                }
        '''

        def train(model, dataloader, optimizer, criterion, steps):
            model.train()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            History = {'Params' : optimizer.state_dict()['param_groups'].copy()}
            History['Loss'] = {}
            History['Params'] = []
            History['Square_avg'] = []
            Loss = []
            Accuracy = []
            aux_initial_parameters = model.state_dict().copy()

            for i, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                images.detach()
                labels.detach()

                Loss.append(loss.item())
                Accuracy.append((outputs.argmax(1) == labels).float().mean().item())

                if isinstance(optimizer, torch.optim.RMSprop):
                    square_avg = [v['square_avg'].clone() if isinstance(v['square_avg'], torch.Tensor) else torch.tensor(v['square_avg'])  for k, v in self.optimizer.state_dict()['state'].items()]
                    square_avg = torch.cat([s.view(-1) for s in square_avg])

                    params = model.parameters()
                    params = torch.cat([p.view(-1) for p in params])
                    # History[i+1] = {'model' : params, 'square_avg' : square_avg}
                    History['Square_avg'].append(square_avg)
                    History['Params'].append(params)
                if isinstance(optimizer, torch.optim.Adam):
                    adam_param = {k: {'exp_avg' : v['exp_avg'].clone(), 'exp_avg_sq' : v['exp_avg_sq'].clone()} for k, v in self.optimizer.state_dict()['state'].items()}
                    History[i+1] = {'model' : model.state_dict().copy(), 'adam_param' : adam_param}

                if i == steps:
                    break

            History['Loss'] = Loss
            History['Accuracy'] = Accuracy
            History['Params'] = torch.stack(History['Params'])
            History['Square_avg'] = torch.stack(History['Square_avg'])
            
            # if isinstance(optimizer, torch.optim.RMSprop):
            #     History[0] = {'model' : aux_initial_parameters, 'square_avg' : torch.zeros_like(History[1]['square_avg'])}

            if isinstance(optimizer, torch.optim.Adam):
                initial_adam_param = {k: {'exp_avg': v['exp_avg'].clone(), 'exp_avg_sq': v['exp_avg_sq'].clone()} for k, v in History[1]['adam_param'].items()}
                for key in initial_adam_param.keys():
                    initial_adam_param[key]['exp_avg'] = torch.zeros_like(initial_adam_param[key]['exp_avg'], device=device)
                    initial_adam_param[key]['exp_avg_sq'] = torch.zeros_like(initial_adam_param[key]['exp_avg_sq'], device=device)

                # History[0] = {'model' : aux_initial_parameters, 'adam_param' : initial_adam_param}

            return History
        
        Different_run = {}
        for i in range(n):
            self.model.load_state_dict(copy.deepcopy(self.initial_model_params))
            dataloader = self.dataset.dataloader(batch_size = batch_size, steps=self.steps)

            if self.optimizer_name == 'RMSPROP':
                if self.beta == None:
                    self.beta = 0.999
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=self.beta, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            elif self.optimizer_name == 'ADAM':
                if self.beta == None: 
                    self.beta = (0.9, 0.999)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.beta, eps=1e-08, weight_decay=0, amsgrad=False)
            else:
                raise ValueError('Invalid optimizer')
            History = train(self.model, dataloader, self.optimizer, self.criterion, steps=self.steps)
            Different_run[i+1] = History
        
        self.Different_run = Different_run
        return Different_run
    
    def save_dict(self, path):
        torch.save(self.Different_run, path)
            
        
    
    def get_model(self):
        return self.model
    



