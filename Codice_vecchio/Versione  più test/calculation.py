import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import CNN

import time
from concurrent.futures import ThreadPoolExecutor


# Preso da https://github.com/pytorch/pytorch/blob/21c04b4438a766cd998fddb42247d4eb2e010f9a/benchmarks/functional_autograd_benchmark/utils.py#L19-L71

# Inserisci qui le funzioni di utilità: _del_nested_attr, _set_nested_attr, extract_weights, load_weights
# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
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


#####################################################################################################################


class function_f_and_Sigma():
    def __init__(self, model, All_models, dataset, eps_sqrt = 1e-5, Verbose = True, batch_size = 64):
        self.eps_sqrt = eps_sqrt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self.model = model.to(self.device)
        self.x_train = dataset.x_train[:1024]
        self.y_train = dataset.y_train[:1024]

        self.train_dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.criterion1 = nn.CrossEntropyLoss(reduction='none')

        self.already_computed_expected_loss_grad, self.already_computed_expected_loss_hessian, self.already_computed_sigma, self.already_computed_gradient_sigma_grad, self.already_computed_var_z_squared = False, False, False, False, False
        self.Verbose = Verbose

        self.number_of_parameters = sum(p.numel() for p in self.model.parameters())
        self.All_models = All_models.to(self.device)
        self.All_models.update_all_cnns_with_params(model.state_dict())        


    def update_parameters(self, new_parameters):
        self.already_computed_expected_loss_grad, self.already_computed_expected_loss_hessian, self.already_computed_sigma, self.already_computed_gradient_sigma_grad, self.already_computed_var_z_squared = False, False, False, False, False

        new_parameters = new_parameters.flatten()
        index = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            new_values = new_parameters[index:index + num_elements]            
            new_values = new_values.view(param.size())            
            param.data.copy_(new_values)            
            index += num_elements
            
        self.All_models.update_all_cnns_with_params(self.model.state_dict())  

    def compute_gradients_f(self):
        if self.already_computed_expected_loss_grad:
            if self.Verbose:
                print('    Already computed grad f')
            return self.expected_loss_grad
        
        self.loss_grad = torch.zeros(self.x_train.shape[0], self.number_of_parameters, device=self.device)
        start = time.time()
        
        if False: # 2
            temp = []
            not_done = 0
            for x, y in self.train_loader:
                if x.shape[0] != self.All_models.number_of_networks:
                    not_done = x.shape[0]
                    break
                x, y = x.to(self.device), y.to(self.device)
                output = self.All_models(x)
                loss = self.criterion(output, y)
                grad = torch.autograd.grad(loss, self.All_models.parameters(), create_graph=True)
                temp += grad
            temp = [g.flatten() for g in temp]
            grad_tensor = torch.cat(temp)
            self.loss_grad = grad_tensor.view(self.x_train.shape[0] - not_done, -1)
        elif False: # 3
            def compute_gradient(i, x, y):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion1(output, y)
                
                for l in loss:
                    grad = torch.autograd.grad(l, self.model.parameters(), create_graph=True)
                    grad = torch.cat([g.flatten() for g in grad])
                    self.loss_grad[i] = grad

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(compute_gradient, i, x, y) for i, (x, y) in enumerate(self.train_loader)]
                for future in futures:
                    future.result()
        elif False: # 3
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion1(output, y)
                for l in loss:
                    grad = torch.autograd.grad(l, self.model.parameters(), create_graph=True)
                    grad = torch.cat([g.flatten() for g in grad])
                    self.loss_grad[i] = grad
                break
        elif True: # 4
            start1 = time.time()
            perm = torch.randperm(self.x_train.shape[0])[: self.All_models.number_of_networks]
            x_batch = self.x_train[perm]
            y_batch = self.y_train[perm]
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            output = self.All_models(x_batch)
            loss = self.criterion(output, y_batch)
            grad = torch.autograd.grad(loss, self.All_models.parameters(), create_graph=True)
            grad = torch.cat([g.flatten() for g in grad])
            self.loss_grad = grad.view(self.All_models.number_of_networks, -1)
            assert grad[1] == self.loss_grad[0, 1], 'Error in computing gradients'
            print('Elapsed time grad f:', time.time() - start1)

            start1 = time.time()
            sum_grad = torch.sum(self.loss_grad, dim=0)
            self.All_hessian = torch.zeros(self.All_models.number_of_networks, self.number_of_parameters, self.number_of_parameters, device=self.device)
            for i, s in enumerate(sum_grad):
                grad_of_partial_derivative = torch.autograd.grad(s, self.All_models.parameters(), create_graph=True)
                grad_of_partial_derivative_vec = torch.cat([g.flatten() for g in grad_of_partial_derivative])
                grad_of_partial_derivative = grad_of_partial_derivative_vec.view(self.All_models.number_of_networks, -1)
                assert grad_of_partial_derivative[0, 1] == grad_of_partial_derivative_vec[1], 'Error in computing hessian'
                self.All_hessian[:, :, i] = grad_of_partial_derivative
            print('Elapsed time hessian:', time.time() - start1)

        print(f'    Elapsed time grad f: {time.time() - start:.2f} s')

        # Ricalcolo expected cosi dipendono tutti dagli stessi parametri
        start = time.time()  
        loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss += self.criterion(output, y)
        loss = loss / (self.x_train.shape[0])
        self.expected_loss_grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        self.expected_loss_grad = torch.cat([g.flatten() for g in self.expected_loss_grad])
        # self.expected_loss_grad = self.loss_grad.mean(dim=0)
        
        if self.Verbose:
            print(f'    Elapsed time grad f: {time.time() - start:.2f} s')

        assert self.expected_loss_grad.shape[0] == self.number_of_parameters, 'Expected loss gradient shape is not the same as the number of parameters'
        self.already_computed_expected_loss_grad = True
        return self.expected_loss_grad
    
    def compute_hessian_f_times_grad_f(self):
        if self.already_computed_expected_loss_hessian:
            if self.Verbose:
                print('    Already computed hessian f')
            return self.expected_loss_hessian
        start = time.time()

        self.hessian_f_times_grad_f = torch.zeros(self.number_of_parameters, device=self.device)
        expected_grad_without_grad = self.expected_loss_grad.detach()
        product = torch.sum(expected_grad_without_grad * self.expected_loss_grad)
        self.hessian_f_times_grad_f = torch.autograd.grad(product, self.model.parameters(), create_graph=True)
        self.hessian_f_times_grad_f = torch.cat([g.flatten() for g in self.hessian_f_times_grad_f])

        if self.Verbose:
            print(f'    Elapsed time hessian f: {time.time() - start:.2f} s')
        return self.hessian_f_times_grad_f

    
    def apply_sigma(self):
        if self.already_computed_sigma:
            if self.Verbose:    
                print('    Already computed sigma')
            return self.Sigma_sqrt, self.diag_Sigma
       
        start = time.time()
        self.Sigma_sqrt = torch.zeros(self.number_of_parameters, self.number_of_parameters, device=self.device)
        X = self.loss_grad - self.expected_loss_grad
        
        U, S, V = torch.svd(X, compute_uv=True)
        self.Sigma_sqrt  = (1 / torch.sqrt(torch.tensor(X.shape[0] - 1, dtype=torch.float32)) ) * V @ torch.diag(torch.sqrt(S + self.eps_sqrt)) @ V.T

        self.diag_Sigma = (1/torch.tensor(X.shape[0] - 1, dtype=torch.float32)) * torch.sum(X**2, dim=0)
        
        self.already_computed_sigma = True
        if self.Verbose:
            print(f'    Elapsed time Sigma: {time.time() - start:.2f} s')
        return self.Sigma_sqrt, self.diag_Sigma
    
    def compute_gradients_sigma_diag_times_grad_f(self):
        if self.already_computed_gradient_sigma_grad:
            if self.Verbose:
                print('    Already computed grad sigma')
            return self.grad_sigma_diag
        start = time.time()
        
        if not self.already_computed_sigma:
            self.apply_sigma()

        self.grad_sigma_diag = torch.zeros(self.number_of_parameters, self.number_of_parameters, device=self.device)
    
        start = time.time()
        m = self.loss_grad.shape[0]
        if False: # 2   # 50 sec (con 512)
            second_term = 2 * (m/(m-1)) * torch.diag(self.expected_loss_grad) @ self.hessian_f_times_grad_f
            for i, grad in enumerate(self.expected_loss_grad):
                # print('iteration', i)   

                aux = self.loss_grad[:, i].detach()
                product = torch.sum(aux * self.loss_grad[:, i])
                first_term = torch.autograd.grad(product, self.All_models.parameters(), create_graph=True)
                first_term = torch.cat([g.flatten() for g in first_term])
                first_term = first_term.view(self.All_models.number_of_networks, -1)
                first_term = torch.sum(first_term, dim = 0)
                self.grad_sigma_diag[i] = (1/(m-1)) * first_term
                # print(time.time() - start)
            self.grad_sigma_diag = self.grad_sigma_diag @ self.expected_loss_grad - second_term
        elif False: # 3      #41 sec (con 512)
            second_term = 2 * (m/(m-1)) * torch.diag(self.expected_loss_grad) @ self.hessian_f_times_grad_f
            for i, grad in enumerate(self.expected_loss_grad):
                print('iteration', i, end='')

                aux = self.loss_grad[:, i].detach()
                product = torch.sum(aux * self.loss_grad[:, i])
                first_term = torch.autograd.grad(product, self.model.parameters(), create_graph=True)
                first_term = torch.cat([g.flatten() for g in first_term])
                self.grad_sigma_diag[i] = (1/(m-1)) * first_term
            self.grad_sigma_diag = self.grad_sigma_diag @ self.expected_loss_grad - second_term
        elif True: # 4 
            second_term = 2 * (m/(m-1)) * torch.diag(self.expected_loss_grad) @ self.hessian_f_times_grad_f
            first_term = self.loss_grad.unsqueeze(2) * self.All_hessian
            first_term = 2 * (1/(m-1)) * torch.sum(first_term, dim=0)
            self.grad_sigma_diag = self.grad_sigma_diag @ self.expected_loss_grad - second_term
            
        # self.grad_sigma_diag = self.grad_sigma_diag @ self.expected_loss_grad
        self.already_computed_gradient_sigma_grad = True
        if self.Verbose:
            print(f'    Elapsed time grad sigma: {time.time() - start:.2f} s')
        return self.grad_sigma_diag
    
    def compute_var_z_squared(self):
        if self.already_computed_var_z_squared:
            if self.Verbose:
                print('    Already computed var_z_squared')
            return self.square_root_var_z_squared
        start = time.time()
        
        if not self.already_computed_sigma:
            self.apply_sigma()

        X = self.diag_Sigma - (self.loss_grad - self.expected_loss_grad) ** 2
        U, S, V = torch.svd(X, compute_uv=True)
        self.square_root_var_z_squared  = (1 / torch.sqrt(torch.tensor(X.shape[0] - 1, dtype=torch.float32)) ) * V @ torch.diag(torch.sqrt(S + self.eps_sqrt)) @ V.T

        assert self.square_root_var_z_squared.shape == (self.number_of_parameters, self.number_of_parameters), 'Variance of z squared shape is not the same as the number of parameters'

        self.already_computed_var_z_squared = True
        if self.Verbose:
            print(f'    Elapsed time var_z_squared: {time.time() - start:.2f} s')
        return self.square_root_var_z_squared

        