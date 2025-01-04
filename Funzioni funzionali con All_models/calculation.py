import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

import time


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

        self.model = model.to(self.device)
        self.x_train = dataset.x_train.to(self.device) 
        self.y_train = dataset.y_train.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.already_computed_expected_loss_grad, self.already_computed_expected_loss_hessian, self.already_computed_sigma, self.already_computed_gradient_sigma_grad, self.already_computed_var_z_squared = False, False, False, False, False
        self.Verbose = Verbose

        self.All_models = All_models.to(self.device)
        self.All_models.update_all_cnns_with_params(model.state_dict())  

        self.number_of_parameters = sum(p.numel() for p in self.model.parameters())
        self.initial_params, self.names  = extract_weights(model) 
        self.split_sizes = [v.numel() for v in self.initial_params]

        self.A_initial_params, self.A_names  = extract_weights(self.All_models)
        self.A_split_sizes = [v.numel() for v in self.A_initial_params]

    
    def Calculate_hessian_gradient(self, new_parameters):
        
        start = time.time()
        self.Hessian_all_data = torch.zeros(self.All_models.number_of_networks, self.number_of_parameters, self.number_of_parameters, device=self.device)
        self.Gradient_all_data = torch.zeros(self.All_models.number_of_networks, self.number_of_parameters, device=self.device)
        perm = torch.randperm(self.x_train.shape[0])[: self.All_models.number_of_networks]
        x_batch = self.x_train[perm]
        y_batch = self.y_train[perm]
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        A_new_parameters = new_parameters.repeat(self.All_models.number_of_networks)
        A_params_reconstructed = torch.split(A_new_parameters, self.A_split_sizes)
        
        def function(*params : Tuple[Tensor, ...]) -> Tensor:
            params = list(params)
            for i in range(len(params)):
                params[i] = params[i].reshape(self.A_initial_params[i].shape)
            params = tuple(params)
            load_weights(self.All_models, self.A_names, params)
            output = self.All_models(x_batch)
            loss = loss_fn(output, y_batch) 
            return loss

        start = time.time()
        J = torch.autograd.functional.jacobian(function, A_params_reconstructed, create_graph=True) 
        print(f'Elapsed time for Jacobian: {time.time() - start:.2f} s')
        # self.Gradient_all_data = torch.cat([v.flatten() for v in J])
        start = time.time()
        print(f'Current time: {time.strftime("%H:%M:%S")}')
        H = torch.autograd.functional.hessian(function, A_params_reconstructed)
        print(f'Elapsed time for Hessian: {time.time() - start:.2f} s')
        breakpoint() 
        hessian_list = [torch.cat(h, dim=1) for h in H]
        self.Hessian_all_data = torch.cat(hessian_list, dim=0)

        if self.Verbose:
            print(f'    Elapsed time for Hessian and gradient: {time.time() - start:.2f} s')

        self.already_computed_expected_loss_grad, self.already_computed_expected_loss_hessian, self.already_computed_sigma, self.already_computed_gradient_sigma_grad, self.already_computed_var_z_squared = False, False, False, False, False


    def compute_gradients_f(self):
        if self.already_computed_expected_loss_grad:
            if self.Verbose:
                print('    Already computed grad f')
            return self.expected_loss_grad
        
        start = time.time()
        self.expected_loss_grad = self.Gradient_all_data.mean(axis=0)
        if self.Verbose:
            print(f'    Elapsed time grad f: {time.time() - start:.2f} s')

        assert self.expected_loss_grad.shape[0] == self.number_of_parameters, 'Expected loss gradient shape is not the same as the number of parameters'
        self.already_computed_expected_loss_grad = True
        return self.expected_loss_grad
    
    def compute_hessian_f(self):
        if self.already_computed_expected_loss_hessian:
            if self.Verbose:
                print('    Already computed hessian f')
            return self.expected_loss_hessian

        start = time.time()
        self.expected_loss_hessian = self.Hessian_all_data.mean(axis=0)
        if self.Verbose:
            print(f'    Elapsed time hessian f: {time.time() - start:.2f} s')
        
        if not torch.allclose(self.expected_loss_hessian, self.expected_loss_hessian.T, atol=1e-5):
            print('Warning: Expected loss Hessian is not symmetric')

        return self.expected_loss_hessian

    
    def apply_sigma(self):
        if self.already_computed_sigma:
            if self.Verbose:    
                print('    Already computed sigma')
            return self.Sigma_sqrt, self.diag_Sigma
       
        start = time.time()
        # self.Sigma = torch.zeros(self.number_of_parameters, self.number_of_parameters, device=self.device)
        # Sigma = - torch.outer(self.expected_loss_grad, self.expected_loss_grad)
        # for l in self.Gradient_all_data:
        #     Sigma += (1 / self.Gradient_all_data.shape[0]) * torch.outer(l, l)

        # self.Sigma = torch.cov(self.Gradient_all_data.T)

        # assert self.Sigma.shape == (self.number_of_parameters, self.number_of_parameters), 'Sigma shape is not the same as the number of parameters'

        # eigvals, eigvecs = torch.linalg.eigh(self.Sigma)  
        # if not torch.all(eigvals + self.eps_sqrt >= 0):
        #     print(f'    Smallest eigenvalue of Sigma: {eigvals.min().item()}')
        #     eigvals[eigvals < 0] = 0
        # self.Sigma_sqrt  = eigvecs @ torch.diag(torch.sqrt(eigvals + self.eps_sqrt)) @ eigvecs.T
        # self.diag_Sigma = torch.diag(self.Sigma)

        self.Sigma_sqrt = torch.zeros(self.number_of_parameters, self.number_of_parameters, device=self.device)
        X = self.Gradient_all_data - self.expected_loss_grad
        
        U, S, V = torch.svd(X, compute_uv=True)
        self.Sigma_sqrt  = (1 / torch.sqrt(torch.tensor(X.shape[0] - 1, dtype=torch.float32)) ) * V @ torch.diag(torch.sqrt(S + self.eps_sqrt)) @ V.T

        self.diag_Sigma = (1/torch.tensor(X.shape[0] - 1, dtype=torch.float32)) * torch.sum(X**2, dim=0)
        

        self.already_computed_sigma = True
        if self.Verbose:
            print(f'    Elapsed time Sigma: {time.time() - start:.2f} s')
        return self.Sigma_sqrt, self.diag_Sigma
    
    def compute_gradients_sigma_diag(self):
        if self.already_computed_gradient_sigma_grad:
            if self.Verbose:
                print('    Already computed grad sigma')
            return self.grad_sigma_diag
        start = time.time()
        
        if not self.already_computed_sigma:
            self.apply_sigma()

        # Try to impl,ement \grad \Sigma_k = 2/N \sum (\partial_k grad f - \partial_k E[grad f]) \grad (\partial_k grad f - \partial_k E[grad f])
        self.grad_sigma_diag = torch.zeros(self.number_of_parameters, self.number_of_parameters, device=self.device)
        for k in range(self.diag_Sigma.shape[0]):
            first_term = (2/self.Hessian_all_data.shape[0]) * (self.Hessian_all_data[:, k, :].T @ self.Gradient_all_data[:, k])
            second_term = 2 * self.expected_loss_hessian[k] * self.expected_loss_grad[k]
            self.grad_sigma_diag[k] = first_term - second_term

        assert self.grad_sigma_diag.shape == (self.number_of_parameters, self.number_of_parameters), 'Gradient of Sigma diagonal shape is not the same as the number of parameters'

        self.already_computed_gradient_sigma_grad = True
        if self.Verbose:
            print(f'    Elapsed time grad sigma: {time.time() - start:.2f} s')
        return self.grad_sigma_diag
    
    def compute_var_z_squared(self):
        if self.already_computed_var_z_squared:
            if self.Verbose:
                print('    Already computed var_z_squared')
            return self.square_root
        start = time.time()
        
        if not self.already_computed_sigma:
            self.apply_sigma()


        X = torch.stack([self.diag_Sigma] * self.Gradient_all_data.shape[0]) - (self.Gradient_all_data - self.expected_loss_grad) ** 2
        U, S, V = torch.svd(X, compute_uv=True)
        self.square_root_var_z_squared  = (1 / torch.sqrt(torch.tensor(X.shape[0] - 1, dtype=torch.float32)) ) * V @ torch.diag(torch.sqrt(S + self.eps_sqrt)) @ V.T


        # var_z_squared = torch.zeros(self.number_of_parameters, self.number_of_parameters, device=self.device)
        # var_z_squared = - torch.outer(self.diag_Sigma, self.diag_Sigma)
        # for  G_l in self.Gradient_all_data:
        #     aux = (G_l-self.expected_loss_grad)**2
        #     var_z_squared += (1 / self.Gradient_all_data.shape[0]) * torch.outer(aux, aux)
        
        # Gaussian approximation
        # var_z_squared = torch.cov(( (self.Gradient_all_data - self.expected_loss_grad)**2 ).T)
        # Sigma_squared_component = self.Sigma ** 2
        # var_z_squared = Sigma_squared_component + torch.outer(self.diag_Sigma, self.diag_Sigma) 


        # if torch.isnan(var_z_squared).any():
        #     breakpoint()

        # eigvals, eigvecs = torch.linalg.eigh(var_z_squared)
        # if not torch.all(eigvals + self.eps_sqrt >= 0):
            # print(f'    Smallest eigenvalue of var_z: {eigvals.min().item()}')
            # eigvals[eigvals < 0] = 0
        # self.square_root_var_z_squared  = eigvecs @ torch.diag(torch.sqrt(eigvals + self.eps_sqrt)) @ eigvecs.T

        assert self.square_root_var_z_squared.shape == (self.number_of_parameters, self.number_of_parameters), 'Variance of z squared shape is not the same as the number of parameters'

        self.already_computed_var_z_squared = True
        if self.Verbose:
            print(f'    Elapsed time var_z_squared: {time.time() - start:.2f} s')
        return self.square_root_var_z_squared

        