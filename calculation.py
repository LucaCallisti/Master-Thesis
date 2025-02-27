import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import CNN


import time

torch.backends.cuda.preferred_linalg_library('magma')

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
    def __init__(self, model, dataset, Verbose = False, dim_dataset = 64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.number_of_parameters = sum(p.numel() for p in self.model.parameters())
        self.CNN_functina = CNN.functional_CNN(self.model)
        self.initial_params = self.CNN_functina.get_initial_params()
        self.split_sizes = [v.numel() for v in self.initial_params]

        self.x_train = dataset.x_train[:dim_dataset]
        self.y_train = dataset.y_train[:dim_dataset]
        if len(self.y_train.shape) == 2:
            self.y_train = self.y_train.view(-1)
        self.CNN_functina.set_input(self.x_train, self.y_train)
        self.dim_dataset = self.x_train.shape[0]

        self.already_computed_expected_loss_grad, self.already_computed_expected_loss_hessian, self.already_computed_sigma, self.already_computed_gradient_sigma_grad, self.already_computed_var_z_squared = False, False, False, False, False
        self.Verbose = Verbose
    
    def update_parameters(self, new_parameters):
        self.params = torch.split(new_parameters, self.split_sizes)
        self.already_computed_expected_loss_grad, self.already_computed_expected_loss_hessian, self.already_computed_sigma, self.already_computed_gradient_sigma_grad, self.already_computed_var_z_squared = False, False, False, False, False


    def compute_f(self):
        self.CNN_functina.which_loss = 'mean'
        return self.CNN_functina.function(*self.params)

    def compute_gradients_f(self):
        if self.already_computed_expected_loss_grad:
            if self.Verbose: print('    Already computed grad f')
            return self.expected_loss_grad
        
        start = time.time()
        self.all_data_grad = self.CNN_functina.All_data_gradient(self.params) 
        self.expected_loss_grad = self.CNN_functina.expected_loss_gradient(self.params)

        if self.Verbose:
            print(f'    Elapsed time grad f: {time.time() - start:.2f} s')
        self.already_computed_expected_loss_grad = True
        return self.expected_loss_grad
    
    def compute_hessian(self):
        if self.already_computed_expected_loss_hessian:
            if self.Verbose:
                print('    Already computed hessian f')
            return self.expected_loss_hessian
        start = time.time()

        self.expected_loss_hessian = self.CNN_functina.expected_loss_hessian(*self.params)

        if self.Verbose:
            print(f'    Elapsed time hessian f: {time.time() - start:.2f} s')
        return self.expected_loss_hessian

    
    def compute_sigma(self):
        if self.already_computed_sigma:
            if self.Verbose:    
                print('    Already computed sigma')
            return self.Sigma_sqrt, self.diag_Sigma
       
        start = time.time()
        self.Sigma_sqrt, self.diag_Sigma = Square_root_matrix(self.all_data_grad - self.expected_loss_grad)
        if torch.isnan(self.Sigma_sqrt).any() or torch.isinf(self.Sigma_sqrt).any():
            print('Warning: NaN or Inf values detected in self.Sigma_sqrt')
            breakpoint()
        
        self.already_computed_sigma = True
        if self.Verbose:
            print(f'    Elapsed time Sigma: {time.time() - start:.2f} s')
        return self.Sigma_sqrt, self.diag_Sigma
    
    
    def compute_gradients_expected_grad_f_squared(self):
        if self.already_computed_gradient_sigma_grad:
            if self.Verbose:
                print('    Already computed grad sigma')
            return self.grad_expected_frad_f_squared
        start = time.time()

        self.grad_expected_frad_f_squared =  self.CNN_functina.jacobian_my_funct(*self.params)

        

        self.already_computed_gradient_sigma_grad = True
        if self.Verbose:
            print(f'    Elapsed time grad sigma: {time.time() - start:.2f} s')
        
        return self.grad_expected_frad_f_squared
    
    def compute_var_z_squared(self):
        if self.already_computed_var_z_squared:
            if self.Verbose:
                print('    Already computed var_z_squared')
            return self.square_root_var_z
        start = time.time()
                
        if not self.already_computed_sigma:
            self.apply_sigma()
        
        # self.square_root_var_z = Square_root_matrix(self.diag_Sigma - (self.all_data_grad - self.expected_loss_grad) ** 2, False)
        self.square_root_var_z = Square_root_matrix((self.all_data_grad - self.expected_loss_grad) ** 2, False)

        if torch.isnan(self.square_root_var_z).any() or torch.isinf(self.square_root_var_z).any():
            print('Warning: NaN or Inf values detected in self.square_root_var_z')
            breakpoint()

        self.already_computed_var_z_squared = True
        if self.Verbose:
            print(f'    Elapsed time var_z_squared: {time.time() - start:.2f} s')
        return self.square_root_var_z

def Square_root_matrix(matrix, diag = True):
    '''
    matrix is the cenetered data matrix with shape (m, n), where m is the number of samples and n is the number of features
    '''
    Sigma = torch.cov(matrix.T).clamp(max = 1e8)
    if torch.isnan(Sigma).any():
        print('Warning: NaN values detected in Sigma')
        breakpoint()
    if len(Sigma.shape) == 2:
        try:
            eigvals, eigvecs = torch.linalg.eigh(Sigma)
        except torch._C._LinAlgError:
            print('Warning: torch.linalg.eigh')
            U, S, V = torch.svd(Sigma)
            eigvals = S
            eigvecs = V   
        
        if eigvals.min() < -1e-6:
            print('Warning: negative eigenvalues detected in Sigma', (eigvals < 0).sum(), eigvals.min())
        eigvals = torch.clamp(eigvals, min=0)
        Sigma_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    else:
        Sigma_sqrt = torch.sqrt(Sigma)

    if torch.isnan(Sigma_sqrt).any() or torch.isinf(Sigma_sqrt).any():
        print('Warning: NaN or Inf values detected in Sigma_sqrt')
        breakpoint()
    if diag == True:
        if len(Sigma.shape) == 2:
            return Sigma_sqrt, torch.diag(Sigma)
        else:
            return Sigma_sqrt, Sigma
    return Sigma_sqrt


