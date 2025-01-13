import torch
import time



class RMSprop_SDE:
    def __init__(self, eta, beta, function_f_and_Sigma, eps = 1e-8, Verbose = True):
        self.noise_type = "general"
        self.sde_type = "ito"

        self.eta = torch.tensor(eta)
        self.c = (1-beta)/eta
        self.eps = eps
        self.C_regularizer = None
        self.function_f_and_Sigma = function_f_and_Sigma


        self.theta_old = None
        self.diffusion = None
        self.drift = None
        self.i = 1
        self.start_new_f = None
        self.found_Nan = False
        self.verbose = Verbose


    def f(self, t, x):
        if self.start_new_f is None:
            self.start_new_f = time.time()
        if self.start_new_f is not None:   
            if self.i % 100 == 0: 
                print(f'time between f {self.i} calls {(time.time() - self.start_new_f):.2f}s, time: {t:.4f}')
                self.start_new_f = time.time()
        self.i += 1
        
        x = x.squeeze()
        aux = x.shape[0] // 3

        theta = x[:aux]
        v = x[aux:2*aux]
        if (v < 0).any():
            if min(v) < -1e-6: print(f"Warning: Some components of v are negative, {(v<0).sum()}, {torch.norm(v[v < 0])}, time: {t}")
            v[v < 0] = 0
        w = x[2*aux:3*aux]

        if self.C_regularizer is None:
            self.C_regularizer = (1-self.eta) * torch.sum(v) + 1e-8

        if self.theta_old is None or (self.theta_old != theta).any():
            self.update_quantities(theta)

        
        if self.verbose:
            print(f'v: {torch.norm(v)/v.shape[0]}, grad: {torch.norm(self.f_grad)/self.f_grad.shape[0]}')        

        f_grad_square = torch.pow(self.f_grad, 2)    

        # Theta coefficient
        denom = 1/(torch.sqrt(regulariz_function(v, self.C_regularizer)) + self.eps)
        # coef_theta =  torch.diag(denom) @ (self.f_hessian @ self.f_grad + self.c * torch.diag( 0.5 * denom * (f_grad_square + self.diag_Sigma - v)) @ (self.f_grad * denom * torch.pow(regulariz_function(v, self.C_regularizer), -0.5) * derivative_regulariz_function(v, self.C_regularizer))) @ denom

        coef_theta = (self.f_hessian * (denom @ denom.T)) @ self.f_grad + self.c*(f_grad_square + self.diag_Sigma - v) * (self.f_grad * torch.pow(denom, 3) * derivative_regulariz_function(v, self.C_regularizer))
        coef_theta = - self.f_grad  * denom - self.eta/2 * coef_theta

        # V coefficient
        coef_v = (self.c + self.c**2 * self.eta/2) * (f_grad_square + self.diag_Sigma - v)
        coef_v = coef_v + 0.5 * self.eta * self.c * ((2 * torch.diag(self.f_grad) @ self.f_hessian + self.diag_grad_sigma)@ self.f_grad) * denom

        # W coefficient
        coef_w = torch.zeros_like(w)
        self.drift = torch.concat((coef_theta, coef_v, coef_w), dim = 0).unsqueeze(0)
        if torch.isnan(self.drift).any():
            print("Warning: NaN values detected in drift")
            self.found_Nan = True
        return self.drift
    

    def g(self, t, x):
        x = x.squeeze()
        aux = x.shape[0] // 3

        theta = x[:aux]
        v = x[aux:2*aux]
        if (v < 0).any():
            if min(v) < -1e-6: print(f"Warning: Some components of v are negative,{(v<0).sum()}, {torch.norm(v[v < 0])}, time: {t}")
            v[v < 0] = 0
        w = x[2*aux:3*aux]  

        if self.C_regularizer is None:
            self.C_regularizer = (1-self.eta) * torch.sum(v) + 1e-8

        if self.theta_old is None or (self.theta_old != theta).any():
            self.update_quantities(theta)
        
        denom = 1/(torch.sqrt(regulariz_function(v, self.C_regularizer)) + self.eps)

        M_theta = torch.sqrt(self.eta) * torch.diag(denom) * self.Sigma_sqrt
        M_v = -2 * torch.sqrt(self.eta) * self.c * torch.diag(self.f_grad) *  self.Sigma_sqrt + torch.sqrt(torch.tensor(1/2)) * self.c * self.square_root_var_z_squared * w
        M_w = torch.ones_like(M_theta)
        self.diffusion = torch.concat((M_theta, M_v, M_w), dim = 0).unsqueeze(0)

        if torch.isnan(self.diffusion).any():
            self.found_Nan = True
            print("Warning: NaN values detected in diffusion")
            breakpoint()
        return self.diffusion
    
    def update_quantities(self, theta):
        self.diffusion = None
        self.drift = None

        self.theta_old = theta

        self.function_f_and_Sigma.update_parameters(theta)
        self.f_grad = self.function_f_and_Sigma.compute_gradients_f()
        self.f_hessian = self.function_f_and_Sigma.compute_hessian()
        self.Sigma_sqrt, self.diag_Sigma = self.function_f_and_Sigma.compute_sigma()
        self.diag_grad_sigma = self.function_f_and_Sigma.compute_gradients_sigma_diag()
        self.square_root_var_z_squared = self.function_f_and_Sigma.compute_var_z_squared()

def regulariz_function(x, C):
    return torch.where(x > C, x, C * torch.exp(x / C - 1))

def derivative_regulariz_function(x, C):
    return torch.where(x > C, 1, torch.exp(x / C - 1))





