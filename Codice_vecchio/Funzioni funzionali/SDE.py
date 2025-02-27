import torch
import time



class RMSprop_SDE:
    def __init__(self, eta, beta, eps, function_f_and_Sigma):
        self.noise_type = "general"
        self.sde_type = "ito"

        self.eta = torch.tensor(eta)
        self.c = (1-beta)/eta
        self.eps = eps
        self.function_f_and_Sigma = function_f_and_Sigma

        self.theta_old = None
        self.diffusion = None
        self.drift = None
        self.i = 1
        self.j = 1
        self.start_new_f = None
        self.found_Nan = False

        self.All_gradient = []

    def get_gradient(self):
        return torch.stack(self.All_gradient)

    def f(self, t, x):
        # if self.found_Nan:
        #     return torch.full((1, 180), float('nan'))
        if self.start_new_f is not None:    
            print(f'time between f {self.i} calls {(time.time() - self.start_new_f):.2f}s, time: {t}')
        self.i += 1
        self.start_new_f = time.time()
        
        x = x.squeeze()
        aux = x.shape[0] // 3

        theta = x[:aux]
        v = x[aux:2*aux]
        if (v < 0).any():
            print(f"Warning: Some components of v are negative, {(v<0).sum()}, {torch.norm(v[v < 0])}")
            v[v < 0] = 0
        w = x[2*aux:3*aux]

        
        if self.theta_old is None or (self.theta_old != theta).any():
            self.drift = None
            self.diffusion = None

            self.theta_old = theta

            self.function_f_and_Sigma.Calculate_hessian_gradient(theta)
            self.f_grad = self.function_f_and_Sigma.compute_gradients_f()
            self.All_gradient.append(self.f_grad)
            self.f_hessian = self.function_f_and_Sigma.compute_hessian_f()
            self.Sigma_sqrt, self.diag_Sigma = self.function_f_and_Sigma.apply_sigma()
            self.diag_grad_sigma = self.function_f_and_Sigma.compute_gradients_sigma_diag()
            self.square_root_var_z_squared = self.function_f_and_Sigma.compute_var_z_squared()

        print(f'v: {torch.norm(v)}, grad: {torch.norm(self.f_grad)}, hessian: {torch.norm(self.f_hessian)}')        

        f_grad_square = torch.pow(self.f_grad, 2)    

        # Theta coefficient
        denom = 1/(torch.sqrt(v) + self.eps)
        coef_theta = -( 1+ self.eta/2 * torch.diag(denom) @ (self.f_hessian + self.c * torch.diag( 0.5 * denom * (f_grad_square + self.diag_Sigma - v))))
        coef_theta = coef_theta @ (self.f_grad * denom)

        # V coefficient
        coef_v = (self.c + self.c**2 * self.eta/2) * (f_grad_square + self.diag_Sigma - v)
        coef_v = coef_v + 0.5 * self.eta * self.c * (2 * torch.diag(self.f_grad) @ self.f_hessian  + self.diag_grad_sigma) @ (self.f_grad * denom)

        # W coefficient
        coef_w = torch.zeros_like(w)

        self.drift = torch.concat((coef_theta, coef_v, coef_w), dim = 0).unsqueeze(0)
        if torch.isnan(self.drift).any():
            print("Warning: NaN values detected in drift")
            self.found_Nan = True
        return self.drift
    

    def g(self, t, x):
        # if self.found_Nan:
        #     return torch.full((1, 180, 60), float('nan'))

        x = x.squeeze()
        aux = x.shape[0] // 3

        theta = x[:aux]
        v = x[aux:2*aux]
        if (v < 0).any():
            print(f"Warning: Some components of v are negative,{(v<0).sum()}, {torch.norm(v[v < 0])}")
            v[v < 0] = 0
        w = x[2*aux:3*aux]  

        if self.theta_old is None or (self.theta_old != theta).any():
            self.diffusion = None
            self.drift = None

            self.theta_old = theta

            self.function_f_and_Sigma.Calculate_hessian_gradient(theta)

            self.f_grad = self.function_f_and_Sigma.compute_gradients_f()
            self.All_gradient.append(self.f_grad)
            self.f_hessian = self.function_f_and_Sigma.compute_hessian_f()
            self.Sigma_sqrt, self.diag_Sigma = self.function_f_and_Sigma.apply_sigma()
            
            self.diag_grad_sigma = self.function_f_and_Sigma.compute_gradients_sigma_diag()
            self.square_root_var_z_squared = self.function_f_and_Sigma.compute_var_z_squared()   

        M_theta = torch.sqrt(self.eta) / (torch.sqrt(v) + self.eps) * self.Sigma_sqrt
        M_v = -2 * torch.sqrt(self.eta) * self.c * torch.diag(self.f_grad) *  self.Sigma_sqrt + torch.sqrt(torch.tensor(1/2)) * self.c * self.square_root_var_z_squared * w
        M_w = torch.ones_like(M_theta)
        self.diffusion = torch.concat((M_theta, M_v, M_w), dim = 0).unsqueeze(0)

        # print(f'chiamate g {self.j}, time: {(time.time() - start_g):.2f}s')
        if torch.isnan(self.diffusion).any():
            self.found_Nan = True
            print("Warning: NaN values detected in diffusion")
        return self.diffusion






