import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Dataset
import time

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
    
def count_model_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    return total_params

class function_f_and_Sigma():
    def __init__(self, model, x_train, y_train):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(device)
        self.x_train = x_train.to(device)
        self.y_train = y_train.to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.loss = None
        self.expected_value_loss = None
        self.expected_loss_grad = None
        self.sigma = None

        self.already_computed_loss = False
        self.already_computed_expected_loss_grad = False
        self.already_computed_sigma = False

        self.number_of_parameters = count_model_parameters(self.model)
    
    def update_model_params(self, new_params):
        self.model.load_state_dict(new_params)

        self.already_computed_loss = False
        self.already_computed_expected_loss_grad = False
        self.already_computed_sigma = False

    def apply_f(self):
        if self.already_computed_loss:
            return self.f
        
        result = self.model(self.x_train)

        # Calculate Loss
        self.loss = self.criterion(result, self.y_train)
        
        # Mean over dataset
        self.expected_value_loss = self.loss.mean(axis=0) 
        self.already_computed_loss = True

        assert self.loss.shape[0] == self.x_train.shape[0], 'Loss shape is not the same as the number of samples'
        assert self.expected_value_loss.dim() == 0, 'Expected loss shape is not the same as the number of samples'

        return self.expected_value_loss
    
    def compute_gradients_f(self):
        if self.already_computed_expected_loss_grad:
            return self.f_grad
        
        self.model.zero_grad()

        if self.already_computed_loss:
            output = self.expected_value_loss
        else:
            output = self.apply_f()

        gradients = torch.autograd.grad(output.mean(), self.model.parameters(), create_graph=True)
        gradients = torch.cat([grad.reshape(-1) for grad in gradients])

        self.expected_loss_grad = gradients
        self.already_computed_expected_loss_grad = True

        assert self.expected_loss_grad.shape[0] == self.number_of_parameters, 'Expected loss gradient shape is not the same as the number of parameters'

        return self.expected_loss_grad
    
    def apply_sigma(self):
        if self.already_computed_sigma:
            return self.sigma

        Sigma = 0
        for l in self.loss:
            self.model.zero_grad()
            l_grad = torch.autograd.grad(l, self.model.parameters(), create_graph=True)
            l_grad = torch.cat([grad.reshape(-1) for grad in l_grad])

            Sigma += torch.ger(l_grad-self.expected_loss_grad, l_grad-self.expected_loss_grad)

        self.sigma = Sigma / self.loss.shape[0]
        self.already_computed_sigma = True

        assert self.sigma.shape == (self.number_of_parameters, self.number_of_parameters), 'Sigma shape is not the same as the number of parameters'

        return self.sigma
    
    def compute_gradients_sigma_diag(self):
        if not self.already_computed_sigma:
            self.apply_sigma()
        
        sigma_diag = torch.diag(self.sigma)
        grad_sigma_diag = []
        for d in sigma_diag:
            self.model.zero_grad()
            gradients = torch.autograd.grad(d, self.model.parameters(), create_graph=True)
            gradients = torch.cat([grad.reshape(-1) for grad in gradients])
            grad_sigma_diag.append(gradients)
        self.grad_sigma_diag = torch.stack(grad_sigma_diag)

        assert self.grad_sigma_diag.shape == (self.number_of_parameters, self.number_of_parameters), 'Gradient of Sigma diagonal shape is not the same as the number of parameters'

        return self.grad_sigma_diag


class Train_n_times():
    def __init__(self, model, dataset, steps=100, lr=0.01, optimizer_name='RMSPROP', eta = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.steps = steps
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.eta = eta
        self.initial_model_params = model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
    


    def train_n_times(self, n, batch_size=32):
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

            History = {'Params' : optimizer.state_dict()['param_groups'].copy()}
            History['Loss'] = {}
            Loss = []
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

                if isinstance(optimizer, torch.optim.RMSprop):
                    square_avg = {k: v['square_avg'].clone() for k, v in self.optimizer.state_dict()['state'].items()}
                    History[i+1] = {'model' : model.state_dict().copy(), 'square_avg' : square_avg}
                if isinstance(optimizer, torch.optim.Adam):
                    adam_param = {k: {'exp_avg' : v['exp_avg'].clone(), 'exp_avg_sq' : v['exp_avg_sq'].clone()} for k, v in self.optimizer.state_dict()['state'].items()}
                    History[i+1] = {'model' : model.state_dict().copy(), 'adam_param' : adam_param}

                if i == steps:
                    break

            History['Loss'] = Loss
            
            if isinstance(optimizer, torch.optim.RMSprop):
                initial_square_avg = History[1]['square_avg'].copy()
                for key in initial_square_avg.keys():
                    initial_square_avg[key] = torch.zeros_like(initial_square_avg[key])

                History[0] = {'model' : aux_initial_parameters, 'square_avg' : initial_square_avg}
            if isinstance(optimizer, torch.optim.Adam):
                initial_adam_param = {k: {'exp_avg': v['exp_avg'].clone(), 'exp_avg_sq': v['exp_avg_sq'].clone()} for k, v in History[1]['adam_param'].items()}
                for key in initial_adam_param.keys():
                    initial_adam_param[key]['exp_avg'] = torch.zeros_like(initial_adam_param[key]['exp_avg'])
                    initial_adam_param[key]['exp_avg_sq'] = torch.zeros_like(initial_adam_param[key]['exp_avg_sq'])

                History[0] = {'model' : aux_initial_parameters, 'adam_param' : initial_adam_param}

            return History
        
        Different_run = {}
        for i in range(n):
            self.model.load_state_dict(self.initial_model_params)
            dataloader = self.dataset.dataloader(batch_size = batch_size, steps=self.steps)

            if self.optimizer_name == 'RMSPROP':
                if self.eta == None:
                    self.eta = 0.9
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=self.eta, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            elif self.optimizer_name == 'ADAM':
                if self.eta == None: 
                    self.eta = (0.9, 0.999)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.eta, eps=1e-08, weight_decay=0, amsgrad=False)
            else:
                raise ValueError('Invalid optimizer')
            History = train(self.model, dataloader, self.optimizer, self.criterion, steps=self.steps)
            import matplotlib.pyplot as plt

            Different_run[i+1] = History
        
        self.Different_run = Different_run
        return Different_run
    
    def save_dict(self, path):
        torch.save(self.Different_run, path)
            
        
    
    def get_model(self):
        return self.model
    



