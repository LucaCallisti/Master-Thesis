import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
            x = F.relu(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x
    
    def get_number_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)   


class Large_CNN(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers, size_img, pool_size=2, number_of_networks = 64): 
        super(Large_CNN, self).__init__()
        self.number_of_networks = number_of_networks
        self.models = nn.ModuleList([CNN(input_channels, num_classes, conv_layers, size_img, pool_size) for _ in range(number_of_networks)])

    def forward(self, X_list):
        assert X_list.shape[0] == self.number_of_networks, f"Expected {self.number_of_networks} inputs, but got {X_list.shape[0]}"

        futures = [torch.jit.fork(self.models[i], X_list[i]) for i in range(self.number_of_networks)]
        outputs = torch.stack([torch.jit.wait(future) for future in futures])
        return outputs.squeeze()
    
    def update_all_cnns_with_params(self, ref_cnn_params):
        """
        Aggiorna tutte le CNN nel modello con i parametri forniti.
        
        Args:
            ref_cnn_params (iterable): Lista o iteratore dei parametri da una CNN di riferimento (ottenuti con list(cnn.parameters())).
        """

        # Itera su tutte le CNN e aggiorna i loro parametri
        for cnn in self.models:
            cnn.load_state_dict(ref_cnn_params)
            # for p_target, p_source in zip(cnn.parameters(), ref_cnn_params):
            #     p_target.data.copy_(p_source.data)


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
    


    def train_n_times(self, n = 1, batch_size=32):
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
                    square_avg = {k: v['square_avg'].clone() for k, v in self.optimizer.state_dict()['state'].items()}
                    History[i+1] = {'model' : model.state_dict().copy(), 'square_avg' : square_avg}
                if isinstance(optimizer, torch.optim.Adam):
                    adam_param = {k: {'exp_avg' : v['exp_avg'].clone(), 'exp_avg_sq' : v['exp_avg_sq'].clone()} for k, v in self.optimizer.state_dict()['state'].items()}
                    History[i+1] = {'model' : model.state_dict().copy(), 'adam_param' : adam_param}

                if i == steps:
                    break

            History['Loss'] = Loss
            History['Accuracy'] = Accuracy
            
            if isinstance(optimizer, torch.optim.RMSprop):
                initial_square_avg = History[1]['square_avg'].copy()
                for key in initial_square_avg.keys():
                    initial_square_avg[key] = torch.zeros_like(initial_square_avg[key], device=device)

                History[0] = {'model' : aux_initial_parameters, 'square_avg' : initial_square_avg}
            if isinstance(optimizer, torch.optim.Adam):
                initial_adam_param = {k: {'exp_avg': v['exp_avg'].clone(), 'exp_avg_sq': v['exp_avg_sq'].clone()} for k, v in History[1]['adam_param'].items()}
                for key in initial_adam_param.keys():
                    initial_adam_param[key]['exp_avg'] = torch.zeros_like(initial_adam_param[key]['exp_avg'], device=device)
                    initial_adam_param[key]['exp_avg_sq'] = torch.zeros_like(initial_adam_param[key]['exp_avg_sq'], device=device)

                History[0] = {'model' : aux_initial_parameters, 'adam_param' : initial_adam_param}

            return History
        
        Different_run = {}
        for i in range(n):
            self.model.load_state_dict(self.initial_model_params)
            dataloader = self.dataset.dataloader(batch_size = batch_size, steps=self.steps)

            if self.optimizer_name == 'RMSPROP':
                if self.eta == None:
                    self.eta = 0.999
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=self.eta, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            elif self.optimizer_name == 'ADAM':
                if self.eta == None: 
                    self.eta = (0.9, 0.999)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.eta, eps=1e-08, weight_decay=0, amsgrad=False)
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
    



