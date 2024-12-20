import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tensorflow.keras.datasets import cifar10
import cv2
import matplotlib.pyplot as plt
import torch

class CIFAR10Dataset:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32).permute(0, 3, 1, 2)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y_train = torch.tensor(self.y_train, dtype=torch.int64).squeeze()
        self.y_test = torch.tensor(self.y_test, dtype=torch.int64).squeeze()

    def to_grayscale(self):
        self.x_train = self.x_train.permute(0, 2, 3, 1)
        self.x_test = self.x_test.permute(0, 2, 3, 1)

        aux = self.x_train.numpy()
        self.x_train = torch.tensor(np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in aux]), dtype=torch.float32)
        size = self.x_train.shape[1]
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, size, size)
        aux = self.x_test.numpy()
        self.x_test = torch.tensor(np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in aux]), dtype=torch.float32)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, size, size)

    def downscale(self, scale_percent):

        self.x_train = self.x_train.permute(0, 2, 3, 1)
        self.x_test = self.x_test.permute(0, 2, 3, 1)

        width = int(self.x_train.shape[1] * scale_percent // 100)
        height = int(self.x_train.shape[2] * scale_percent // 100)
        
        dim = (width, height)

        aux = self.x_train.numpy()
        self.x_train = torch.tensor(np.array([cv2.resize(image, dim, interpolation=cv2.INTER_AREA) for image in aux]), dtype=torch.float32)
        aux = self.x_test.numpy()
        self.x_test = torch.tensor(np.array([cv2.resize(image, dim, interpolation=cv2.INTER_AREA) for image in aux]), dtype=torch.float32)

        if len(self.x_train.shape) == 3:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.x_train.shape[1], self.x_train.shape[2])
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.x_test.shape[1], self.x_test.shape[2])
        else:
            self.x_train = self.x_train.permute(0, 3, 1, 2)
            self.x_test = self.x_test.permute(0, 3, 1, 2)

    def show_image(self, index):
        aux = self.x_train[index].to(torch.uint8).permute(1, 2, 0)
        if self.x_train[index].shape[2] == 1:  
            plt.imshow(aux.squeeze(), cmap='gray')
        else:  
            plt.imshow(aux)
        plt.title(f"Label: {self.y_train[index]}")
        plt.show()

    def get_image_size(self):
        return self.x_train[0].shape

    def dataloader(self, batch_size, steps, seed = 0):
        
        torch.manual_seed(seed)
        indices = torch.randint(0, self.x_train.shape[0], (batch_size * steps,))
        sampled_x_train = self.x_train[indices]
        sampled_y_train = self.y_train[indices]

        tensor_dataset = TensorDataset(sampled_x_train, sampled_y_train)
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        return dataloader




def main():
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

if __name__ == '__main__':
    main()