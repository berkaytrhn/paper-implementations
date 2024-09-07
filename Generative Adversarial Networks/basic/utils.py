import torch
from torch import nn
import os
import numpy as np
from torch.utils.data import Dataset
from functools import reduce
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_noise(n_samples, noise_dim, device) :
    # create normally distributed noise data for generator
    return torch.randn(n_samples, noise_dim).to(device)

def save_model(model:nn.Module, epochs: int, base_path: str, name: str) -> None:
    torch.save(model, os.path.join(base_path, f"{name}_{epochs}.pth"))
    
def save_losses(losses: list, name:str) -> None:
    np.save(os.path.join("./losses", name), np.array(losses))

def load_losses(path: str) -> np.ndarray:
    return np.load(path)

def get_image_dimension(dataset: Dataset) -> int:
    image_dim = reduce(lambda x,y: x*y, dataset[0][0].shape)
    return image_dim



def show_tensor_images(image_tensor, num_images:int=25, size:tuple=(1, 28, 28)) -> None:
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def plot_losses(epochs, generator_losses, discriminator_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, generator_losses, label="Generator Loss", color='blue')
    plt.plot(epochs, discriminator_losses, label="Discriminator Loss", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses')
    plt.legend()  # Shows the labels for the plot
    plt.grid(True)
    plt.show()