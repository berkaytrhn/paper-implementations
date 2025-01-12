import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2   
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
  
def save_tensor_images(image_tensor, save_path, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(save_path)
    plt.close()

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