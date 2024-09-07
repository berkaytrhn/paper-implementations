from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from tqdm import tqdm




from model import Generator, Discriminator
from utils import get_image_dimension, get_noise, show_tensor_images, save_model, save_losses


def calculate_generator_loss(discriminator, fake_images, criterion):
    # calculate discriminator prediction for generated fake images
    pred_fake = discriminator(fake_images)
    # calculate generator loss according to discriminator pred for new fake images
    """
    Our y_true value for loss calculation is a bit trivial since as in generator side, 
    our aim is to generate images that discriminator will classify as real ones so our 
    ground_truth values are all ones.
    """
    generator_loss = criterion(pred_fake, torch.ones_like(pred_fake))
    return generator_loss

def calculate_discriminator_loss(discriminator: nn.Module, fake_images, real_images, criterion, device):
    pred_fake = discriminator(fake_images)
    # they are fake, so y_true should be all zeros and same shape with pred_fake
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
    
    pred_real = discriminator(real_images)
    # they are real, so y_true should be all ones and same shape with pred_real
    loss_real = criterion(pred_real, torch.ones_like(pred_real))
    
    # return average of them
    return (loss_real + loss_fake) / 2


def train():

    # Data

    composed = transforms.Compose([
        transforms.ToTensor()
    ])


    dataset = MNIST("../", transform=composed, download=True)


    # Hyper Parameters

    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    noise_dim = 64
    print_every = 5
    batch_size = 128
    lr = 0.00001
    save_model_every = 20 # epochs
    model_base_path = "./models"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    device = "cuda"

    

    # calculate image dim dynamically

    image_dimension = get_image_dimension(dataset)

    gen = Generator(
        noise_dim=noise_dim,
        image_dim=image_dimension
    ).to(device)


    # 3072 for cifar-10
    discriminator = Discriminator(image_dim=image_dimension).to(device)


    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    
    # Training Loop
    debug_print = False
    gen_losses = []
    disc_losses = []


    for epoch in range(n_epochs):
        # Epoch Losses 
        disc_loss, gen_loss = 0, 0
        for real_images, _ in tqdm(dataloader):
            # get batch size
            batch_size = real_images.shape[0]
        
            #  reshape from (batch_size, channel, height, width) to (batch_size, channel*height*width)
            real_images = real_images.reshape(batch_size, -1).to(device)
            # real_images = real_images.view(batch_size, -1).to(device)
            
            
            # Discriminator Part 
            # zero grad before gradient calculations
            disc_opt.zero_grad()
            # get random noise data
            noise = get_noise(batch_size, noise_dim, device)
            # TODO: code works but throws gen not callable!
            fake_images = gen(noise).detach()
            # calculating discriminator loss
            discriminator_loss = calculate_discriminator_loss(discriminator, fake_images, real_images, criterion, device)
            # calculate discriminator gradients according to both fake and real images together 
            """
            We have detached generator output which removed it from computational graph
            for autograd to calculate gradients chained forward pass like:
                fake = Generator(noise) -> output = Discriminator(fake) = Discriminator(Generator(noise))
            We only calculate Discrimator loss, generator only used for passing a fake image to discriminator
            """
            discriminator_loss.backward(retain_graph=True)
            # updating the parameters of discriminator
            disc_opt.step()
            
            
            # Generator Part
            # zero grad before gradient calculations
            gen_opt.zero_grad()
            # generate new noise for generator
            noise_new = get_noise(batch_size, noise_dim, device)
            # generate new fake images for generator step
            fake_images_new = gen(noise_new)
            # calculating generator loss using discriminator output
            generator_loss = calculate_generator_loss(discriminator, fake_images_new, criterion)
            # calculate generator gradients using disc. retained graph and prev. loss calculation
            generator_loss.backward()
            # updating the parameters of generator
            gen_opt.step()
            
            
            
            # for epoch losses
            gen_loss += generator_loss.item()
            disc_loss += discriminator_loss.item()   
        
        if (epoch%save_model_every==0) and (epoch>0):
            save_model(gen, epoch, model_base_path, "generator")
            save_model(discriminator, epoch, model_base_path, "discriminator")

        
        dataloader_length = len(dataloader)
        avg_disc_loss = disc_loss / dataloader_length
        avg_gen_loss = gen_loss / dataloader_length
        
        disc_losses.append(avg_disc_loss)
        gen_losses.append(avg_gen_loss)
        
        if (epoch%print_every==0):
            
            print(f"Epoch: {epoch} --> Generator loss: {avg_gen_loss}, Discriminator loss: {avg_disc_loss}")
            if debug_print:
                fake_noise = get_noise(batch_size, noise_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                show_tensor_images(real_images)
            
    
    save_losses(gen_losses, "generator_losses.npy")
    save_losses(disc_losses, "discriminator_losses.npy")
    
    save_model(gen, epoch, model_base_path, "generator")
    save_model(discriminator, epoch, model_base_path, "discriminator")    
            
        
    


if __name__ == "__main__":
    # TODO: integrate with config or running parameters
    train()
    
    
""" if (epoch%print_every==0) and (epoch>0):    
    avg_disc_loss = disc_loss / len(dataloader)
    avg_gen_loss = gen_loss / len(dataloader)
    print(f"Epoch {epoch}: Generator Loss: {avg_gen_loss}, Discriminator Loss: {avg_disc_loss}")
"""