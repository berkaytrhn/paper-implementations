import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from config import Config
from dto import DatasetConfiguration, LoggingConfiguration, ModelSaveConfiguration, TrainConfiguration
from model import Discriminator, Generator
from utils import show_tensor_images, save_tensor_images
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Train:
    
    summary_logger: SummaryWriter=None
    
    def __init__(self, config: Config):
        cfg = config.config
        self.data_cfg = DatasetConfiguration(cfg["data"])
        self.train_cfg = TrainConfiguration(cfg["train"])
        self.logging_cfg = LoggingConfiguration(cfg["logging"])
        self.model_cfg = ModelSaveConfiguration(cfg["model"])
    
    
    def set_device(self) -> None:
        if torch.cuda.is_available() and self.train_cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def load_dataset(self) -> None:
        composed_transforms = transforms.Compose([
            transforms.ToTensor(), # performs scaling by default for image datasets between range(0-1)
        ])
        
        train_val_set = MNIST(
            self.data_cfg.train_set, 
            train=True, 
            transform=composed_transforms, 
            download=True
        )
        
        
        self.test_set = MNIST(
            self.data_cfg.test_set, 
            train=False, 
            transform=composed_transforms, 
            download=True
        )
        
        # split train-val into train set and validation set
        self.train_set, self.validation_set = random_split(
            dataset=train_val_set, 
            lengths=[
                self.data_cfg.train_set_length, 
                self.data_cfg.test_set_length
            ]
        )
        
    def configure_data_loaders(self) -> None:
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.train_cfg.batch_size, 
            shuffle=True
        )
        
        self.validation_loader = DataLoader(
            self.validation_set, 
            batch_size=self.train_cfg.batch_size, 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            self.test_set, 
            batch_size=self.train_cfg.batch_size, 
            shuffle=True
        )
        
    
    def build_model(self):
        # Adam optimizer beta values
        beta_1 = 0.5 
        beta_2 = 0.999
        self.gen = Generator(noise_dim=self.train_cfg.noise_dimension, image_channel=self.train_cfg.image_channel).to(self.device)
        self.gen_optimizer = Adam(self.gen.parameters(), lr=self.train_cfg.learning_rate, betas=(beta_1, beta_2))

        # image channel as 1 for MNIST
        self.disc = Discriminator(image_channel=self.train_cfg.image_channel).to(self.device)
        self.disc_optimizer = Adam(self.disc.parameters(), lr=self.train_cfg.learning_rate, betas=(beta_1, beta_2))
        
        # set loss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
            
    def configure_logging(self):
        self.summary_logger = SummaryWriter(
            os.path.join(
                self.logging_cfg.directory,
                datetime.now().strftime("%d.%m.%Y"),
                self.logging_cfg.sub_directory,
                self.logging_cfg.model_name
            )
        )
    

    def _calculate_generator_loss(self, discriminator, fake_images):
        # calculate discriminator prediction for generated fake images
        pred_fake = discriminator(fake_images)
        # calculate generator loss according to discriminator pred for new fake images
        """
        Our y_true value for loss calculation is a bit trivial since as in generator side, 
        our aim is to generate images that discriminator will classify as real ones so our 
        ground_truth values are all ones.
        """
        generator_loss = self.criterion(pred_fake, torch.ones_like(pred_fake))
        return generator_loss
                    
    def _calculate_discriminator_loss(self, discriminator, fake_images, real_images):
        pred_fake = discriminator(fake_images)
        # they are fake, so y_true should be all zeros and same shape with pred_fake
        loss_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
        
        pred_real = discriminator(real_images)
        # they are real, so y_true should be all ones and same shape with pred_real
        loss_real = self.criterion(pred_real, torch.ones_like(pred_real))
        
        # return average of them
        return (loss_real + loss_fake) / 2
    
    def _get_noise(self):
        # create normally distributed noise data for generator
        return torch.randn(self.train_cfg.batch_size, self.train_cfg.noise_dimension, device=self.device)

    def save_model(self, model:torch.nn.Module, base_path: str, name: str):
        torch.save(model, os.path.join(base_path, f"{name}_{self.train_cfg.epochs}.pth"))
    
    def _train_step(self, epoch: int):
        
        dataset_length:int = len(self.train_loader)
        
        _generator_loss = 0
        _discriminator_loss = 0
        cur_step = 0
        for real_images, _ in tqdm(self.train_loader):
            # get batch size
            batch_size = real_images.shape[0]

            # to device
            real_images = real_images.to(self.device)
            
            # Discriminator Part 
            # zero grad before gradient calculations
            self.disc_optimizer.zero_grad()
            # get random noise data
            noise = self._get_noise()
            fake_images = self.gen(noise).detach()
            # calculating discriminator loss
            discriminator_loss = self._calculate_discriminator_loss(
                self.disc, 
                fake_images, 
                real_images
            )
            # Keep track of the average discriminator loss
            _discriminator_loss += discriminator_loss.item()
            
            # calculate discriminator gradients according to both fake and real images together 
            """
            We have detached generator output which removed it from computational graph
            for autograd to calculate gradients chained forward pass like:
                fake = Generator(noise) -> output = Discriminator(fake) = Discriminator(Generator(noise))
            We only calculate Discrimator loss, generator only used for passing a fake image to discriminator
            """
            discriminator_loss.backward(retain_graph=True)
            # updating the parameters of discriminator
            self.disc_optimizer.step()
            

            # Generator Part
            # zero grad before gradient calculations
            self.gen_optimizer.zero_grad()
            # generate new noise for generator
            noise_new = self._get_noise()
            # generate new fake images for generator step
            fake_images_new = self.gen(noise_new)
            # calculating generator loss using discriminator output
            generator_loss = self._calculate_generator_loss(
                self.disc, 
                fake_images_new
            )
            # calculate generator gradients using disc. retained graph and prev. loss calculation
            generator_loss.backward()
            # updating the parameters of generator
            self.gen_optimizer.step()

            # Keep track of the average generator loss
            _generator_loss += generator_loss.item()


            if (self.train_cfg.show_and_save_by) and (cur_step % self.train_cfg.show_and_save_by == 0) and (cur_step > 0):
                show_tensor_images(fake_images)
                save_tensor_images(fake_images, f"./{self.model_cfg.results}/generated_{epoch}_{cur_step}.png")
            
            cur_step += 1
        
        _generator_loss /= dataset_length
        _discriminator_loss /= dataset_length    
        
        return _generator_loss, _discriminator_loss
    
    def train(self):
        for epoch in range(self.train_cfg.epochs):
            
            # Dataloader returns the batches
            _train_generator_loss, _train_discriminator_loss = self._train_step(epoch)
            
            print(f"Epoch {epoch}, Generator loss: {_train_generator_loss}, discriminator loss: {_train_discriminator_loss}")
            
            # SummaryWriter for losses
            self.summary_logger.add_scalars(
                main_tag="Losses",
                tag_scalar_dict={
                    "train/generator_loss": _train_generator_loss,
                    "train/discriminator_loss": _train_discriminator_loss
                },
                global_step=epoch
            )
            
            
    # TODO: Implement the validation and test process(Another paper implementation: FID)
              
                
def main(args: argparse.Namespace):
    
    cfg = Config(args.cfg)
    
    trainer = Train(cfg)
    trainer.set_device()
    trainer.load_dataset()
    trainer.configure_data_loaders()
    trainer.build_model()
    trainer.configure_logging()
    
    trainer.gen = trainer.gen.apply(trainer.weights_init)
    trainer.disc = trainer.disc.apply(trainer.weights_init)
    
    trainer.train()
    trainer.save_model(trainer.gen, "./models", "dcgan_model_gen")
    trainer.save_model(trainer.disc, "./models", "dcgan_model_disc")
    
    


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(
        prog='DCGAN Train',
        description='DCGAN Training Process')
    
    
    parser.add_argument("-c", "--cfg", default="./train.yml", required=False)
    
    args = parser.parse_args()
    main(args)