from torch import nn


class Generator(nn.Module):
    """
    Deep Convolutional GAN generator class, TBD
    """

    def __init__(self, noise_dim=10, image_channel=1, hidden_dim=64) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.image_channel = image_channel
        # Building the generator block of the network

        self.gen = nn.Sequential(
            self.generator_block(self.noise_dim, hidden_dim * 4),
            self.generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.generator_block(hidden_dim * 2, hidden_dim),
            self.generator_block(hidden_dim, self.image_channel, kernel_size=4, final_layer=True),
        )

    def generator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
    ):
        # Conditional return according to the layer type
        return (
            nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
            if not final_layer
            else nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )
        )

    def unsqueeze_noise_vector(self, noise):
        """
        Reshaping noise vector by using .view to be compatible with model shapes
        """
        return noise.view(len(noise), self.noise_dim, 1, 1)
    
    
    def forward(self, noise): 
        """
        Forward propagation of network
        Parameters:
            noise: a noise tensor with dimensions (n_samples, noise_dim)
        """
        return self.gen(self.unsqueeze_noise_vector(noise))
    
    
class Discriminator(nn.Module):
    
    def __init__(self, image_channel=1, hidden_dim=16):
        super().__init__()
        self.discriminator = nn.Sequential(
            self.discriminator_block(image_channel, hidden_dim),
            self.discriminator_block(hidden_dim, hidden_dim*2),
            self.discriminator_block(hidden_dim*2, 1, final_layer=True),
        )
    
    def discriminator_block(
        self, 
        input_dim, 
        output_dim, 
        kernel_size=4,
        stride=2,
        final_layer=False,
    ):
        return (
            nn.Sequential(
                nn.Conv2d(input_dim,output_dim, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)   
            )    
            if not final_layer
            else nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride)
            )
        )
    
    def forward(self, image):
        disc_pred = self.discriminator(image)
        return disc_pred.view(len(disc_pred), -1)
    
    