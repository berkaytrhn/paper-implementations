from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_dim=10, image_dim=784, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            # Linear Layers with BatchNormalizations
            self.generator_block(noise_dim, hidden_dim),
            self.generator_block(hidden_dim, hidden_dim*2),
            self.generator_block(hidden_dim*2, hidden_dim*4),
            self.generator_block(hidden_dim*4, hidden_dim*8),
            
            # Output Layer and Sigmoid
            nn.Linear(hidden_dim*8, image_dim),
            nn.Sigmoid()
        )

    def forward(self, noise_vector):
        return self.gen(noise_vector)

    def generator_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.BatchNorm1d(out_features), 
            nn.ReLU(inplace=True)
        )

        

class Discriminator(nn.Module):
    def __init__(self, image_dim=784, hidden_dim=128):
        super().__init__() # in python 2.x we call with class itself like "super(Discriminator, self).__init__"
        self.discriminator = nn.Sequential(
            # Basic Linear Layers with activation
            self.discriminator_block(image_dim, hidden_dim*4),
            self.discriminator_block(hidden_dim*4, hidden_dim*2),
            self.discriminator_block(hidden_dim*2, hidden_dim),
            
            # Last Fully Connected Layer
            nn.Linear(hidden_dim, 1)
        )
        
        
    def forward(self, image):
        return self.discriminator(image)
    
    def discriminator_block(self, input_dim, output_dim):
        """
        LeakyReLU is for avoiding dying relu problem, since we are
        leaking our activations a bit, we tend to avoid dying relu.
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )




if __name__=="__main__":
    generator = Generator()
    print(generator)
    discriminator = Discriminator()
    print(discriminator)