from torch import nn

class LeNet5(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Input -> (batch_size,1,28,28)
            # Layer 1
            nn.Conv2d(
                in_channels=1, 
                out_channels=6,
                kernel_size=5,
                padding=2, # padding as 2 for achieving expected size
                stride=1), # -> (batch_size,6,28,28)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # -> (batch_size,6,14,14)
            
            # Layer 2
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0), # -> (batch_size,16,10,10)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # (batch_size,16,5,5)
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(), # (batch_size,400)
            nn.Linear(in_features=400, out_features=120), # (batch_size,120)
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84), # (batch_size,84)
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10), # (batch_size,10)
        )
    
    def forward(self, x):
        return self.fully_connected(self.feature_extractor(x))
    
    
        