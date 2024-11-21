
# Utility 
import argparse
from argparse import Namespace
import os
from tqdm import tqdm 
from datetime import datetime


# Custom 
from config import Config
from dto import TrainConfiguration, DatasetConfiguration, LoggingConfiguration, ModelSaveConfiguration
from model import LeNet5

# Torch
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics import Accuracy
from torchmetrics.metric import Metric

class Train:
    
    # Dataset
    train_set: Dataset=None
    validation_set: Dataset=None
    test_set: Dataset=None
    
    # Configurations
    data_cfg: DatasetConfiguration=None
    train_cfg: TrainConfiguration=None
    logging_cfg: LoggingConfiguration=None
    model_cfg: ModelSaveConfiguration=None
    
    
    # Model
    model: LeNet5=None
    optimizer: Optimizer=None
    criterion: CrossEntropyLoss=None
    metric: Metric=None
    
    # Logging
    summary_logger: SummaryWriter=None
    
    # Dataloaders
    train_dataloader: DataLoader=None
    validation_dataloader: DataLoader=None
    test_dataloader: DataLoader=None
    
    

    def __init__(self, config: Config) -> None:
        cfg = config.config
        self.data_cfg = DatasetConfiguration(cfg["data"])
        self.train_cfg = TrainConfiguration(cfg["train"])
        self.logging_cfg = LoggingConfiguration(cfg["logging"])
        self.model_cfg = ModelSaveConfiguration(cfg["model"])

    def load_dataset(self) -> None:
        composed_transforms = transforms.Compose([
            transforms.ToTensor(), # performs scaling by default for image datasets between range(0-1)
            # scale between -1 and 1 to increse model performance in terms of tanh activation
            transforms.Normalize((0.5,), (0.5,)) 
        ]) 
        
        # load train-validation set
        train_val_set = MNIST(
            self.data_cfg.train_set, 
            train=True, 
            transform=composed_transforms, 
            download=True
        )
    
        # load test set  
        self.test_set = MNIST(
            self.data_cfg.test_set, 
            train=False, 
            transform=composed_transforms, 
            download=True
        )

        # split train-val into train set and validation set
        self.train_set, self.validation_set = torch.utils.data.random_split(
            dataset=train_val_set, 
            lengths=[
                self.data_cfg.train_set_length, 
                self.data_cfg.test_set_length
            ]
        )
         
    def configure_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_set, 
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=8)

        self.validation_dataloader = DataLoader(
            self.validation_set, 
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=8)

        self.test_dataloader = DataLoader(
            self.test_set, 
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=8)

    def build_model(self):
        self.model=LeNet5().to(self.train_cfg.device)
        self.optimizer = SGD(self.model.parameters(), self.train_cfg.learning_rate) # Adam is way way better then SGD
        self.criterion =  CrossEntropyLoss()
        self.metric = Accuracy(task='multiclass', num_classes=10).to(self.train_cfg.device)

    def configure_logging(self):
        self.summary_logger = SummaryWriter(
            os.path.join(
                self.logging_cfg.directory,
                datetime.now().strftime("%d.%m.%Y"),
                self.logging_cfg.sub_directory,
                self.logging_cfg.model_name
        )
)
    
    def train(self):
        train_dataloader_length = len(self.train_dataloader)
        val_dataloader_length = len(self.validation_dataloader)
            
        print("Training...")
        for epoch in range(self.train_cfg.epochs):
            
            # init losses for this epoch
            train_loss, train_accuracy = 0.0,0.0
            # train loop
            for X_train, y_train in tqdm(self.train_dataloader):
                # activate train mode, not batchNorm or Dropout for this model but convention
                self.model.train()
                
                # to gpu if available
                X_train = X_train.to(self.train_cfg.device)
                y_train = y_train.to(self.train_cfg.device)
                
                # zeroing gradients
                self.optimizer.zero_grad()
                
                # prediction
                pred = self.model(X_train)
                
                
                # calculation of loss
                loss = self.criterion(pred, y_train)
                train_loss += loss
                
                acc = self.metric(pred, y_train)
                train_accuracy+=acc
                
                # print("*******")        
                # backpropagation operation over loss
                loss.backward()
                
                # update params according to optimizer algorithm and model parameters
                self.optimizer.step()        
            
            # average loss and accuracy on batch
            train_loss /= train_dataloader_length
            train_accuracy /= train_dataloader_length
                
                
            
            # validation loop
            validation_loss, validation_accuracy = 0.0,0.0
            # evaluation mode, no grad, high efficiency
            self.model.eval()
            print("Validation...")
            with torch.inference_mode():
                for X_validation, y_validation in tqdm(self.validation_dataloader):
                
                    # to gpu if available
                    X_validation = X_validation.to(self.train_cfg.device)
                    y_validation = y_validation.to(self.train_cfg.device)
                    
                    # prediction
                    pred = self.model(X_validation)
                    
                    # calculating loss for val
                    loss = self.criterion(pred, y_validation)
                    validation_loss += loss
                    
                    # calculate accuracy using lightning metrics and add for current epoch
                    acc = self.metric(pred, y_validation)
                    validation_accuracy+=acc
                    
                # average loss and accuracy on batch
                validation_loss/=val_dataloader_length
                validation_accuracy/=val_dataloader_length
                
            # SummaryWriter for losses
            self.summary_logger.add_scalars(
                main_tag="Losses",
                tag_scalar_dict={
                    "train/loss": train_loss,
                    "validation/loss": validation_loss
                },
                global_step=epoch
            )
            # SummaryWriter for accuracies
            self.summary_logger.add_scalars(
                main_tag="Accuracies",
                tag_scalar_dict={
                    "train/accuracy": train_accuracy,
                    "validation/accuracy": validation_accuracy
                },
                global_step=epoch
            )
            
            print(f" Epoch: {epoch} -- Train Loss: {train_loss: .3f} -- Train Acc : {train_accuracy: .3f} -- Val Loss: {validation_loss: .3f} -- Val Acc: {validation_accuracy: .3f} ")

    def test(self):
        test_dataloader_length = len(self.test_dataloader)

        # evaluation mode
        self.model.eval()
        test_loss, test_accuracy = 0.0,0.0
        with torch.inference_mode():
            for X_test, y_test in tqdm(self.test_dataloader):
                
                X_test = X_test.to(self.train_cfg.device) 
                y_test = y_test.to(self.train_cfg.device) 
                
                # test prediction
                pred = self.model(X_test)
                
                # calculating loss for test set
                loss = self.criterion(pred, y_test)
                test_loss += loss
                
                # calculate accuracy using lightning metrics and add for current epoch
                acc = self.metric(pred, y_test)
                test_accuracy+=acc

                
            # average loss and accuracy on batch
            test_loss/=test_dataloader_length
            test_accuracy/=test_dataloader_length
            
        print(f" Test Loss: {test_loss: .3f} -- Test Acc : {test_accuracy: .3f}")

        # SummaryWriter for losses
        self.summary_logger.add_scalars(
            main_tag="Losses",
            tag_scalar_dict={
                "test/loss": test_loss,
            }
        )

        # SummaryWriter for accuracies
        self.summary_logger.add_scalars(
            main_tag="Accuracies",
            tag_scalar_dict={
                "test/accuracy": test_accuracy
            }
        )








def main(args: Namespace):
    
    
    cfg = Config(args.cfg)
    
    trainer = Train(cfg)
    
    trainer.load_dataset()
    trainer.configure_dataloaders()
    trainer.configure_logging()
    trainer.build_model()
    trainer.train()
    trainer.test()
    


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(
        prog='LeNet5 Train',
        description='LeNet5 Training Process')
    
    
    parser.add_argument("-c", "--cfg", default="./config.yml", required=False)
    
    args = parser.parse_args()
    main(args)