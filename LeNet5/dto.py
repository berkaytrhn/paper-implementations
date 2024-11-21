from typing import Protocol
from dataclasses import dataclass, fields
from abc import ABC


class ConfigurationClass:
    def __init__(self, cfg: dict):
        for field, value in zip(fields(self), cfg.values()):
            setattr(self, field.name, value)


@dataclass
class TrainConfiguration(ConfigurationClass):
    """ Train hyperparameters interface like structure """
    device: str
    learning_rate: float
    epochs: int
    batch_size: int
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        
        
@dataclass    
class DatasetConfiguration(ConfigurationClass):
    """ Dataset Configuration Interface like structure """
    train_set: str
    test_set: str
    train_set_length: int
    test_set_length: int
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

@dataclass    
class LoggingConfiguration(ConfigurationClass):
    """ Logging Configuration Interface like Structure """
    directory: str 
    sub_directory: str
    model_name: str
        
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
    
@dataclass    
class ModelSaveConfiguration(ConfigurationClass):
    """ Model Sacing Configuration Interface like structure"""
    save_directory: str
    name: str
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)