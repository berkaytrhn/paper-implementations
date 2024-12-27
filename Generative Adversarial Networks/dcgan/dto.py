from typing import Protocol
from dataclasses import dataclass, fields
from abc import ABC


class ConfigurationClass:
    def __init__(self, cfg: dict):
        for field, value in zip(fields(self), cfg.values()):
            setattr(self, field.name, value)


@dataclass
class TrainConfiguration(ConfigurationClass):
    """ Train hyperparameters dto"""
    device: str
    noise_dimension: int
    image_channel: int
    learning_rate: float
    epochs: int
    batch_size: int
    print_every: int
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        
        
@dataclass    
class DatasetConfiguration(ConfigurationClass):
    """ Dataset Configuration dto"""
    train_set: str
    test_set: str
    train_set_length: int
    test_set_length: int
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

@dataclass    
class LoggingConfiguration(ConfigurationClass):
    """ Logging Configuration dto"""
    directory: str 
    sub_directory: str
    model_name: str
        
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
    
@dataclass    
class ModelSaveConfiguration(ConfigurationClass):
    """ Model Sacing Configuration dto"""
    results: str
    save_directory: str
    name: str
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)