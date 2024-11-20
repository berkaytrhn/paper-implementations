import argparse
from argparse import Namespace
from config import Config

def load_dataset():
    pass 


def configure_hyper_parameters():
    pass


def build_model():
    pass





def main(args: Namespace):
    cfg = Config(args.cfg).config
    
    
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    logging_cfg = cfg["logging"]
    model_cfg = cfg["model"]
    
    # TODO: Complete implementation and config management




if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        prog='LeNet5 Train',
        description='LeNet5 Training Process',
        epilog='Text at the bottom of help')
    
    
    parser.add_argument("-c", "--cfg", default="./config.yml", required=False, help="Config Yaml path need to be provided!")
    
    args = parser.parse_args()
    main(args)