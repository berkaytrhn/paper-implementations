from dto import InferenceConfiguration
import argparse
from config import Config
import cv2 as cv
import torch
from model import Generator
import os

class Inference:
    
    
    def __init__(self, config: InferenceConfiguration):
        cfg = config.config
        self.inference_cfg = InferenceConfiguration(cfg["parameters"])
    
    
    def _get_noise(self):
        # create normally distributed noise data for generator
        return torch.randn(
            self.inference_cfg.number_of_images, 
            self.inference_cfg.noise_dimension,
            device=self.inference_cfg.device
        )
    
    def load_model(self):
        self.model: torch.nn.Module = torch.load(
            self.inference_cfg.model
        ).to(self.inference_cfg.device)
        
        assert isinstance(self.model, Generator), "Model is not an instance of Generator"
    
    @torch.no_grad()
    def generate_image(self):
        noise = self._get_noise()
        self.output_images = self.model(noise)
        
        
    def post_process(self):
        """
        Postprocess and saving generated images to specified directory
        """
        
        # if not exists
        if not os.path.exists(self.inference_cfg.output_dir):
            os.makedirs(self.inference_cfg.output_dir)
        
        for i, img in enumerate(self.output_images):
            print(f"Saving image {i}")
            cv.imwrite(
                f"{self.inference_cfg.output_dir}/image_{i}.png",
                img.to(torch.device("cpu")).numpy().transpose(1, 2, 0) * 255
            )
                
def main(args: argparse.Namespace):
    
    cfg = Config(args.cfg)
    
    inference = Inference(cfg)
    inference.load_model()
    inference.generate_image()
    inference.post_process()


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(
        prog='DCGAN Train',
        description='DCGAN Training Process')
    
    
    parser.add_argument("-c", "--cfg", default="./inference.yml", required=False)
    
    args = parser.parse_args()
    main(args)