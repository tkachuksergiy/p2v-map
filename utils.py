import yaml
import torch
from loguru import logger

def log_gpu_memory(logger):
    device = torch.device('cuda:0')
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB
    cached = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"GPU memory allocated: {allocated:.2f} MB")
    logger.info(f"GPU memory cached: {cached:.2f} MB")

def read_yaml(x: str) -> dict:
    """
    Read yaml files
        x: name of the file
    """
    with open(x, "r") as con:
        config = yaml.safe_load(con)
    return config