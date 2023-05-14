import yaml
import torch
import pandas as pd
from loguru import logger
from src.data_loader import *
from src.model import *

with open("config.yaml", "r") as con:
    config = yaml.safe_load(con)

baskets = pd.read_parquet(f"{config['paths']['data']}/market-baskets.parquet")
products = pd.read_parquet(f"{config['paths']['data']}/products.parquet")
n_products = products.shape[0]
logger.info(f"n_products = {n_products}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    p2v_data_stream = DataStreamP2V(data=baskets, **config["data"]["data_streamer"])
    dl_train, dl_valid = build_data_loader(streamer=p2v_data_stream,
                                           **config["p2v"]["data-loader"])
    p2v_model = P2V(n_products=n_products, device=device,
                                   **config["p2v"]["model"])
    p2v_trainer = TrainerP2V(model=p2v_model,
                                            train=dl_train,
                                            validation=dl_valid,
                                            device=device,
                                            path=config["paths"]["results"])
    p2v_trainer.fit(**config["p2v"]["trainer"])