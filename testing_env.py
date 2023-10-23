import pandas as pd
from transformers import AlbertModel, AlbertTokenizer
from utils.dataset import ADRDataset
from utils.misc import load_config
from utils.modeling import build_model
from utils.trainer import Trainer
import pdb
from torch.utils.data import DataLoader
import torch


def main():
    cfg = load_config('configs/liverfailure_config.yaml')
    # model = build_model(cfg)
    # inferbert_trainer = Trainer('configs/liverfailure_config.yaml')
    # inferbert_trainer.evaluate()

def test_dataloader():
    cfg = load_config('configs/liverfailure_config.yaml')
    inferbert_trainer = Trainer('configs/liverfailure_config.yaml')
    train_dataloader = inferbert_trainer.dataloader_train
    train_dataset = inferbert_trainer.dataloader_train.dataset
    train_df = inferbert_trainer.dataloader_train.dataset.df

    idx = 0
    ids, _, _, _ = train_dataset.__getitem__(idx)
    tokenizer = AlbertTokenizer.from_pretrained(cfg['model']['model_version'])
    decoded_ids = tokenizer.decode(ids)
    item = train_df.iloc[idx]['sentence']
    print(f'Item from df: {item}')
    print(f"Decoded input: {decoded_ids}")
    pdb.set_trace()
    



if __name__ == '__main__':
    main()
    # test_dataloader()