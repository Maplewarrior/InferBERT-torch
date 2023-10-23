import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

class ADRDataset(Dataset):
    def __init__(self, df : pd.DataFrame, CFG : dict) -> None:
        self.df = df
        self.max_seq_len = CFG['data']['max_seq_len']
        self.lowercase = CFG['data']['is_lowercase']
        self.tokenizer = AlbertTokenizer.from_pretrained(CFG['model']['model_version'])

    def __getitem__(self, idx):
        input = self.df.loc[idx, 'sentence'] #+ ' <MASK>'
        label = self.df.loc[idx, 'label']
        # tokenize data
        inputs = self.tokenizer(input,
                       padding='max_length',
                       truncation=True,
                       max_length=self.max_seq_len,
                       return_tensors='pt')

        label = torch.tensor(label, dtype=torch.float32) # convert label to float tensor (needed for BCEWtihLogitLoss)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), inputs['token_type_ids'].squeeze(0), label

    def __len__(self):
        return len(self.df)