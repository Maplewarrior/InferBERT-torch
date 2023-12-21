import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

class ADRDataset(Dataset):
    def __init__(self, df : pd.DataFrame, CFG : dict) -> None:
        self.CFG = CFG
        self.df = df
        self.max_seq_len = CFG['data']['max_seq_len']
        self.lowercase = CFG['data']['is_lowercase']
        self.tokenizer = AlbertTokenizer.from_pretrained(CFG['model']['model_version'])

    def __getitem__(self, idx):
        input = self.df.loc[idx, 'sentence']
        label = self.df.loc[idx, 'label']
        # tokenize data
        inputs = self.tokenizer(input,
                       padding='max_length',
                       truncation=True,
                       max_length=self.max_seq_len,
                       return_tensors='pt')
        # float tensor needed for BCEWtihLogitLoss and int64 needed for focal loss
        label = torch.tensor(label, dtype=torch.float32) if not self.CFG['model']['use_focal_loss'] else torch.tensor(label, dtype=torch.int64)  
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), inputs['token_type_ids'].squeeze(0), label

    def __len__(self):
        return len(self.df)