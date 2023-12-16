import pandas as pd
from transformers import AlbertModel, AlbertTokenizer
from utils.dataset import ADRDataset
from utils.misc import load_config
from utils.modeling import build_model
from utils.trainer import Trainer
import pdb
from torch.utils.data import DataLoader
import torch
from utils.calibration_test import CalibrationAnalyser


def main():
    cfg = load_config('configs/liverfailure_config.yaml')
    model = build_model(cfg)
    test_ipt = 'Patient (Female, 40-64 years old) took Duloxetine Hydrochloride with equal or smaller than 100 MG to treat DEPRESSION leading to Death, Hospitalization.'
    # test_ipt = 'Patient (Female, 18-35 years old) took Duloxetine Hydrochloride leading to Death.'

    df_dummy = pd.DataFrame.from_dict({'sentence': [test_ipt], 'label':[1]})
    dataloader_dummy = DataLoader(ADRDataset(df_dummy, cfg), batch_size=1)
    
    _x = next(iter(dataloader_dummy))
    mean, var = model.uncertainty_est_inference(input_ids=_x[0], attention_mask=_x[1], token_type_ids=_x[2])
    print(f'Mean probability {mean[0]}\n Variance: {var[0]}')
    pdb.set_trace()
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
    
def run_calibration_test(config_path: str = 'configs/tramadol_config.yaml'):
    CA = CalibrationAnalyser(config_path)
    CA.test_calibration(frac_steps=0.05)



if __name__ == '__main__':
    # run_calibration_test()
    df = pd.read_csv('experiments/reproduction/outputs/tramadol/calibration_results.csv')
    print(df)
    # main()
    # test_dataloader()