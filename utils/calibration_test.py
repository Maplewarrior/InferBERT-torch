import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer
from utils.misc import load_config, get_calibrated_subset
from utils.dataset import ADRDataset
from utils.modeling import build_model

class CalibrationAnalyser:
    def __init__(self, cfg_path) -> None:
        self.CFG = load_config(cfg_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device depending on availability
        self.__init_model()
        self.__init_data()

    def __init_model(self):
        self.model = build_model(self.CFG).to(self.device)
        self.tokenizer = AlbertTokenizer.from_pretrained(self.CFG['model']['model_version'])

    def __init_data(self):
        self.df_test = pd.read_csv(self.CFG['data']['test_path'])
        self.result_dict = {'mean_probability': [],
                            'positive_fraction': []}

        
    def decode_ids(self, input_ids):
        # decode inputs
        decoded = []
        for ids in input_ids:
            decoded.append(self.tokenizer.decode(ids).split('[SEP]')[0].strip('[CLS]'))
        return decoded
    
    def test_calibration(self, frac_steps=0.05):
        fractions = [e / (1/frac_steps) for e in range(1, int(1 / frac_steps+1))]
        for frac in fractions:
            print(f'Fraction: {frac}')
            df_sub = get_calibrated_subset(self.df_test, positive_frac=frac)
            assert abs(len(df_sub.loc[df_sub['label'] == 1]) / len(df_sub) - frac) < 0.02, 'Calibration went wrong...'

            dataloader = DataLoader(ADRDataset(df=get_calibrated_subset(self.df_test, positive_frac=frac), CFG=self.CFG), 
                                                        batch_size=self.CFG['causal_inference']['prediction']['batch_size'], 
                                                        num_workers=self.CFG['causal_inference']['prediction']['num_workers'],
                                                        shuffle=True)
            
            predictions = []
            for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(dataloader):
                input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                out = self.model(input_ids, attn_masks, token_type_ids) # get predictions
                # sentences = self.decode_ids(input_ids=input_ids) # extract input sentences

                ### log results ###
                if input_ids.size(0) == 1: # batch_size = 1
                    predictions.append(out['probs'].item())
                else: # batch_size > 1
                    predictions.extend([e.item() for e in out['probs']])
                
                if max(1, i) % (len(dataloader) // 10) == 0:
                    print(f'Iter: {i}/{len(dataloader)}')
            self.result_dict['mean_probability'].append(torch.mean(torch.tensor(predictions), dtype=torch.float).item())
            self.result_dict['positive_fraction'].append(frac)

        calibration_df = pd.DataFrame.from_dict(self.result_dict)
        save_dir = self.CFG['training']['out_dir']
        calibration_df.to_csv(f'{save_dir}/calibration_results.csv')