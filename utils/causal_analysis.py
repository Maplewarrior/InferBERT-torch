import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer
from utils.dataset import ADRDataset
from utils.misc import load_config
from utils.modeling import build_model
import pdb
from utils.causal_inference import causal_inference, generate_causal_tree
class CausalAnalyser:
    def __init__(self, cfg_path) -> None:
        self.CFG = load_config(cfg_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device depending on availability
        self.__init_model()
        self.__init_data()

    def __init_model(self):
        self.model = build_model(self.CFG).to(self.device)
        self.tokenizer = AlbertTokenizer.from_pretrained(self.CFG['model']['model_version'])

    def __init_data(self):
        self.inference_df = pd.read_csv(self.CFG['causal_inference']['prediction']['input_file_path'])
        self.result_dict = {'probability': [],
                            'sentence': [],
                            'actual_label': []}
        
        # check whether uncertainty estimation is performed
        if self.CFG['causal_inference']['prediction']['do_uncertainty_est'] == True:
            self.result_dict['mean'] = self.result_dict['probability'].pop()
            self.result_dict['var'] = []

        self.dataloader_predict = DataLoader(ADRDataset(df=self.inference_df, CFG=self.CFG), 
                                                        batch_size=self.CFG['causal_inference']['prediction']['batch_size'], 
                                                        num_workers=self.CFG['causal_inference']['prediction']['num_workers'],
                                                        shuffle=False)

    def decode_ids(self, input_ids):
        # decode inputs
        decoded = []
        for ids in input_ids:
            decoded.append(self.tokenizer.decode(ids).split('[SEP]')[0].strip('[CLS]'))
        return decoded

    def get_probabilities(self):
        self.model.eval()
        with torch.no_grad():
            for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader_predict):
                input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                out = self.model(input_ids, attn_masks, token_type_ids) # get predictions
                sentences = self.decode_ids(input_ids=input_ids) # extract input sentences
                ### log results ###
                if input_ids.size(0) == 1: # batch_size = 1
                    self.result_dict['probability'].append(out['probs'].item())
                    self.result_dict['sentence'].append(sentences)
                    self.result_dict['actual_label'].append(int(labels.item()))
                else: # batch_size > 1
                    self.result_dict['probability'].extend([e.item() for e in out['probs']])
                    self.result_dict['sentence'].extend(sentences)
                    self.result_dict['actual_label'].extend([int(e.item()) for e in labels])
                
                if max(1, i) % (len(self.dataloader_predict.dataset) // 20) == 0:
                    print(f'Iter: {i}/{len(self.dataloader_predict.dataset)}')

        # save prediction file
        df_prediction = pd.DataFrame.from_dict(self.result_dict)
        df_prediction.to_csv(self.CFG['causal_inference']['prediction']['output_file_path'])
        print("Saved pred prediction file to {path}!".format(path=self.CFG['causal_inference']['prediction']['output_file_path']))
    
    def get_uncertainties(self):
        self.model.train() # set in train to ensure dropout is applied
        with torch.no_grad():
            for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader_predict):
                input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                means, vars = self.model.uncertainty_est_inference(input_ids, attn_masks, token_type_ids) # get predictions
                sentences = self.decode_ids(input_ids=input_ids) # extract input sentences
                ### log results ###
                if input_ids.size(0) == 1: # batch_size = 1
                    self.result_dict['mean'].append(means)
                    self.result_dict['var'].append(vars)
                    self.result_dict['sentence'].append(sentences)
                    self.result_dict['actual_label'].append(int(labels.item()))
                else: # batch_size > 1
                    self.result_dict['mean'].extend(means)
                    self.result_dict['var'].extend(vars)
                    self.result_dict['sentence'].extend(sentences)
                    self.result_dict['actual_label'].extend([int(e.item()) for e in labels])
                
                if max(1, i) % (len(self.dataloader_predict.dataset) // 20) == 0:
                    print(f'Iter: {i}/{len(self.dataloader_predict.dataset)}')

        # save prediction file
        df_prediction = pd.DataFrame.from_dict(self.result_dict)
        df_prediction.to_csv(self.CFG['causal_inference']['prediction']['uncertainty_file_path'])
        print("Saved pred prediction file to {path}!".format(path=self.CFG['causal_inference']['prediction']['uncertainty_file_path']))

    def run_causal_inference(self):
        # run inference at a root level
        causal_inference(self.CFG['causal_inference']['analysis']['probability_file_path'],
                         self.CFG['causal_inference']['analysis']['feature_file_path'],
                         self.CFG['causal_inference']['analysis']['output_dir'])
        # run inference at a tree level
        generate_causal_tree(self.CFG['causal_inference']['analysis']['probability_file_path'],
                         self.CFG['causal_inference']['analysis']['feature_file_path'],
                         self.CFG['causal_inference']['analysis']['output_dir'])
