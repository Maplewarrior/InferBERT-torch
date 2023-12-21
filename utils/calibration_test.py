import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer
# from utils.misc import load_config
# from utils.dataset import ADRDataset
# from utils.modeling import build_model
from misc import load_config
from dataset import ADRDataset
from modeling import build_model

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

def calculate_ECE(probs, y_true, n_bins=10):
    predictions = np.array(probs)
    # threshold=0.5
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    all_vals = len(y_true)
    bin_perf = []
    current_bin = 0

    reals_internal = []            
    predictions_internal = []

    ## compute bins (last one is extended with the remainder)
    intercept_bins = [x for x in range(1,all_vals) if x % n_bins == 0]
    remainder = all_vals % n_bins
    if len(intercept_bins) == 0:
        intercept_bins = [all_vals]
    intercept_bins[-1] += remainder

    intercept_index = 0
    for j in range(all_vals):

        if j == intercept_bins[intercept_index] and j > 0:

            if intercept_index < len(intercept_bins)-1:
                intercept_index += 1

            current_bin += 1
            equals = np.where(np.array(reals_internal) == np.array(predictions_internal))
            acc_bin = len(equals)/len(predictions_internal)

            conf_bin = np.mean(np.array(predictions_internal))
            bin_perf.append([current_bin, acc_bin, conf_bin,len(reals_internal)])

            reals_internal = [y_true[j]]
            predictions_internal = [predictions[j]]

        else:
            reals_internal.append(y_true[j])
            predictions_internal.append(predictions[j])

    ece_score_final = 0
    for bins in bin_perf:
        bin_size = bins[3]
        total = len(probs)
        partial = (bin_size/total) * np.abs(bins[1] - bins[2])
        ece_score_final += partial

    return ece_score_final


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
        self.dataloader = DataLoader(ADRDataset(df=self.df_test, CFG=self.CFG), 
                                                        batch_size=self.CFG['causal_inference']['prediction']['batch_size'], 
                                                        num_workers=self.CFG['causal_inference']['prediction']['num_workers'],
                                                        shuffle=False)
    def __set_dropout_to_train(self, m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    def test_calibration2(self, with_uncertainty_est = False):
        self.model.eval()
        if with_uncertainty_est:
            self.model.apply(self.__set_dropout_to_train)

        y_true = []
        y_pred_proba = []
        with torch.no_grad():
            for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader):
                input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                if not with_uncertainty_est:
                    out = self.model(input_ids, attn_masks, token_type_ids)['probs'] # get predictions
                else:
                    out, _ = self.model.uncertainty_est_inference(input_ids, attn_masks, token_type_ids) # get mean prediction
                
                ### log results ###
                if input_ids.size(0) == 1: # batch_size = 1
                    y_pred_proba.append(out.item())
                    y_true.append(labels.item())
                else: # batch_size > 1
                    y_pred_proba.extend([e.item() for e in out])
                    y_true.extend([e.item() for e in labels])
            
                if max(1, i) % (len(self.dataloader) // 5) == 0:
                    print(f"iter: {i}/{len(self.dataloader)}")

                
        ECE = calculate_ECE(y_pred_proba, y_true, n_bins=3)
        print(f'ECE score: {ECE}')
        fop, mpv = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot([0, 1], [0, 1], linestyle='--', color = "black")
        plt.plot(mpv, fop, marker='.', color = "red")
        
        plt.xlabel("Mean prediction value")
        plt.ylabel("Fraction of positives")
        plt.savefig(self.CFG['training']['out_dir']+"/validation_cal_{}_{}_visualization.pdf".format('all_dropout', 0.5), dpi = 300)
        plt.clf()
    
    def create_calibration_plots(self, model_paths: list[str]):
        all_preds = []
        model_names = ['BCE', 'focal loss']
        focal_loss = [False, True]
        colors = ['red', 'blue']
        for i, path in enumerate(model_paths):
            print(f'Path: {path}')
            self.CFG['model']['pretrained_ckpt'] = path
            self.CFG['model']['use_focal_loss'] = focal_loss[i]
            self.__init_model()
            y_pred_proba = []
            y_true = []
            self.model.eval()
            with torch.no_grad():
                for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader):
                    input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                    out = self.model(input_ids, attn_masks, token_type_ids)['probs'] # get predictions
                
                    ### log results ###
                    if input_ids.size(0) == 1: # batch_size = 1
                        y_pred_proba.append(out.item())
                        y_true.append(labels.item())
                    else: # batch_size > 1
                        if out.size(1) == 2:
                            y_pred_proba.extend([e[1].item() for e in out])
                        else:
                            y_pred_proba.extend([e.item() for e in out])
                        y_true.extend([e.item() for e in labels])
                all_preds.append(y_pred_proba)
        plt.figure(figsize=(6,3))

        for j in range(len(all_preds)):
            ECE = calculate_ECE(all_preds[j], y_true, n_bins=10)
            print(f'ECE score for {model_names[j]}: {ECE}')
            fop, mpv = calibration_curve(y_true, all_preds[j], n_bins=10)
            plt.plot(mpv, fop, marker='.', color = colors[j], label=model_names[j])
        
        plt.plot([0, 1], [0, 1], linestyle='--', color = "black")
        plt.xlabel("Mean prediction value")
        plt.ylabel("Fraction of positives")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.CFG['training']['out_dir']+"/Calibration_{}_visualization.pdf".format('all_methods'), dpi = 300)
        plt.clf()

if __name__ == '__main__':
    CA = CalibrationAnalyser(cfg_path='configs/tramadol_config_ALF.yaml')
    model_paths = ['experiments/reproduction/outputs/tramadol_1/model_weights.pt', 'experiments/reproduction/outputs/tramadol_focal/model_weights_31_adaptive.pt']
    CA.create_calibration_plots(model_paths)

                    