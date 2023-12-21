import pandas as pd
import random
import yaml
import torch

ps = [0.2, 0.5]
# gammas = [5.0, 3.0]
gammas = [3.0, 1.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, size_average=False, device=None):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
def load_config(path : str) -> dict:
    """
    Loads and parses a YAML configuration file
    """
    with open(path, 'r', encoding='utf-8') as yamlfile:
        cfg = yaml.safe_load(yamlfile)

    return cfg

def get_calibrated_subset(df, positive_frac=0.1):
        """
        Function that returns a the maximum subset of a dataset with a fraction of "positive_frac" observations that are positive.
        """
        pos_samples = df.loc[df['label'] == 1]
        neg_samples = df.loc[df['label'] == 0]
        N = len(df)
        n_positive = len(pos_samples)
        n_negative = len(df) - n_positive
        true_pos_frac = n_positive / N

        if positive_frac < true_pos_frac: # subsample positive class
            N_pos_sub = int(n_negative * positive_frac / (1 - positive_frac))
            subset_idxs = random.sample(list(pos_samples.index), N_pos_sub) + neg_samples.index.tolist()
        else: # subsample negative class
            N_neg_sub = int(n_positive * (1-positive_frac) / positive_frac)
            subset_idxs = random.sample(list(neg_samples.index), N_neg_sub) + pos_samples.index.tolist()
        
        # filter input data and return
        return df.iloc[subset_idxs].reset_index(drop=True)