import pandas as pd
import random
import yaml

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