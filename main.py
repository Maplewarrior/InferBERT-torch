import pdb
import os
import pandas as pd
from utils.misc import load_config
from utils.modeling import InferBERT
from utils.causal_analysis import CausalAnalyser
def main():
    CFG = load_config('configs/liverfailure_config.yaml')
    # print("Config:\n", CFG)
    # print("CWD: ", os.getcwd())

    # CA = CausalAnalyser('configs/liverfailure_config.yaml')
    # CA.get_probabilities()
    df_root = pd.read_csv('experiments/reproduction/outputs/liverfailure/causality_output/root.csv')
    print(df_root.loc[df_root['value'].isin(['Acetaminophen', 'Death', '18-39', 'larger than 100 MG', 'Female'])])
    pdb.set_trace()

if __name__ == '__main__':
    main()