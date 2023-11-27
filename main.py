import pdb
import os
import pandas as pd
from utils.misc import load_config
from utils.modeling import InferBERT
from utils.trainer import Trainer
from utils.causal_analysis import CausalAnalyser
import argparse

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument("--config", type=str, default="configs/liverfailure_config_ALF.yaml", 
                        help="Config to use")
    parser.add_argument('--train', action='store_true',
                        help='Flag to start model training')
    parser.add_argument('--causal_analysis', action='store_true',
                        help='Flag to start "causal analysis"')
    
    args = parser.parse_args()
    config_path = args.config
    # CFG = load_config(config_path)
    # print("Config:\n", CFG)
    # print("CWD: ", os.getcwd())

    if args.train:
        inferbert_trainer = Trainer(config_path)
        inferbert_trainer.train()

    if args.causal_analysis: 
        CA = CausalAnalyser(config_path)
        CA.get_probabilities()

    # df_root = pd.read_csv('experiments/reproduction/outputs/liverfailure/causality_output/root.csv')
    # print(df_root.loc[df_root['value'].isin(['Acetaminophen', 'Death', '18-39', 'larger than 100 MG', 'Female'])])

if __name__ == '__main__':
    main()