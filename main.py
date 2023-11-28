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
    parser.add_argument('--causal_predictions', action='store_true',
                        help='Flag to get "causal predictions"')
    parser.add_argument('--causal_inference', action='store_true',
                        help='Flag to run "causal inferenc"')
    
    args = parser.parse_args()
    config_path = args.config
    # CFG = load_config(config_path)
    # print("Config:\n", CFG)
    # print("CWD: ", os.getcwd())

    if args.train:
        inferbert_trainer = Trainer(config_path)
        inferbert_trainer.train()

    if args.causal_predictions: 
        CA = CausalAnalyser(config_path)
        CA.get_probabilities()

    if args.causal_inference:
        CA = CausalAnalyser(config_path)
        CA.run_causal_inference()

    # df_root = pd.read_csv('experiments/reproduction/outputs/liverfailure/causality_output/root.csv')
    # print(df_root.loc[df_root['value'].isin(['Acetaminophen', 'Death', '18-39', 'larger than 100 MG', 'Female'])])

if __name__ == '__main__':
    main()
