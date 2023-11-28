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

    # Load your DataFrame
    # # df_part = pd.read_csv('experiments/reproduction/outputs/liverfailure/causality_output/root.csv')
    # df_part = pd.read_csv('experiments/reproduction/outputs/tramadol/causality_output/root.csv')
    # print(df_part['value'])
    # # df_part = df_part.loc[df_part['value'].isin(['Acetaminophen', 'Death', '18-39', 'larger than 100 MG', 'Female'])]
    # df_part = df_part.loc[df_part['value'].isin(['SUICIDE ATTEMPT', '40-64', 'Hydrocodone Bitartrate', 'Drug abuse', 'Male'])]

    # # Custom formatting function
    # def format_float(value, decimals):
    #     return f"{value:.{decimals}f}"

    # # Apply custom formatting to each column
    # df_part['z score'] = df_part['z score'].apply(lambda x: format_float(x, 2))
    # df_part['probability of do value'] = df_part['probability of do value'].apply(lambda x: format_float(x, 2))
    # df_part['probability of not do value'] = df_part['probability of not do value'].apply(lambda x: format_float(x, 2))
    # df_part['probability difference'] = df_part['probability difference'].apply(lambda x: format_float(x, 2))

    # # Format 'p value' in scientific notation
    # df_part['p value'] = df_part['p value'].apply(lambda x: '{:.2e}'.format(x))

    # # Convert to LaTeX
    # print(df_part.to_latex(index=False))


