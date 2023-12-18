import pdb
import os
import pandas as pd
from utils.misc import load_config
from utils.modeling import InferBERT
from utils.trainer import Trainer
from utils.causal_analysis import CausalAnalyser
import argparse
from utils.modeling import build_model
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument("--config", type=str, default="configs/liverfailure_config_ALF.yaml", 
                        help="Config to use")
    parser.add_argument('--train', action='store_true',
                        help='Flag to start model training')
    # Train multiple experiments. Used for robustness evalutation
    parser.add_argument("--num_train", type=int, default=1, help="Number of experiments with same parameters to run for train")
    parser.add_argument("--num_ca", type=int, default=1, help="Number of experiments with same parameters to run for causal analysis. Requires that --train has been run first with the same number of experiments")
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
        num_experiments = int(args.num_train)
        print(f"Training {num_experiments} models ... ")
        if num_experiments > 1:
            for i in range(num_experiments):
                num_exp = i+1
                inferbert_trainer = Trainer(config_path)

                # Update config
                conf = inferbert_trainer.CFG
                new_out_dir = f"{conf['training']['out_dir']}_{num_exp}"
                conf["training"]["out_dir"] = new_out_dir
                conf["training"]["model_path"] = f"{new_out_dir}/model_weights.pt"
                conf["training"]["train_results_path"] = f"{new_out_dir}/logs/train_log.json"
                conf["training"]["val_results_path"] = f"{new_out_dir}/logs/val_log.json"
                conf["training"]["val_results_path"] = f"{new_out_dir}/logs/test_log.json"
                inferbert_trainer.CFG = conf

                print(f"Running experiment {num_exp}/{num_experiments}")
                inferbert_trainer.train()
                print("Done")
        else:
            inferbert_trainer = Trainer(config_path)
            inferbert_trainer.train()

    
    if args.causal_predictions: 
        num_causa = int(args.num_ca)
        if num_causa > 1:
            # Update config
            for i in range(num_causa):
                num_exp = i+1
                CA = CausalAnalyser(config_path)

                conf = CA.CFG
                new_out_dir = f"{conf['training']['out_dir']}_{num_exp}"
                conf['causal_inference']['prediction']['output_file_path'] = f"{new_out_dir}/probability_file.csv"
                conf['model']['pretrained_ckpt'] = f"{new_out_dir}/model_weights.pt"
                conf['causal_inference']['analysis']['probability_file_path'] = f"{new_out_dir}/probability_file.csv"
                conf['causal_inference']['analysis']['output_dir'] = f"{new_out_dir}/causality_output"
                # update model
                CA.model = build_model(conf).to(device)

                CA.get_probabilities()
        else: 
            CA = CausalAnalyser(config_path)
            CA.get_probabilities()

    if args.causal_inference:


        num_causa = int(args.num_ca)
        if num_causa > 1:
            # Update config
            for i in range(num_causa):
                num_exp = i+1

                CA = CausalAnalyser(config_path)

                conf = CA.CFG
                new_out_dir = f"{conf['training']['out_dir']}_{num_exp}"
                conf['causal_inference']['prediction']['output_file_path'] = f"{new_out_dir}/probability_file.csv"
                conf['model']['pretrained_ckpt'] = f"{new_out_dir}/model_weights.pt"
                conf['causal_inference']['analysis']['probability_file_path'] = f"{new_out_dir}/probability_file.csv"
                conf['causal_inference']['analysis']['output_dir'] = f"{new_out_dir}/causality_output"
                # update model
                CA.model = build_model(conf).to(device)

                CA.run_causal_inference()
        else:
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


