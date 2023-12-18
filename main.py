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
    


